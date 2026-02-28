package com.alibaba.mnnllm.android.hotspot

import android.content.Context
import android.net.wifi.SoftApConfiguration
import android.net.wifi.WifiManager
import android.net.wifi.WifiManager.LocalOnlyHotspotCallback
import android.net.wifi.WifiManager.LocalOnlyHotspotReservation
import android.net.wifi.WifiSsid
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.net.NetworkInterface

private const val TAG = "LocalHotspotManager"

/**
 * Manages the lifecycle of a [WifiManager.startLocalOnlyHotspot] session.
 *
 * Requires:
 *  - Manifest permissions: ACCESS_FINE_LOCATION and ACCESS_WIFI_STATE
 *  - API 26+ (Android 8.0) for startLocalOnlyHotspot
 *  - Location services enabled on the device
 *
 * The returned [Flow] emits a [Result] with [HotspotInfo] on success, or a
 * failure if the hotspot could not be started. It completes when the hotspot
 * is stopped (collector cancels or [LocalOnlyHotspotReservation.close] is called).
 */
class LocalHotspotManager(context: Context) {

    data class HotspotInfo(
        val ssid: String,
        val passphrase: String,
        /** The gateway/host IP that connecting clients will reach this device on. */
        val gatewayIp: String,
    )

    private val wifiManager =
        context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

    /**
     * Starts the local-only hotspot and emits [HotspotInfo] once it's ready.
     *
     * The hotspot stays alive as long as the flow is collected. Cancel the
     * coroutine (or let the flow complete) to release the reservation.
     */
    fun hotspotInfoFlow(ssid: String? = null, password: String? = null): Flow<Result<HotspotInfo>> = callbackFlow {
        var reservation: LocalOnlyHotspotReservation? = null

        val callback = object : LocalOnlyHotspotCallback() {
            override fun onStarted(r: LocalOnlyHotspotReservation) {
                reservation = r
                val info = extractHotspotInfo(r)
                if (info != null) {
                    trySend(Result.success(info))
                } else {
                    trySend(Result.failure(IllegalStateException("Could not determine hotspot credentials")))
                    close()
                }
            }

            override fun onStopped() {
                Log.d(TAG, "Local hotspot stopped")
                close()
            }

            override fun onFailed(reason: Int) {
                trySend(Result.failure(IllegalStateException("Hotspot failed to start, reason=$reason")))
                close()
            }
        }

        // If running on TIRAMISU+ and caller provided SSID/password, attempt to apply a SoftApConfiguration.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && (!ssid.isNullOrBlank() || !password.isNullOrBlank())) {
            try {
                // SoftApConfiguration.Builder is hidden (@SystemApi) before API 36, requiring reflection to compile. Allegedly...
                val builderClass = Class.forName("android.net.wifi.SoftApConfiguration\$Builder")
                val builder = builderClass.getDeclaredConstructor().newInstance()

                if (!ssid.isNullOrBlank()) {
                    val wifiSsid = WifiSsid.fromBytes(ssid.toByteArray(Charsets.UTF_8))  // API 33+
                    val setWifiSsidMethod = builderClass.getMethod("setWifiSsid", WifiSsid::class.java)
                    setWifiSsidMethod.invoke(builder, wifiSsid)
                }

                if (!password.isNullOrBlank()) {
                    // Use WPA3 when a passphrase is provided
                    val setPassphraseMethod = builderClass.getMethod("setPassphrase", String::class.java, Int::class.java)
                    setPassphraseMethod.invoke(builder, password, SoftApConfiguration.SECURITY_TYPE_WPA3_SAE)
                }

                val buildMethod = builderClass.getMethod("build")
                val cfg = buildMethod.invoke(builder)

                try {
                    // WifiManager.setSoftApConfiguration is also @SystemApi
                    val setConfigMethod = wifiManager.javaClass.getMethod("setSoftApConfiguration", SoftApConfiguration::class.java)
                    setConfigMethod.invoke(wifiManager, cfg)
                } catch (e: Exception) {
                    Log.w(TAG, "Unable to set SoftApConfiguration (may require system privileges)", e)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to build/apply SoftApConfiguration", e)
            }
        }

        wifiManager.startLocalOnlyHotspot(callback, Handler(Looper.getMainLooper()))

        awaitClose {
            reservation?.close()
            reservation = null
        }
    }

    private fun extractHotspotInfo(reservation: LocalOnlyHotspotReservation): HotspotInfo? {
        return try {
            val ssid: String?
            val passphrase: String?

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val config = reservation.softApConfiguration
                @Suppress("DEPRECATION")
                ssid = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    config.wifiSsid?.toString()?.trim('"')
                } else {
                    config.ssid
                }
                passphrase = config.passphrase
            } else {
                throw UnsupportedOperationException("API 30+ is required to access hotspot configuration")
            }

            if (ssid == null || passphrase == null) {
                Log.w(TAG, "Hotspot SSID or passphrase is null")
                return null
            }

            val gatewayIp = findHotspotGatewayIp() ?: DEFAULT_GATEWAY_IP
            HotspotInfo(ssid = ssid, passphrase = passphrase, gatewayIp = gatewayIp)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract hotspot info", e)
            null
        }
    }

    /**
     * Attempts to find the IP address of the hotspot interface by enumerating
     * network interfaces. Falls back to [DEFAULT_GATEWAY_IP] if not found.
     */
    private fun findHotspotGatewayIp(): String? {
        return try {
            val interfaces = NetworkInterface.getNetworkInterfaces() ?: return null
            for (iface in interfaces.asSequence()) {
                if (!iface.isUp || iface.isLoopback) continue
                val name = iface.name
                // Hotspot interfaces are typically named wlan0, ap0, swlan0, etc.
                if (!name.startsWith("wlan") && !name.startsWith("ap") && !name.startsWith("swlan")) continue
                for (addr in iface.inetAddresses.asSequence()) {
                    if (addr.isLoopbackAddress || addr.isLinkLocalAddress) continue
                    val ip = addr.hostAddress ?: continue
                    if (ip.contains(':')) continue // Skip IPv6
                    Log.d(TAG, "Found hotspot interface $name with IP $ip")
                    return ip
                }
            }
            null
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enumerate network interfaces", e)
            null
        }
    }

    companion object {
        /** Fallback gateway IP used when the interface cannot be determined programmatically. */
        const val DEFAULT_GATEWAY_IP = "192.168.43.1"
    }
}