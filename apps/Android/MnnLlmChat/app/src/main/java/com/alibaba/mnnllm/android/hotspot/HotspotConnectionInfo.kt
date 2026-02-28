package com.alibaba.mnnllm.android.hotspot

data class HotspotConnectionInfo(
    val ssid: String?,
    val password: String?,
    val gatewayIp: String,
    val port: Int,
) {
    /** Standard Wi-Fi QR format recognized natively by Android and iOS cameras. */
    val wifiQrContent: String
        get() {
            // Build fields only when values are present. Null or empty ssid/password are supported.
            val fields = mutableListOf<String>()
            val s = ssid.escapeWifiQr()
            val p = password.escapeWifiQr()

            if (s.isNotEmpty()) {
                fields.add("S:$s")
            }

            if (p.isNotEmpty()) {
                // When a password exists, mark the auth type as WPA (common default).
                fields.add("T:WPA")
                fields.add("P:$p")
            }

            return if (fields.isEmpty()) {
                // Minimal valid QR for an unspecified network
                "WIFI:;;"
            } else {
                "WIFI:${fields.joinToString(";")};;"
            }
        }

    /** URL the user should navigate to after joining the hotspot. */
    val urlQrContent: String
        get() = "http://$gatewayIp:$port"

    companion object {
        /**
         * Characters that must be escaped in the WIFI: QR format:
         * \ ; , " :
         *
         * Support nullable receivers so callers can safely call `.escapeWifiQr()` on nullable values.
         */
        private fun String?.escapeWifiQr(): String =
            this?.replace("\\", "\\\\")
                ?.replace(";", "\\;")
                ?.replace(",", "\\,")
                ?.replace("\"", "\\\"")
                ?.replace(":", "\\:")
                ?: ""
    }
}
