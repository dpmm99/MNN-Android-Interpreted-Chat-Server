package com.alibaba.mnnllm.android.hotspot

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.pm.ShortcutInfoCompat
import androidx.core.content.pm.ShortcutManagerCompat
import androidx.core.graphics.drawable.IconCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.main.MainActivity
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.File
import java.net.NetworkInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "ChatServerFragment"

class ChatServerFragment : Fragment(), MainActivity.BackPressHandler {

    private lateinit var statusCard: View
    private lateinit var runningCard: View
    private lateinit var tvStatus: TextView
    private lateinit var btnStop: Button
    private lateinit var ivWifiQr: ImageView
    private lateinit var ivUrlQr: ImageView
    private lateinit var tvUrl: TextView
    private lateinit var tvUsers: TextView
    private lateinit var webView: WebView
    private lateinit var btnOpenBrowser: Button
    private lateinit var tvDebugPrompt: TextView
    private lateinit var tvDebugOutput: TextView
    private lateinit var debugPanel: View
    private lateinit var debugScroll: View
    private lateinit var tvQrWifiLabel: LabelCyclerView
    private lateinit var tvQrUrlLabel: LabelCyclerView
    private lateinit var btnStartServerHotspot: Button
    private lateinit var btnStartServerWifi: Button
    private lateinit var btnCreateShortcut: Button
    private lateinit var etHotspotSsid: android.widget.EditText
    private lateinit var etHotspotPassword: android.widget.EditText

    private var hotspotManager: LocalHotspotManager? = null
    private var serverManager: ChatServerManager? = null
    private var hotspotJob: kotlinx.coroutines.Job? = null
    private var inferenceDebugJob: kotlinx.coroutines.Job? = null

    // Job that polls for server readiness and loads the loopback URL when ready.
    private var loopbackLoadJob: Job? = null

    // Keep the generated QR bitmaps locally so we can set them on the server manager
    // even if the service bind happens slightly later than UI update.
    private var cachedWifiBitmap: Bitmap? = null
    private var cachedUrlBitmap: Bitmap? = null

    // Local loopback URL used for WebView / "open in browser"
    private var localLoopbackUrl: String? = null

    private var currentMode: String = "hotspot"
    private var lastModelId: String = ""
    private var lastConfigPath: String = ""

    // Must be registered before onStart; property initialisation satisfies that requirement.
    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            if (grants.values.all { it }) {
                pickModelAndStart()
            } else {
                Toast.makeText(
                    requireContext(),
                    R.string.chat_server_permission_denied,
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    // ActivityResult launcher used by the WebView file chooser.
    // Registered early so it's available when WebChromeClient.onShowFileChooser runs.
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null
    private val filePickerLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            val cb = fileChooserCallback
            fileChooserCallback = null
            if (cb == null) return@registerForActivityResult

            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                val intent = result.data!!
                val clip = intent.clipData
                if (clip != null && clip.itemCount > 0) {
                    val uris = Array(clip.itemCount) { i -> clip.getItemAt(i).uri }
                    cb.onReceiveValue(uris)
                } else {
                    val uri = intent.data
                    if (uri != null) {
                        cb.onReceiveValue(arrayOf(uri))
                    } else {
                        cb.onReceiveValue(null)
                    }
                }
            } else {
                cb.onReceiveValue(null)
            }
        }

    // ── Service binding ───────────────────────────────────────────────────────
    private val serviceConnection = object : android.content.ServiceConnection {
        override fun onServiceConnected(name: android.content.ComponentName, service: IBinder) {
            serverManager = (service as ChatServerService.LocalBinder).getManager()
            // Ensure the server manager gets any cached QR images we generated earlier.
            serverManager?.setCachedQrCodes(cachedWifiBitmap, cachedUrlBitmap)
            // If we already have a loopback URL queued, ensure it will be loaded when server is ready.
            localLoopbackUrl?.let { loadLoopbackWhenReady(it) }
            updateUi()
        }
        override fun onServiceDisconnected(name: android.content.ComponentName) {
            serverManager = null
            inferenceDebugJob?.cancel()
            inferenceDebugJob = null
        }
    }
    private var serviceBound = false

    private val exportBridge = WebExportBridge(this) // For export

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View = inflater.inflate(R.layout.fragment_chat_server, container, false)

    override fun onResume() {
        super.onResume()
        val prefs = requireContext().getSharedPreferences("ChatServerAutoStart", android.content.Context.MODE_PRIVATE)
        if (prefs.getBoolean("AUTO_START", false)) {
            prefs.edit().putBoolean("AUTO_START", false).apply()
            val mode   = prefs.getString("MODE", "wifi") ?: "wifi"
            val modelId = prefs.getString("MODEL_ID", null)
            val configPath = modelId?.let { com.alibaba.mnnllm.android.model.ModelUtils.getConfigPathForModel(it) }
            currentMode = mode
            if (modelId != null && configPath != null) {
                startServer(modelId, configPath)
            } else {
                // Model ID missing or config not found — fall through to normal UI
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        statusCard = view.findViewById(R.id.card_status)
        runningCard = view.findViewById(R.id.card_running)
        tvStatus = view.findViewById(R.id.tv_status)
        btnStartServerHotspot = view.findViewById(R.id.btn_start_server_hotspot)
        btnStartServerWifi = view.findViewById(R.id.btn_start_server_wifi)
        btnStop = view.findViewById(R.id.btn_stop_server)
        btnCreateShortcut = view.findViewById(R.id.btn_create_shortcut)
        ivWifiQr = view.findViewById(R.id.iv_wifi_qr)
        ivUrlQr = view.findViewById(R.id.iv_url_qr)
        tvUrl = view.findViewById(R.id.tv_chat_url)
        tvUsers = view.findViewById(R.id.tv_users)
        webView = view.findViewById(R.id.webview_chat)
        btnOpenBrowser = view.findViewById(R.id.btn_open_browser)
        debugPanel = view.findViewById(R.id.debug_panel)
        tvDebugPrompt = view.findViewById(R.id.tv_debug_prompt)
        tvDebugOutput = view.findViewById(R.id.tv_debug_output)
        tvQrWifiLabel = view.findViewById(R.id.tv_qr_wifi_label)
        tvQrUrlLabel  = view.findViewById(R.id.tv_qr_url_label)

        debugScroll = view.findViewById(R.id.debug_scroll)
        debugScroll.isNestedScrollingEnabled = true
        debugScroll.setOnTouchListener { v, event ->
            v.parent.requestDisallowInterceptTouchEvent(true)
            false
        }
        
        // Hotspot SSID / Password inputs (pre-populated from prefs)
        etHotspotSsid = view.findViewById(R.id.et_hotspot_ssid)
        etHotspotPassword = view.findViewById(R.id.et_hotspot_password)
        val hotspotPrefs = requireContext().getSharedPreferences("HotspotConfig", Context.MODE_PRIVATE)
        etHotspotSsid.setText(hotspotPrefs.getString("SSID", ""))
        etHotspotPassword.setText(hotspotPrefs.getString("PASSWORD", ""))

        btnStartServerHotspot.setOnClickListener {
            currentMode = "hotspot"
            checkAndRequestHotspotPermissions()
        }
        btnStartServerWifi.setOnClickListener {
            currentMode = "wifi"
            checkAndRequestWifiPermissions()
        }

        btnStop.setOnClickListener { stopServer() }
        btnOpenBrowser.setOnClickListener {
            // Open the loopback URL in the default browser when available.
            val url = localLoopbackUrl ?: tvUrl.text.toString()
            if (url.isNotEmpty()) {
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
            }
        }
        btnCreateShortcut.setOnClickListener {
            createFrozenShortcut()
        }
        ChatNotificationHelper.createChannel(requireContext())
        setupWebView()
        updateUi()
    }

    override fun handleBackPress(): Boolean {
        // Only allow WebView back navigation when it's internal in-page / same-origin
        // (e.g. SPA panels that pushState). Prevent navigating back to about:blank
        // or out-of-app pages before prompting to shut down the server.
        try {
            if (webView.canGoBack()) {
                val bf = webView.copyBackForwardList()
                val idx = bf.currentIndex
                if (idx > 0) {
                    val currUrl = bf.getItemAtIndex(idx).url ?: ""
                    val prevUrl = bf.getItemAtIndex(idx - 1).url ?: ""

                    // Allow goBack only when both current and previous are the local loopback origin.
                    if ((currUrl.startsWith("http://127.0.0.1") && prevUrl.startsWith("http://127.0.0.1")) || (currUrl.startsWith("https://127.0.0.1") && prevUrl.startsWith("https://127.0.0.1"))) {
                        webView.goBack()
                        return true
                    }

                    // Otherwise do not navigate the WebView back (prevents going to about:blank).
                }
            }
        } catch (e: Exception) {
            // fall through to server stop check
        }

        if (serverManager?.isRunning() == true) {
            MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.chat_server_stop_confirm_title)
                .setMessage(R.string.chat_server_stop_confirm_message)
                .setPositiveButton(R.string.chat_server_stop_and_leave) { _, _ ->
                    stopServer()
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
            return true
        }
        return false
    }

    private fun createFrozenShortcut() {
        if (!ShortcutManagerCompat.isRequestPinShortcutSupported(requireContext())) {
            Toast.makeText(context, "Shortcuts not supported on this device.", Toast.LENGTH_SHORT).show()
            return
        }

        // 1. Gather exact frozen state
        val frozenModelId = lastModelId
        val hostUser = serverManager?.getHostUser()
        val frozenUsername = hostUser?.username ?: ""
        val frozenLanguage = hostUser?.language ?: "en"
        val frozenBase64Avatar = hostUser?.avatarBase64 ?: ""

        // 2. Cache the Avatar string to a file to prevent Intent size crash
        val avatarCacheFile = File(requireContext().filesDir, "shortcut_avatar_${System.currentTimeMillis()}.txt")
        if (frozenBase64Avatar.isNotEmpty()) {
            avatarCacheFile.writeText(frozenBase64Avatar) // Write Base64 directly to disk
        }

        // 3. Create the Intent
        val intent = Intent(requireContext(), MainActivity::class.java).apply {
            action = "ACTION_START_FROZEN_SERVER"
            putExtra("FROZEN_MODE", currentMode)
            putExtra("FROZEN_MODEL_ID", frozenModelId)
            putExtra("FROZEN_USERNAME", frozenUsername)
            putExtra("FROZEN_LANGUAGE", frozenLanguage)
            if (frozenBase64Avatar.isNotEmpty()) {
                putExtra("FROZEN_AVATAR_PATH", avatarCacheFile.absolutePath)
            }
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }

        val shortcutInfo = ShortcutInfoCompat.Builder(requireContext(), "server_${System.currentTimeMillis()}")
            .setShortLabel("Chat ($currentMode)")
            .setLongLabel("Start Chat Server ($currentMode)")
            .setIcon(IconCompat.createWithResource(requireContext(), R.mipmap.ic_launcher)) // Or a custom icon
            .setIntent(intent)
            .build()

        ShortcutManagerCompat.requestPinShortcut(requireContext(), shortcutInfo, null)
        Toast.makeText(context, "Shortcut requested", Toast.LENGTH_SHORT).show()
    }

    // ── Permissions ────────────────────────────────────────────────────────────

    /**
     * Checks for the permission required by [WifiManager.startLocalOnlyHotspot]:
     *  - API 33+: NEARBY_WIFI_DEVICES
     *  - API 26–32: ACCESS_FINE_LOCATION
     *
     * Proceeds to [pickModelAndStart] immediately when already granted, otherwise
     * triggers the system permission dialog via [permissionLauncher].
     */
    private fun checkAndRequestHotspotPermissions() {
        val required = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.NEARBY_WIFI_DEVICES)
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        val missing = required.filter {
            ContextCompat.checkSelfPermission(requireContext(), it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            pickModelAndStart()
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    // Doesn't actually need any extra permissions.
    private fun checkAndRequestWifiPermissions() {
        val ip = getLocalWifiIpAddress()
        if (ip == null) {
            Toast.makeText(context, R.string.chat_server_no_wifi, Toast.LENGTH_SHORT).show()
            return
        }
        pickModelAndStart()
    }

    // ── WebView ────────────────────────────────────────────────────────────────

    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            allowContentAccess = true
        }

        webView.setOnTouchListener { v, event ->
            v.parent.requestDisallowInterceptTouchEvent(true)
            false
        }

        // Intercept page finished to inject a small shim that routes Notification() and vibrate() calls
        // to our Android bridge. The web app calls new Notification(...); we intercept.
        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                try {
                    // language=JavaScript
                    val js = """
                        (function(){
                          try{
                        // ── 1. Intercept new Notification() ──────────────────
                            const _orig = window.Notification;
                            window.Notification = function(title, options){
                              try{
                                AndroidExport.notifyFromWeb(title || '', (options && options.body) || '');
                              }catch(e){}
                              return { close: function(){} };
                            };
                            window.Notification.permission = 'granted';
                            window.Notification.requestPermission = function(cb){
                              if(cb) cb('granted');
                              return Promise.resolve('granted');
                            };

                            // ── 2. Intercept navigator.vibrate() ─────────────────
                            // navigator.vibrate is a no-op inside Android WebView; route it through the native bridge instead.
                            try {
                              Object.defineProperty(navigator, 'vibrate', {
                                configurable: true,
                                writable: true,
                                value: function(pattern) {
                                  try {
                                    // Normalise to a JSON array string so Kotlin can parse it.
                                    const p = Array.isArray(pattern)
                                      ? pattern
                                      : (typeof pattern === 'number' ? [pattern] : [200]);
                                    AndroidExport.vibrate(JSON.stringify(p));
                                  } catch(e) {}
                                  return true;
                                }
                              });
                            } catch(e) {}
                          }catch(e){}
                        })();
                    """.trimIndent()
                    view?.evaluateJavascript(js, null)
                } catch (_: Exception) {}
            }
        }

        // Expose the native bridge to JS as "AndroidExport"
        webView.addJavascriptInterface(exportBridge, "AndroidExport")

        // Provide WebChromeClient so <input type="file"> opens system picker inside the app WebView.
        webView.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                // Clear previous callback if any
                fileChooserCallback?.onReceiveValue(null)
                fileChooserCallback = filePathCallback

                // Try to use the FileChooserParams intent if available, otherwise fallback to generic ACTION_GET_CONTENT
                val pickIntent: Intent = try {
                    fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "image/*"
                        putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
                    }
                } catch (e: ActivityNotFoundException) {
                    // No activity to handle intent
                    try {
                        fileChooserCallback?.onReceiveValue(null)
                    } catch (_: Exception) {}
                    fileChooserCallback = null
                    return false
                }

                try {
                    filePickerLauncher.launch(pickIntent)
                } catch (e: Exception) {
                    try {
                        fileChooserCallback?.onReceiveValue(null)
                    } catch (_: Exception) {}
                    fileChooserCallback = null
                    return false
                }
                return true
            }
        }
    }

    // Poll and load loopback URL when ChatServerManager is running.
    private fun loadLoopbackWhenReady(url: String) {
        loopbackLoadJob?.cancel()
        loopbackLoadJob = lifecycleScope.launch(Dispatchers.Main) {
            val deadline = System.currentTimeMillis() + 11_000L // >10s
            while (isAdded && System.currentTimeMillis() <= deadline) {
                if (serverManager?.isRunning() == true) {
                    try {
                        webView.loadUrl(url)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to load loopback URL", e)
                    }
                    return@launch
                }
                delay(200L)
            }
            // timeout: server didn't start in time
            if (isAdded) {
                Toast.makeText(requireContext(), getString(R.string.chat_server_start_failed_to_bind), Toast.LENGTH_LONG).show()
                // Avoid loading a broken 127.0.0.1 page; keep blank so user can retry.
                try { webView.loadUrl("about:blank") } catch (_: Exception) {}
            }
        }
    }

    // ── Start / stop ───────────────────────────────────────────────────────────

    private fun pickModelAndStart() {
        lifecycleScope.launch {
            val models = withContext(Dispatchers.IO) {
                val state = ModelListManager.observeModelList()
                    .filterIsInstance<ModelListManager.ModelListState.Success>()
                    .first()
                state.models.filter { it.downloadedModelInfo != null || it.isLocal }
            }

            if (!isAdded) return@launch

            if (models.isEmpty()) {
                Toast.makeText(requireContext(), R.string.chat_server_no_models, Toast.LENGTH_LONG).show()
                return@launch
            }

            val names = models.map { it.displayName }.toTypedArray()
            MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.chat_server_pick_model)
                .setItems(names) { _, idx ->
                    val wrapper = models[idx]
                    val modelId = wrapper.modelItem.modelId ?: ""

                    if (modelId.isEmpty()) {
                        Toast.makeText(requireContext(), requireContext().getString(R.string.model_not_found, ""), Toast.LENGTH_LONG).show()
                        return@setItems
                    }

                    val configPath = ModelUtils.getConfigPathForModel(modelId)
                    if (configPath != null) {
                        startServer(modelId, configPath)
                    } else {
                        Toast.makeText(requireContext(), R.string.config_file_not_found, Toast.LENGTH_LONG).show()
                    }
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
        }
    }

    private fun startServer(modelId: String, configPath: String) {
        tvStatus.setText(R.string.chat_server_starting)
        btnStartServerHotspot.isEnabled = false
        btnStartServerWifi.isEnabled = false
        lastModelId = modelId
        lastConfigPath = configPath
        setBottomNavVisible(false)

        val intent = Intent(requireContext(), ChatServerService::class.java).apply {
            action = ChatServerService.ACTION_START
            putExtra(ChatServerService.EXTRA_MODEL_ID, modelId)
            putExtra(ChatServerService.EXTRA_CONFIG_PATH, configPath)
        }
        androidx.core.content.ContextCompat.startForegroundService(requireContext(), intent)
        requireContext().bindService(intent, serviceConnection, android.content.Context.BIND_AUTO_CREATE)
        serviceBound = true

        if (currentMode == "wifi") {
            val ip = getLocalWifiIpAddress() ?: run {
                tvStatus.setText(R.string.chat_server_no_wifi)
                btnStartServerHotspot.isEnabled = true
                btnStartServerWifi.isEnabled = true
                return
            }
            val connInfo = HotspotConnectionInfo(
                ssid = null,
                password = null,
                gatewayIp = ip,
                port = CHAT_SERVER_PORT,
            )
            lifecycleScope.launch(Dispatchers.Default) {
                val urlQr = QrCodeGenerator.generate(connInfo.urlQrContent)
                withContext(Dispatchers.Main) {
                    if (isAdded) showRunningStateWifi(connInfo, urlQr)
                }
            }
        } else {
            hotspotManager = LocalHotspotManager(requireContext())
            // read last-used SSID/password from prefs and pass into hotspotInfoFlow
            val hotspotPrefs = requireContext().getSharedPreferences("HotspotConfig", Context.MODE_PRIVATE)
            val prefSsid = hotspotPrefs.getString("SSID", null)
            val prefPass = hotspotPrefs.getString("PASSWORD", null)

            hotspotJob = lifecycleScope.launch(Dispatchers.IO) {
                hotspotManager!!.hotspotInfoFlow(prefSsid, prefPass)
                    .catch { e ->
                        withContext(Dispatchers.Main) {
                            if (isAdded) {
                                tvStatus.text = getString(R.string.chat_server_hotspot_failed, e.message)
                                btnStartServerHotspot.isEnabled = true
                                btnStartServerWifi.isEnabled = true
                            }
                        }
                    }
                    .collect { result ->
                        result.onSuccess { info ->
                            val connInfo = HotspotConnectionInfo(
                                ssid = info.ssid,
                                password = info.passphrase,
                                gatewayIp = info.gatewayIp,
                                port = CHAT_SERVER_PORT,
                            )
                            val wifiQr = withContext(Dispatchers.Default) { QrCodeGenerator.generate(connInfo.wifiQrContent) }
                            val urlQr  = withContext(Dispatchers.Default) { QrCodeGenerator.generate(connInfo.urlQrContent) }
                            withContext(Dispatchers.Main) {
                                if (isAdded) {
                                    // persist the successful SSID/password as "last used"
                                    requireContext().getSharedPreferences("HotspotConfig", Context.MODE_PRIVATE)
                                        .edit()
                                        .putString("SSID", info.ssid)
                                        .putString("PASSWORD", info.passphrase)
                                        .apply()

                                    showRunningState(connInfo, wifiQr, urlQr)
                                }
                            }
                        }
                        result.onFailure { e ->
                            withContext(Dispatchers.Main) {
                                if (isAdded) {
                                    tvStatus.text = getString(R.string.chat_server_hotspot_failed, e.message)
                                    btnStartServerHotspot.isEnabled = true
                                    btnStartServerWifi.isEnabled = true
                                }
                            }
                        }
                    }
            }
        }
    }

    private fun getLocalWifiIpAddress(): String? {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()
            while (interfaces.hasMoreElements()) {
                val iface = interfaces.nextElement()
                // Ignore loopback and inactive interfaces
                if (iface.isLoopback || !iface.isUp) continue

                val addresses = iface.inetAddresses
                while (addresses.hasMoreElements()) {
                    val addr = addresses.nextElement()
                    // Check if it's an IPv4 address
                    if (!addr.isLoopbackAddress && addr is java.net.Inet4Address) {
                        val ip = addr.hostAddress
                        // Prioritize standard local subnets (192.168.x.x, 10.x.x.x)
                        if (ip?.startsWith("192.168.") == true || ip?.startsWith("10.") == true || ip?.startsWith("172.") == true) {
                            return ip
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting local IP", e)
        }
        return null
    }

    private fun stopServer() {
        hotspotJob?.cancel()
        hotspotJob = null
        hotspotManager = null

        // Cancel inference debug collector so it can be restarted cleanly on next server start.
        inferenceDebugJob?.cancel()
        inferenceDebugJob = null

        loopbackLoadJob?.cancel()
        loopbackLoadJob = null
        if (serviceBound) {
            requireContext().unbindService(serviceConnection)
            serviceBound = false
        }
        requireContext().startService(
            Intent(requireContext(), ChatServerService::class.java).apply {
                action = ChatServerService.ACTION_STOP
            }
        )
        serverManager = null
        setBottomNavVisible(true)
        if (isAdded) showStoppedState()
    }

    /** Show or hide MainActivity's bottom navigation bar. */
    private fun setBottomNavVisible(visible: Boolean) { //TODO: don't use reflection for this, hahaha... but I guess it's okay for now because I'd rather not edit Alibaba's code directly if I can help it.
        val activity = activity ?: return
        try {
            val field = activity.javaClass.getDeclaredField("bottomNav")
            field.isAccessible = true
            val nav = field.get(activity) as? android.view.View ?: return
            nav.visibility = if (visible) android.view.View.VISIBLE else android.view.View.GONE
        } catch (e: Exception) {
            // bottomNav field not found or not accessible - ignore
        }
    }

    fun pinServerShortcut(mode: String) { // mode = "wifi" or "hotspot"
        if (ShortcutManagerCompat.isRequestPinShortcutSupported(requireContext())) {
        
            val intent = Intent(requireContext(), MainActivity::class.java).apply {
                action = "ACTION_START_SERVER"
                putExtra("SERVER_MODE", mode)
                // Save model settings to intent or rely on SharedPrefs fallback
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            }

            val shortcutInfo = ShortcutInfoCompat.Builder(requireContext(), "start_server_$mode")
                .setShortLabel("Chat ($mode)")
                .setLongLabel("Start Chat Server ($mode)")
                .setIcon(IconCompat.createWithResource(requireContext(), R.mipmap.ic_launcher))
                .setIntent(intent)
                .build()

            ShortcutManagerCompat.requestPinShortcut(requireContext(), shortcutInfo, null)
        }
    }

    // ── UI helpers ─────────────────────────────────────────────────────────────

    private fun showRunningState(info: HotspotConnectionInfo, wifiQr: Bitmap?, urlQr: Bitmap?) {
        tvStatus.setText(R.string.chat_server_running)
        statusCard.visibility = View.GONE
        runningCard.visibility = View.VISIBLE

        // cache locally and update the service if already bound
        ivWifiQr.setImageBitmap(wifiQr)
        ivUrlQr.setImageBitmap(urlQr)
        cachedWifiBitmap = wifiQr
        cachedUrlBitmap = urlQr
        serverManager?.setCachedQrCodes(cachedWifiBitmap, cachedUrlBitmap)

        // Keep QR and displayed URL as-is (these point to the hotspot IP).
        tvUrl.text = info.urlQrContent
        tvQrWifiLabel.setEntries(HotspotLabelCycler.buildWifiEntries(requireContext()))
        tvQrUrlLabel.setEntries(HotspotLabelCycler.buildUrlEntries(requireContext()))

        // Use loopback for the WebView and the "open in browser" action.
        // Keep port from the hotspot connection info.
        localLoopbackUrl = "http://127.0.0.1:${info.port}/"
        loadLoopbackWhenReady(localLoopbackUrl!!)

        observeConnectedCount()
        observeInferenceDebug()
    }
    
    private fun showRunningStateWifi(info: HotspotConnectionInfo, urlQr: Bitmap?) {
        tvStatus.setText(R.string.chat_server_running)
        statusCard.visibility = View.GONE
        runningCard.visibility = View.VISIBLE
        ivWifiQr.visibility = View.GONE          // no Wi-Fi credentials to share
        tvQrWifiLabel.visibility = View.GONE

        // cache locally and update the service if already bound
        ivUrlQr.setImageBitmap(urlQr)
        cachedWifiBitmap = null
        cachedUrlBitmap = urlQr
        serverManager?.setCachedQrCodes(cachedWifiBitmap, cachedUrlBitmap)

        tvUrl.text = info.urlQrContent
        tvQrUrlLabel.setEntries(HotspotLabelCycler.buildUrlEntries(requireContext(), 1))
        localLoopbackUrl = "http://127.0.0.1:${info.port}/"
        loadLoopbackWhenReady(localLoopbackUrl!!)
        observeConnectedCount()
        observeInferenceDebug()
    }

    private fun showStoppedState() {
        statusCard.visibility = View.VISIBLE
        runningCard.visibility = View.GONE
        btnStartServerHotspot.isEnabled = true
        btnStartServerWifi.isEnabled = true
        ivWifiQr.visibility = View.VISIBLE
        tvQrWifiLabel.visibility = View.VISIBLE
        tvStatus.setText(R.string.chat_server_idle)
        
        // clear cached images locally (server is already halted)
        ivWifiQr.setImageBitmap(null)
        ivUrlQr.setImageBitmap(null)
        cachedWifiBitmap = null
        cachedUrlBitmap = null
        
        tvUrl.text = ""
        tvQrWifiLabel.setEntries(emptyList())
        tvQrUrlLabel.setEntries(emptyList())
        // Clear loopback URL when stopped so the browser button falls back to the displayed QR URL (if any).
        localLoopbackUrl = null
        loopbackLoadJob?.cancel()
        loopbackLoadJob = null
        webView.loadUrl("about:blank")
        debugPanel.visibility = View.GONE
    }

    private fun updateUi() {
        if (serverManager?.isRunning() == true) {
            statusCard.visibility = View.GONE
            runningCard.visibility = View.VISIBLE
            observeConnectedCount()
            observeInferenceDebug()
        } else {
            showStoppedState()
        }
    }

    private fun observeConnectedCount() {
        val mgr = serverManager ?: return
        lifecycleScope.launch {
            mgr.connectedCountFlow.collect { count ->
                if (isAdded) {
                    tvUsers.text = resources.getQuantityString(
                        R.plurals.chat_server_user_count, count, count
                    )
                }
            }
        }
    }

    private fun updateUserCount() {
        val count = serverManager?.getConnectedUserCount() ?: 0
        tvUsers.text = resources.getQuantityString(R.plurals.chat_server_user_count, count, count)
    }

    private fun observeInferenceDebug() {
        if (inferenceDebugJob?.isActive == true) return   // already collecting
        val mgr = serverManager ?: return
        inferenceDebugJob = lifecycleScope.launch {
            mgr.inferenceDebugFlow.collect { state ->
                if (!isAdded) return@collect
                debugPanel.visibility = View.VISIBLE
                tvDebugPrompt.text = state.prompt
                tvDebugOutput.text = state.partialOutput
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        if (serviceBound) {
            requireContext().unbindService(serviceConnection)
            serviceBound = false
        }
        loopbackLoadJob?.cancel()
        loopbackLoadJob = null
        inferenceDebugJob?.cancel()
        inferenceDebugJob = null
        webView.destroy()
    }
}