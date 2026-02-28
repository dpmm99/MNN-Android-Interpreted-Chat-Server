package com.alibaba.mnnllm.android.hotspot

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.alibaba.mnnllm.android.main.MainActivity

class WebExportBridge(private val fragment: Fragment) {

    private var pendingContent: String? = null

    val createDocumentLauncher: ActivityResultLauncher<Intent> =
        fragment.registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val uri = result.data?.data ?: return@registerForActivityResult
                val content = pendingContent ?: return@registerForActivityResult
                fragment.requireContext().contentResolver.openOutputStream(uri)?.use { stream ->
                    stream.write(content.toByteArray(Charsets.UTF_8))
                }
                pendingContent = null
            }
        }

    @JavascriptInterface
    fun exportText(text: String) {
        fragment.requireActivity().runOnUiThread {
            pendingContent = text
            val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "text/plain"
                putExtra(Intent.EXTRA_TITLE, "chat-export-${java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(java.util.Date())}.txt")
            }
            createDocumentLauncher.launch(intent)
        }
    }

    // ── Notifications ────────────────────────────────────────────────────────

    @JavascriptInterface
    fun notifyFromWeb(title: String, body: String) {
        val ctx = fragment.requireContext()

        // Always post a real system notification — works foreground & background.
        ChatNotificationHelper.notify(
            context = ctx,
            title   = title.ifBlank { "New message" },
            body    = body,
            tapIntent = Intent(ctx, MainActivity::class.java)
        )

        // Bring app to front only when it is already running in the foreground
        // (mimics the old behaviour without disrupting the user when backgrounded).
        if (!isAppInBackground()) {
            fragment.requireActivity().runOnUiThread {
                fragment.requireActivity().window.decorView.requestFocus()
            }
        }
    }

    // ── Vibration ────────────────────────────────────────────────────────────

    @JavascriptInterface
    fun vibrate(patternJson: String) {
        try {
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = fragment.requireContext()
                    .getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vm.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                fragment.requireContext().getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }

            // Parse the JSON array that JS sent us, e.g. "[200,100,200]"
            val rawList = patternJson
                .trim('[', ']', ' ')
                .split(',')
                .mapNotNull { it.trim().toLongOrNull() }
            val pattern = if (rawList.isEmpty()) longArrayOf(0, 200) else {
                // Android vibration patterns must start with a silence segment.
                if (rawList.first() != 0L) longArrayOf(0L) + rawList.toLongArray()
                else rawList.toLongArray()
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(pattern, -1)
            }
        } catch (_: Exception) {}
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private fun isAppInBackground(): Boolean {
        val appProcessInfo = android.app.ActivityManager.RunningAppProcessInfo()
        android.app.ActivityManager.getMyMemoryState(appProcessInfo)
        return appProcessInfo.importance !=
               android.app.ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND
    }
}