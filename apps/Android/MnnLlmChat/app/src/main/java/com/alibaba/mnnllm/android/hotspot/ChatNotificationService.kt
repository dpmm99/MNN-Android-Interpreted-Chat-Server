package com.alibaba.mnnllm.android.hotspot

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat

object ChatNotificationHelper {

    private const val CHANNEL_ID   = "chat_messages"
    private const val CHANNEL_NAME = "Chat Messages"
    private var nextId = 1000

    /** Call once (e.g. in Application.onCreate or Activity.onCreate). */
    fun createChannel(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                val channel = NotificationChannel(
                    CHANNEL_ID,
                    CHANNEL_NAME,
                    NotificationManager.IMPORTANCE_HIGH          // shows as heads-up
                ).apply {
                    description = "Incoming chat messages"
                    enableVibration(true)
                    vibrationPattern = longArrayOf(0, 200, 100, 200)
                }
                mgr.createNotificationChannel(channel)
            }
        }
    }

    /**
     * Post a system notification.  Safe to call from any thread.
     *
     * @param tapIntent  Optional intent launched when the user taps the notification.
     *                   Pass your MainActivity intent to bring the app to the front.
     */
    fun notify(
        context: Context,
        title: String,
        body: String,
        tapIntent: Intent? = null
    ) {
        val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        val pendingIntent: PendingIntent? = tapIntent?.let {
            it.addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            PendingIntent.getActivity(
                context,
                0,
                it,
                PendingIntent.FLAG_UPDATE_CURRENT or
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                            PendingIntent.FLAG_IMMUTABLE else 0
            )
        }

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_email)   // replace with your own icon
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setVibrate(longArrayOf(0, 200, 100, 200))
            .apply { pendingIntent?.let { setContentIntent(it) } }
            .build()

        mgr.notify(nextId++, notification)
    }
}
