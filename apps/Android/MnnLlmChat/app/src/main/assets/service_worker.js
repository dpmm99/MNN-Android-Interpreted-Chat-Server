// Receives forwarded SSE messages from the main page via BroadcastChannel
// and shows OS-level notifications even when the tab is backgrounded.
// No Push API / no internet required.

const BC_NAME = 'mnn_chat_sw';

// Map of userId → latest notification tag, used for grouping rapid messages.
// We keep this in SW memory (reset on SW restart, which is fine).

const bc = new BroadcastChannel(BC_NAME);

bc.onmessage = e => {
    const { type, msg, myUserId } = e.data;

    if (type === 'incoming' && msg) {
        // Never notify for our own messages (the page forwards all messages,
        // so we must filter here too).
        if (msg.userId === myUserId) return;

        self.registration.showNotification(msg.username, {
            body: msg.text,
            tag: 'msg-' + msg.userId,   // groups rapid messages from same sender
            renotify: true,             // still buzz even if replacing same tag
            silent: false,
        });
    }

    if (type === 'ping') {
        // Keepalive ping from the page — no-op, just prevents SW from sleeping.
    }
};

// Required: activate immediately without waiting for old SW to be released.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));