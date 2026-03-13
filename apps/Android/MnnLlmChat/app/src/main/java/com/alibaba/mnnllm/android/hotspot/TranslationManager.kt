package com.alibaba.mnnllm.android.hotspot

import android.util.Log
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.PriorityBlockingQueue
import java.util.Locale
import java.security.MessageDigest

private const val TAG = "TranslationManager"

/** Snapshot of what the LLM is currently doing, for debug display. */
data class InferenceDebugState(
    val prompt: String = "",
    val partialOutput: String = "",
    val idle: Boolean = true,
)

/**
 * Manages a priority queue of translation tasks processed sequentially by one LLM session.
 *
 * Priority (lowest number = highest priority):
 *   0 → UI text translation for a new language
 *   1 → newly sent messages (FIFO)
 *   2 → retranslation with more context
 *   3 → historical messages for a newcomer (newest→oldest)
 */
class TranslationManager(
    private val llmSession: LlmSession,
    private val onTranslationReady: suspend (messageId: String, language: String, text: String, retranslationCount: Int, previousVersions: List<String>) -> Unit,
    private val onUiTranslationChunkReady: suspend (language: String, translations: Map<String, String>) -> Unit,
) {
    private val queue = PriorityBlockingQueue<TranslationTask>()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val enqueuedKeys = mutableSetOf<String>()

    /**
     * Tracks enqueued UI translation work at the granularity of individual string keys,
     * so callers can quickly check whether a specific "$language:$uiKey" is already in
     * flight without scanning the entire queue.
     *
     * Entries are added when a [TranslationTask.UiTranslationTask] chunk is accepted by
     * [enqueue] and removed when that chunk's task key leaves [enqueuedKeys] (i.e. when
     * the worker dequeues it).  Because a single chunk covers multiple UI keys, all keys
     * in the chunk share the same lifetime.
     *
     * Guarded by the same [enqueuedKeys] lock.
     */
    private val enqueuedUiKeysByLang = mutableSetOf<String>() // "$language:$uiKey"

    /**
     * Chunks that have permanently failed all retry attempts (for this app session).
     * Key format for UI text: "$language:<sha1-of-sorted-keys>". We use a hash of the chunk's keys
     * (the set of UI map keys being translated) rather than the chunk index because
     * chunks can be re-computed at runtime and indexes may change. Guarded by
     * [enqueuedKeys] lock. Intentionally NOT cleared in [stop] so that persistently
     * broken chunks are not retried as long as the same server instance is running.
     */
    private val permanentlyFailedChunks = mutableSetOf<String>()

    // Helper: build a stable permanent key for a UI chunk based on language + sorted keys hash
    private fun uiChunkPermanentKey(language: String, keys: Set<String>): String {
        val sorted = keys.toList().sorted().joinToString("|")
        val sha = MessageDigest.getInstance("SHA-1").digest(sorted.toByteArray(Charsets.UTF_8))
        val hex = sha.joinToString("") { "%02x".format(it) }
        return "$language:$hex"
    }

    // ── Debug state ────────────────────────────────────────────────────────────
    private val _debugFlow = MutableStateFlow(InferenceDebugState())
    val debugFlow: StateFlow<InferenceDebugState> = _debugFlow.asStateFlow()

    init {
        scope.launch { processQueue() }
    }

    fun enqueue(task: TranslationTask): Boolean {
        val added = synchronized(enqueuedKeys) {
            // Reject any UI chunk that has permanently failed all retries.
            if (task is TranslationTask.UiTranslationTask) {
                val permanentKey = uiChunkPermanentKey(task.language, task.chunk.keys)
                if (permanentlyFailedChunks.contains(permanentKey)) {
                    Log.d(TAG, "Ignoring enqueue for permanently failed UI chunk: $permanentKey (chunkIndex=${task.chunkIndex})")
                    return@synchronized false
                }
            }
            if (enqueuedKeys.add(task.key)) {
                queue.offer(task)
                // Track per-UI-key enqueue state so callers can cheaply query individual keys.
                if (task is TranslationTask.UiTranslationTask) {
                    for (uiKey in task.chunk.keys) {
                        enqueuedUiKeysByLang.add("${task.language}:$uiKey")
                    }
                }
                true
            } else {
                false
            }
        }
        return added
    }

    /**
     * Returns true if a translation for [uiKey] in [language] is already enqueued
     * (i.e. a UI chunk containing it is in the queue and not yet processed).
     * Thread-safe; acquires the [enqueuedKeys] lock.
     */
    fun isUiKeyEnqueued(language: String, uiKey: String): Boolean =
        synchronized(enqueuedKeys) { enqueuedUiKeysByLang.contains("$language:$uiKey") }

    fun stop() {
        scope.cancel()
        queue.clear()
        synchronized(enqueuedKeys) {
            enqueuedKeys.clear()
            enqueuedUiKeysByLang.clear()
        }
        permanentlyFailedChunks.clear(); // Allow more retries if the server is restarted, even if the app isn't.
        _debugFlow.value = InferenceDebugState() // reset to idle
    }

    private suspend fun processQueue() {
        while (true) {
            val task = try {
                // Blocking take - waits until a task is available
                kotlinx.coroutines.withContext(Dispatchers.IO) { queue.take() }
            } catch (e: InterruptedException) {
                break
            }
            synchronized(enqueuedKeys) {
                enqueuedKeys.remove(task.key)
                // Remove per-UI-key tracking entries now that the chunk is being processed.
                if (task is TranslationTask.UiTranslationTask) {
                    for (uiKey in task.chunk.keys) {
                        enqueuedUiKeysByLang.remove("${task.language}:$uiKey")
                    }
                }
            }
            try {
                processTask(task)
            } catch (e: Exception) {
                Log.e(TAG, "Translation task failed: ${task.key}", e)
            }
        }
    }

    private suspend fun processTask(task: TranslationTask) {
        when (task) {
            is TranslationTask.MessageTranslationTask -> translateMessage(task)
            is TranslationTask.UiTranslationTask -> translateUiChunk(task)
            is TranslationTask.HistoryTranslationTask -> {
                // Reuse MessageTranslationTask logic
                translateMessage(
                    TranslationTask.MessageTranslationTask(task.messageId, task.oldLanguage, task.language)
                )
            }
        }
    }

    /**
     * Translate a single chat message. If contextCount > 0, the previous
     * [contextCount] messages in the conversation history are included in the
     * prompt to aid contextual accuracy.
     */
    private suspend fun translateMessage(task: TranslationTask.MessageTranslationTask) {
        val session = llmSession
        session.setKeepHistory(false)

        val languageName = languageNameFor(task.language)
        val oldLanguageName = languageNameFor(task.oldLanguage)
        val prompt = buildTranslationPrompt(task, oldLanguageName, languageName)

        val result = StringBuilder()

        // Repetition tracking
        val recentTokens = ArrayDeque<String>(24)
        val linesSeenThisInference = mutableMapOf<String, Int>()
        var currentLine = StringBuilder()
        var shouldStop = false

        // Start a new debug cycle -> overwrite prior debug state
        _debugFlow.value = InferenceDebugState(prompt = prompt, partialOutput = "", idle = false)

        try {
            session.generate(prompt, emptyMap(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress == null) return true

                    result.append(progress)
                    currentLine.append(progress)

                    // ── Token repetition check (last 24 tokens) ──────────────
                    val token = progress
                    recentTokens.addLast(token)
                    if (recentTokens.size > 24) recentTokens.removeFirst()
                    val tokenRepeatCount = recentTokens.count { it == token }
                    if (tokenRepeatCount >= 8) {
                        Log.w(TAG, "Stopping inference: token '$token' repeated $tokenRepeatCount times in last 24 tokens")
                        shouldStop = true
                    }

                    // ── Line repetition check ────────────────────────────────
                    if (progress.contains('\n')) {
                        val parts = progress.split('\n')
                        // Complete the current line with text before the first newline
                        currentLine.append(parts[0])
                        val completedLine = currentLine.toString().trim()
                        if (completedLine.isNotEmpty()) {
                            val lineCount = (linesSeenThisInference[completedLine] ?: 0) + 1
                            linesSeenThisInference[completedLine] = lineCount
                            if (lineCount >= 5) {
                                Log.w(TAG, "Stopping inference: line '$completedLine' repeated $lineCount times")
                                shouldStop = true
                            }
                        }
                        // Start fresh for whatever came after the last newline
                        currentLine = StringBuilder(parts.last())
                    }

                    // Update debug display (still in-progress)
                    _debugFlow.value = InferenceDebugState(
                        prompt = prompt,
                        partialOutput = result.toString(),
                        idle = false,
                    )

                    return shouldStop
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "LLM generate failed for ${task.messageId}", e)
            // Preserve the most-recent prompt and partial output, but mark idle.
            _debugFlow.value = InferenceDebugState(
                prompt = prompt,
                partialOutput = result.toString(),
                idle = true
            )
            return
        }

        // Completed: mark idle but keep the completed partial output visible
        _debugFlow.value = InferenceDebugState(
            prompt = prompt,
            partialOutput = result.toString(),
            idle = true
        )

        var translated = result.toString().trim()

        // Strip <think>...</think> prefix if the output starts with one
        if (translated.startsWith("<think>")) {
            val closeTag = "</think>"
            val closeIdx = translated.indexOf(closeTag)
            if (closeIdx >= 0) {
                translated = translated.substring(closeIdx + closeTag.length).trim()
            }
        }

        if (translated.isNotEmpty()) {
            val previous = if (task.previousTranslation != null) listOf(task.previousTranslation) else emptyList()
            onTranslationReady(task.messageId, task.language, translated, task.contextCount, previous)
        }
    }

    private fun buildTranslationPrompt(
        task: TranslationTask.MessageTranslationTask,
        oldLanguageName: String,
        languageName: String,
    ): String {
        val messageText = ChatServerManager.instance?.getMessageText(task.messageId) ?: ""
        return if (task.contextCount > 0) {
            val contextMessages = ChatServerManager.instance
                ?.getContextMessages(task.messageId, task.contextCount)
                ?.joinToString("\n") { "${it.username}: ${it.text}" }
                ?: ""
            "Translate the last message from $oldLanguageName to $languageName. Reply with ONLY the translation.\n\nConversation context:\n$contextMessages\n\nLast message to translate:\n$messageText"
        } else {
            "Translate the following text from $oldLanguageName to $languageName. Reply with ONLY the translation, nothing else.\n$messageText"
        }
    }

/** Translate a single chunk of UI strings to the given language. */
    private suspend fun translateUiChunk(task: TranslationTask.UiTranslationTask) {
        val session = llmSession
        session.setKeepHistory(false)

        val languageName = languageNameFor(task.language)
        val uiStrings = task.chunk.entries.joinToString("\n") { (k, v) -> "$k: $v" }
        
        // Build JSON schema based on chunk keys
        val schemaKeys = task.chunk.keys.joinToString(", ") { "\"$it\"" }
        val jsonSchema = """{
            "type": "object",
            "properties": {
                ${task.chunk.keys.joinToString(",\n                ") { "\"$it\": {\"type\": \"string\", \"minLength\": 1}" }}
            },
            "required": [$schemaKeys],
            "additionalProperties": false
        }""".trimIndent()
        
        val prompt = if (task.isRepair) {
            """The following JSON is malformed. Fix it and output ONLY valid JSON.
It should be a single flat object with as many lower_snake_case keys as there are labels/messages.
Do not add any explanations or formatting.
Input:
${task.rawBadJson}"""
        } else {
            """Translate each UI string to $languageName for a chat application.
Output ONLY a JSON object with the same keys and translated values. No other text.
Input:
$uiStrings"""
        }

        val result = StringBuilder()

        // Repetition tracking
        val recentTokens = ArrayDeque<String>(24)
        val linesSeenThisInference = mutableMapOf<String, Int>()
        var currentLine = StringBuilder()
        var shouldStop = false

        // Start a new debug cycle -> overwrite prior debug state
        _debugFlow.value = InferenceDebugState(prompt = prompt, partialOutput = "", idle = false)

        // Enable JSON-constrained decoding with schema for UI translations
        session.setJsonMode(true)
        session.setJsonSchema(jsonSchema)
        Log.d(TAG, "UI translation schema: $jsonSchema")

        // Apply a nonzero temperature for retry attempts that aren't JSON-repair tasks.
        if (task.temperature > 0f) {
            llmSession.updateConfig("""{"temperature": ${task.temperature}}""")
        }
        try {
            session.generate(prompt, emptyMap(), object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (progress == null) return true

                    result.append(progress)
                    currentLine.append(progress)

                    // ── Token repetition checks ───────────────────────────────
                    val token = progress
                    recentTokens.addLast(token)
                    if (recentTokens.size > 24) recentTokens.removeFirst()
                    val tokenRepeatCount = recentTokens.count { it == token }
                    if (tokenRepeatCount >= 8) {
                        Log.w(TAG, "Stopping UI inference: token '$token' repeated $tokenRepeatCount times")
                        shouldStop = true
                    }
                    // This catches a few more common looping patterns (1/2/1/2 or 1/2/3/1/2/3 cycles).
                    val lastTen = recentTokens.takeLast(10).toList()
                    if (lastTen.size == 10) {
                        val distinct = lastTen.toSet()
                        if (distinct.size in 1..3) {
                            Log.w(TAG, "Stopping UI translation: last 10 tokens contain only ${distinct.size} distinct token(s): ${distinct.joinToString(", ")}")
                            shouldStop = true
                        }
                    }


                    // ── Line repetition check ────────────────────────────────
                    if (progress.contains('\n')) {
                        val parts = progress.split('\n')
                        currentLine.append(parts[0])
                        val completedLine = currentLine.toString().trim()
                        if (completedLine.isNotEmpty()) {
                            val lineCount = (linesSeenThisInference[completedLine] ?: 0) + 1
                            linesSeenThisInference[completedLine] = lineCount
                            if (lineCount >= 5) {
                                Log.w(TAG, "Stopping UI inference: line '$completedLine' repeated $lineCount times")
                                shouldStop = true
                            }
                        }
                        currentLine = StringBuilder(parts.last())
                    }

                    _debugFlow.value = InferenceDebugState(
                        prompt = prompt,
                        partialOutput = result.toString(),
                        idle = false,
                    )

                    return shouldStop
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "UI translation failed for ${task.language} chunk ${task.chunkIndex}", e)
            // Preserve prompt + partial output but mark idle
            _debugFlow.value = InferenceDebugState(
                prompt = prompt,
                partialOutput = result.toString(),
                idle = true
            )
            return
        } finally {
            // Always restore greedy decoding after a temperature-modified attempt.
            if (task.temperature > 0f) {
                llmSession.updateConfig("""{"temperature": 0.0}""")
            }
            // Disable JSON-constrained decoding and clear schema after generation
            session.clearJsonSchema()
            session.setJsonMode(false)
        }

        // Completed: mark idle but keep the completed partial output visible
        _debugFlow.value = InferenceDebugState(
            prompt = prompt,
            partialOutput = result.toString(),
            idle = true
        )

        var raw = result.toString().trim()

        // Strip <think>...</think> prefix if present
        if (raw.startsWith("<think>")) {
            val closeTag = "</think>"
            val closeIdx = raw.indexOf(closeTag)
            if (closeIdx >= 0) {
                raw = raw.substring(closeIdx + closeTag.length).trim()
            }
        }

        try {
            val cleanedJsonStr = raw
                // poorly-quoted keys:  'key': " or `key": ' or "key': ` or anything like that -> "key": "...
                .replace(Regex("""(?<=[{,\s])['"`]([a-z_]+)['"`]\s*:\s*['"`]"""), "\"$1\": \"")
                // unquoted keys:  key:  -> "key":
                .replace(Regex("""(?<=[{,\s])([a-z_]+)\s*:\s*"""), "\"$1\": ")
                // bad ending quote on value: "can't',\n or "you`\n} or "me',\n} or similar - MUST have a \n to be captured, but comma or not
                .replace(Regex("""['"`](,?\s*\n)"""), "\"$1")
                // trailing comma before closing curly bracket (any amount of whitespace in between)
                .replace(Regex(""",\s*\}"""), "}")
                // wrong style of double quotes
                .replace(Regex("""[“”]"""), "\"")

            val jsonStart = cleanedJsonStr.indexOf('{')
            val jsonEnd = cleanedJsonStr.lastIndexOf('}')
            if (jsonStart < 0 || jsonEnd < jsonStart) throw Exception("No JSON brackets found")
            
            val finalJson = cleanedJsonStr.substring(jsonStart, jsonEnd + 1)
            
            val type = object : com.google.gson.reflect.TypeToken<Map<String, String>>() {}.type
            val map = com.google.gson.Gson().fromJson<Map<String, String>>(finalJson, type)
            
            Log.d(TAG, "UI chunk translation parsed successfully: ${map.size} keys for ${task.language} chunk ${task.chunkIndex}")
            onUiTranslationChunkReady(task.language, map)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse UI translations for ${task.language} chunk ${task.chunkIndex}", e)
            
            // Check a few things to ensure it actually *tried* to make a JSON object
            val bracketCount = raw.count { it == '{' || it == '}' }
            val hasEscapedQuotes = raw.contains("\\\"")
            val permanentKey = uiChunkPermanentKey(task.language, task.chunk.keys)

            if ((bracketCount >= 3 || (bracketCount == 2 && hasEscapedQuotes)) && task.repairTriesAllowed > 0) {
                // Output looks like malformed JSON — attempt a targeted repair pass.
                Log.w(TAG, "Malformed UI JSON. Enqueuing repair task for chunkIndex=${task.chunkIndex} (permanentKey=$permanentKey)...")
                enqueue(task.copy(
                    repairTriesAllowed = task.repairTriesAllowed - 1,
                    isRepair = true,
                    rawBadJson = raw
                ))
            } else if (task.repairTriesAllowed > 0) {
                // Output doesn't resemble complete JSON; retry with nonzero temperature
                // so the model samples a different path.
                Log.w(TAG, "UI translation produced no JSON. Enqueuing temperature retry (t=0.9) for chunkIndex=${task.chunkIndex} (permanentKey=$permanentKey)")
                enqueue(task.copy(
                    repairTriesAllowed = task.repairTriesAllowed - 1,
                    temperature = 0.9f
                ))
            } else {
                // All retry paths exhausted. Permanently suppress future attempts for
                // this chunk so we don't waste power retrying in the same session.
                Log.e(TAG, "All retries exhausted for UI chunk chunkIndex=${task.chunkIndex} (permanentKey=$permanentKey). Permanently blacklisting.")
                synchronized(enqueuedKeys) {
                    permanentlyFailedChunks.add(permanentKey)
                }
            }
        }
    }

    companion object {
        /** Full English name for a BCP-47 / ISO 639-1 language code. */
        fun languageNameFor(code: String): String {
            val trimmed = code.lowercase().take(2)
            val name = Locale(trimmed).getDisplayLanguage(Locale.ENGLISH)
            // getDisplayLanguage returns the code itself if it can't resolve it
            return name.ifBlank { code }
        }

        /** English UI strings used as the source for translation. NOTE: These are in a deliberate order based on when they should be seen by a new user. */
        val UI_STRINGS_EN = mapOf(
            "setup_username" to "Choose a username",
            "setup_username_hint" to "Enter your name",
            "setup_avatar" to "Profile picture (optional)",
            "btn_choose_photo" to "Choose photo",
            "btn_join" to "Join chat",
            "chat_title" to "Chat",
            "chat_placeholder" to "Type a message…",
            "btn_send" to "Send",
            "translating" to "Translating…",
            "connection_lost" to "Connection lost. Reconnecting…",
            "qc_title" to "Quick chat",
            "qc_cat_conversation" to "Conversation",
            "qc_cat_about_you" to "About You",
            "qc_cat_right_here" to "Right Here",
            "qc_cat_common_ground" to "Common Ground",
            "qc_cat_spend_time" to "Spend Time",
            "qc_cat_wrapping_up" to "Wrapping Up",
            "qc_pronounce_my_name"     to "Hello. I will start by pronouncing my name out loud.",
            "qc_translation_unclear"   to "Can you explain that? The translation was unclear.",
            "qc_say_more_simply"       to "Could you say that more simply?",
            "qc_think_app_interesting" to "I think this app is interesting.",
            "qc_do_you_have_internet"  to "Do you have internet access?",
            "ctx_reply" to "Reply", // The context menu options that become available soonest
            "ctx_copy_message" to "Copy message",
            "ctx_view_original" to "View original",
            "ctx_retranslate" to "Retranslate with more context",
            "connected_users" to "Connected users", // The next several you have to look at the user chat panel to even see
            "btn_enable_notifs" to "Enable Notifications",
            "btn_disable_notifs" to "Disable Notifications",
            "btn_share_link" to "Share Chat Link",
            "connection_restored" to "Connected",
            "you" to "You",
            "server_host" to "Host",
            "btn_export" to "Export chat", // Near the end of a chat
            "ctx_cancel_reply" to "Cancel reply", // Only appears after selecting "Reply" on a message, plus you generally can't see it on a phone at all
            "ctx_view_translation" to "View translation", // Only after you hit 'view original', which requires a translation first
            "retranslating" to "Retranslating…", // Much lower priority than 'translating' because it requires the use of a context menu option.
            "ctx_prev_translation" to "View previous translation", // Only after receiving a retranslation
            "ctx_view_latest" to "View latest translation", // Only after receiving a retranslation AND hitting view previous
            "qc_where_do_you_live"     to "Where do you live now?",
            "qc_how_long_been_here"    to "How long have you been here?",
            "qc_visiting_or_local"     to "Are you visiting or do you live here?",
            "qc_what_do_for_work"      to "What do you do for work?",
            "qc_what_are_hobbies"      to "What are your hobbies?",
            "qc_have_you_traveled"     to "Have you traveled much?",
            "qc_what_languages_speak"  to "What other languages can you speak?",
            "qc_whats_in_profile_pic"  to "What's in your profile picture?",
            "qc_do_you_enjoy_work"     to "Do you enjoy your work?",
            "qc_what_brings_you_here"  to "What brings you here today?",
            "qc_first_time_here"       to "Is this your first time here?",
            "qc_how_did_you_hear"      to "How did you hear about this place?",
            "qc_here_alone_friends"    to "Are you here alone or with friends?",
            "qc_how_long_staying"      to "How long are you staying?",
            "qc_what_is_area_known_for" to "What is this area known for?",
            "qc_what_think_of_place"   to "What do you think of this place?",
            "qc_do_you_know_anyone"    to "Do you know anyone else here?",
            "qc_what_music_like"       to "What kind of music do you like?",
            "qc_what_do_weekends"      to "What do you like to do on weekends?",
            "qc_what_passionate_about" to "What are you passionate about?",
            "qc_what_are_you_reading"  to "What have you been reading or watching lately?",
            "qc_want_grab_drink"       to "Would you like to get a drink together?",
            "qc_want_grab_food"        to "Would you like to get something to eat?",
            "qc_want_keep_walking"     to "Do you want to keep walking together?",
            "qc_want_join_group"       to "Would you like to join our group?",
            "qc_know_good_place_nearby" to "Do you know a good place nearby?",
            "qc_nice_meeting_you"      to "It was really nice meeting you.",
            "qc_talk_again_later"      to "Let us talk again later.",
            "qc_hope_meet_again"       to "I hope we run into each other again.",
            "qc_get_contact_info"      to "Can I get your contact information?",
            "qc_have_great_day"        to "Have a great rest of your day.",
        )

        /**
         * Computes chunks of the given list of key-label maps.
         * Each chunk accumulates entries until the combined key+value character count exceeds 250.
         * This is just a rough way of trying to optimize performance (not too long due to quadratic
         * attention, but not too short due to repeat processing of the instruction part of the prompt).
         */
        fun buildChunks(entries: Map<String, String>): List<Map<String, String>> {
            val chunks = mutableListOf<Map<String, String>>()
            var current = mutableMapOf<String, String>()
            var currentLen = 0
            for ((k, v) in entries) {
                val entryLen = k.length + v.length
                current[k] = v
                currentLen += entryLen
                if (currentLen > 250) {
                    chunks.add(current)
                    current = mutableMapOf()
                    currentLen = 0
                }
            }
            if (current.isNotEmpty()) chunks.add(current)
            return chunks
        }

        /**
         * Pre-computed chunks of [UI_STRINGS_EN].
         */
        val STRING_CHUNKS: List<Map<String, String>> by lazy {
            buildChunks(UI_STRINGS_EN)
        }

        /** Built-in UI strings keyed by language code, for languages we ship translations for. */
        val BUILTIN_UI_STRINGS: Map<String, Map<String, String>> by lazy {
            mapOf(
                "en" to UI_STRINGS_EN,
                "ko" to UI_STRINGS_KO,
                "ja" to UI_STRINGS_JA,
            )
        }

        val UI_STRINGS_KO = mapOf(
            "setup_username" to "사용자 이름 선택",
            "setup_username_hint" to "이름을 입력하세요",
            "setup_avatar" to "프로필 사진 (선택)",
            "btn_choose_photo" to "사진 선택",
            "btn_join" to "채팅 참여",
            "chat_title" to "채팅",
            "chat_placeholder" to "메시지를 입력하세요…",
            "btn_send" to "전송",
            "translating" to "번역 중…",
            "connection_lost" to "연결 끊김. 재연결 중…",
            "qc_title" to "퀵 채팅",
            "qc_cat_conversation" to "대화",
            "qc_cat_about_you" to "당신에 대해",
            "qc_cat_right_here" to "바로 여기",
            "qc_cat_common_ground" to "공통점",
            "qc_cat_spend_time" to "함께 시간 보내기",
            "qc_cat_wrapping_up" to "마무리",
            "qc_pronounce_my_name"      to "안녕하세요. 먼저 제 이름을 소리 내어 말씀드릴게요.",
            "qc_translation_unclear"    to "그게 무슨 뜻인지 설명해 주실 수 있나요? 번역이 불분명했습니다.",
            "qc_say_more_simply"        to "좀 더 간단하게 말씀해 주실 수 있나요?",
            "qc_think_app_interesting"  to "이 앱이 흥미롭다고 생각해요.",
            "qc_do_you_have_internet"   to "인터넷에 접속할 수 있으신가요?",
            "ctx_reply" to "답장",
            "ctx_copy_message" to "메시지 복사",
            "ctx_view_original" to "원문 보기",
            "ctx_retranslate" to "더 많은 맥락으로 재번역",
            "connected_users" to "접속자",
            "btn_enable_notifs" to "알림 활성화",
            "btn_disable_notifs" to "알림 비활성화",
            "btn_share_link" to "채팅 링크 공유",
            "connection_restored" to "연결됨",
            "you" to "나",
            "server_host" to "호스트",
            "btn_export" to "채팅 내보내기",
            "ctx_cancel_reply" to "답장 취소",
            "ctx_view_translation" to "번역 보기",
            "retranslating" to "재번역 중…",
            "ctx_prev_translation" to "이전 번역 보기",
            "ctx_view_latest" to "최신 번역 보기",
            "qc_where_do_you_live"      to "지금은 어디에 사세요?",
            "qc_how_long_been_here"     to "여기 오신 지 얼마나 되셨나요?",
            "qc_visiting_or_local"      to "방문 중이세요, 아니면 여기 사세요?",
            "qc_what_do_for_work"       to "어떤 일을 하세요?",
            "qc_what_are_hobbies"       to "취미가 뭔가요?",
            "qc_have_you_traveled"      to "여행을 많이 다니셨나요?",
            "qc_what_languages_speak"   to "다른 언어도 하실 수 있으신가요?",
            "qc_whats_in_profile_pic"   to "프로필 사진에 뭐가 있나요?",
            "qc_do_you_enjoy_work"      to "일이 즐거우신가요?",
            "qc_what_brings_you_here"   to "오늘 여기는 어떤 일로 오셨나요?",
            "qc_first_time_here"        to "여기 처음 오셨나요?",
            "qc_how_did_you_hear"       to "이곳을 어떻게 알게 되셨나요?",
            "qc_here_alone_friends"     to "혼자 오셨나요, 아니면 친구들과 함께 오셨나요?",
            "qc_how_long_staying"       to "얼마나 계실 예정인가요?",
            "qc_what_is_area_known_for" to "이 지역은 무엇으로 유명한가요?",
            "qc_what_think_of_place"    to "이 장소가 어떤 것 같으세요?",
            "qc_do_you_know_anyone"     to "여기서 아는 분이 있으신가요?",
            "qc_what_music_like"        to "어떤 음악을 좋아하세요?",
            "qc_what_do_weekends"       to "주말에 주로 뭐 하시나요?",
            "qc_what_passionate_about"  to "어떤 것에 열정을 느끼시나요?",
            "qc_what_are_you_reading"   to "최근에 무엇을 읽거나 보고 계신가요?",
            "qc_want_grab_drink"        to "같이 음료 한잔 하실래요?",
            "qc_want_grab_food"         to "같이 뭔가 드실래요?",
            "qc_want_keep_walking"      to "같이 계속 걸을까요?",
            "qc_want_join_group"        to "저희 그룹에 합류하실래요?",
            "qc_know_good_place_nearby" to "근처에 좋은 곳 알고 계신가요?",
            "qc_nice_meeting_you"       to "만나서 정말 반가웠어요.",
            "qc_talk_again_later"       to "나중에 다시 얘기해요.",
            "qc_hope_meet_again"        to "또 다시 만날 수 있으면 좋겠어요.",
            "qc_get_contact_info"       to "연락처를 알 수 있을까요?",
            "qc_have_great_day"         to "남은 하루도 좋은 하루 되세요.",
        )

        val UI_STRINGS_JA = mapOf(
            "setup_username" to "ユーザー名を入力",
            "setup_username_hint" to "名前を入力してください",
            "setup_avatar" to "プロフィール画像（任意）",
            "btn_choose_photo" to "写真を選ぶ",
            "btn_join" to "チャットに参加",
            "chat_title" to "チャット",
            "chat_placeholder" to "メッセージを入力…",
            "btn_send" to "送信",
            "translating" to "翻訳中…",
            "connection_lost" to "接続が切れました。再接続中…",
            "qc_title" to "クイックチャット",
            "qc_cat_conversation" to "会話",
            "qc_cat_about_you" to "あなたについて",
            "qc_cat_right_here" to "今ここで",
            "qc_cat_common_ground" to "共通の話題",
            "qc_cat_spend_time" to "一緒に過ごす",
            "qc_cat_wrapping_up" to "締めくくり",
            "qc_pronounce_my_name"      to "はじめまして。まず自分の名前を声に出して言いますね。",
            "qc_translation_unclear"    to "どういう意味ですか？翻訳がわかりにくかったです。",
            "qc_say_more_simply"        to "もっとシンプルに言っていただけますか？",
            "qc_think_app_interesting"  to "このアプリは面白いと思います。",
            "qc_do_you_have_internet"   to "インターネットに接続できますか？",
            "ctx_reply" to "返信",
            "ctx_copy_message" to "メッセージをコピー",
            "ctx_view_original" to "原文を見る",
            "ctx_retranslate" to "より多くの文脈で再翻訳",
            "connected_users" to "接続ユーザー",
            "btn_enable_notifs" to "通知を有効にする",
            "btn_disable_notifs" to "通知を無効にする",
            "btn_share_link" to "チャットリンクを共有",
            "connection_restored" to "接続済み",
            "you" to "あなた",
            "server_host" to "ホスト",
            "btn_export" to "チャットをエクスポート",
            "ctx_cancel_reply" to "返信をキャンセル",
            "ctx_view_translation" to "翻訳を見る",
            "retranslating" to "再翻訳中…",
            "ctx_prev_translation" to "前の翻訳を見る",
            "ctx_view_latest" to "最新の翻訳を表示",
            "qc_where_do_you_live"      to "今はどこに住んでいますか？",
            "qc_how_long_been_here"     to "こちらに来てどのくらいになりますか？",
            "qc_visiting_or_local"      to "旅行中ですか、それともここに住んでいますか？",
            "qc_what_do_for_work"       to "どのようなお仕事をしていますか？",
            "qc_what_are_hobbies"       to "趣味は何ですか？",
            "qc_have_you_traveled"      to "旅行はよくされますか？",
            "qc_what_languages_speak"   to "他にどんな言語が話せますか？",
            "qc_whats_in_profile_pic"   to "プロフィール写真には何が写っていますか？",
            "qc_do_you_enjoy_work"      to "お仕事は楽しいですか？",
            "qc_what_brings_you_here"   to "今日はどういったご用件でいらっしゃいましたか？",
            "qc_first_time_here"        to "ここに来るのは初めてですか？",
            "qc_how_did_you_hear"       to "こちらをどうやって知りましたか？",
            "qc_here_alone_friends"     to "おひとりですか、それともお友達と一緒ですか？",
            "qc_how_long_staying"       to "どのくらい滞在される予定ですか？",
            "qc_what_is_area_known_for" to "この地域は何で有名ですか？",
            "qc_what_think_of_place"    to "この場所はいかがですか？",
            "qc_do_you_know_anyone"     to "ここで知っている方はいますか？",
            "qc_what_music_like"        to "どんな音楽が好きですか？",
            "qc_what_do_weekends"       to "週末は何をするのが好きですか？",
            "qc_what_passionate_about"  to "何に情熱を感じますか？",
            "qc_what_are_you_reading"   to "最近、何を読んだり観たりしていますか？",
            "qc_want_grab_drink"        to "一緒に飲みに行きませんか？",
            "qc_want_grab_food"         to "一緒に何か食べませんか？",
            "qc_want_keep_walking"      to "一緒にもう少し歩きませんか？",
            "qc_want_join_group"        to "私たちのグループに加わりませんか？",
            "qc_know_good_place_nearby" to "近くに良い場所を知っていますか？",
            "qc_nice_meeting_you"       to "お会いできて本当によかったです。",
            "qc_talk_again_later"       to "また後で話しましょう。",
            "qc_hope_meet_again"        to "またどこかでお会いできると嬉しいです。",
            "qc_get_contact_info"       to "連絡先を教えていただけますか？",
            "qc_have_great_day"         to "残りの一日も良い一日をお過ごしください。",
        )
    }
}