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

    // ── Debug state ────────────────────────────────────────────────────────────
    private val _debugFlow = MutableStateFlow(InferenceDebugState())
    val debugFlow: StateFlow<InferenceDebugState> = _debugFlow.asStateFlow()

    init {
        scope.launch { processQueue() }
    }

    fun enqueue(task: TranslationTask): Boolean {
        val added = synchronized(enqueuedKeys) {
            if (enqueuedKeys.add(task.key)) {
                queue.offer(task)
                true
            } else {
                false
            }
        }
        return added
    }

    fun stop() {
        scope.cancel()
        queue.clear()
        synchronized(enqueuedKeys) { enqueuedKeys.clear() }
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
            synchronized(enqueuedKeys) { enqueuedKeys.remove(task.key) }
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
        val prompt = """Translate each UI string to $languageName for a chat application. 
Output ONLY a JSON object with the same keys and translated values. No other text.
Input:
$uiStrings"""

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

                    // ── Token repetition check ───────────────────────────────
                    val token = progress
                    recentTokens.addLast(token)
                    if (recentTokens.size > 24) recentTokens.removeFirst()
                    val tokenRepeatCount = recentTokens.count { it == token }
                    if (tokenRepeatCount >= 8) {
                        Log.w(TAG, "Stopping UI inference: token '$token' repeated $tokenRepeatCount times")
                        shouldStop = true
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

        // Extract the JSON object from the response //TODO: would possibly be more efficient to translate in smaller chunks, but certainly harder to code.
        val jsonStart = raw.indexOf('{')
        val jsonEnd = raw.lastIndexOf('}')
        if (jsonStart < 0 || jsonEnd < jsonStart) return
        
        try {
            val jsonStr = raw.substring(jsonStart, jsonEnd + 1)
            Log.d(TAG, "UI translation JSON to parse: $jsonStr")

            // Clean up any potentially malformed JSON
            val cleanedJsonStr = jsonStr
                // poorly-quoted keys:  'key': " or `key": ' or "key': ` or anything like that -> "key": "...
                .replace(Regex("""(?<=[{,\s])['"`]([a-z_]+)['"`]\s*:\s*['"`]"""), "\"$1\": \"")
                // unquoted keys:  key:  -> "key":
                .replace(Regex("""(?<=[{,\s])([a-z_]+)\s*:\s*"""), "\"$1\": ")
                // bad ending quote on value: "can't',\n or "you`\n} or "me',\n} or similar - MUST have a \n to be captured, but comma or not
                .replace(Regex("""['"`](,?\s*\n)"""), "\"$1")
                // trailing comma before closing curly bracket (any amount of whitespace in between)
                .replace(Regex(""",\s*\}"""), "}")

            Log.d(TAG, "Cleaned JSON: $cleanedJsonStr")

            val map = com.google.gson.Gson()
                .fromJson<Map<String, String>>(cleanedJsonStr, object : com.google.gson.reflect.TypeToken<Map<String, String>>() {}.type)
            Log.d(TAG, "UI chunk translation parsed successfully: ${map.size} keys for ${task.language} chunk ${task.chunkIndex}")
            onUiTranslationChunkReady(task.language, map)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse UI translations for ${task.language} chunk ${task.chunkIndex}", e)
            //TODO: Handle by using English until the app restarts, to avoid wasting power retrying repeatedly.
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

        /** English UI strings used as the source for translation. */
        val UI_STRINGS_EN = mapOf(
            "chat_title" to "Chat",
            "lang_title" to "Select your language",
            "setup_username" to "Choose a username",
            "setup_username_hint" to "Enter your name",
            "setup_avatar" to "Profile picture (optional)",
            "btn_choose_photo" to "Choose photo",
            "btn_skip" to "Skip",
            "btn_join" to "Join chat",
            "chat_placeholder" to "Type a message…",
            "btn_send" to "Send",
            "btn_export" to "Export chat",
            "translating" to "Translating…",
            "retranslating" to "Retranslating…",
            "ctx_view_original" to "View original",
            "ctx_view_translation" to "View translation",
            "ctx_retranslate" to "Retranslate with more context",
            "ctx_prev_translation" to "View previous translation",
            "ctx_reply" to "Reply",
            "ctx_cancel_reply" to "Cancel reply",
            "connected_users" to "Connected users",
            "connection_lost" to "Connection lost. Reconnecting…",
            "connection_restored" to "Connected",
            "you" to "You",
            "server_host" to "Host",
            "welcome" to "Welcome to the chat!",
            "no_messages" to "No messages yet. Say hello!",
            "ctx_view_latest" to "View latest translation",
            "ctx_copy_message" to "Copy message",
            "qc_title" to "Quick chat",
            "qc_cat_conversation" to "Conversation",
            "qc_cat_about_you" to "About You",
            "qc_cat_right_here" to "Right Here",
            "qc_cat_common_ground" to "Common Ground",
            "qc_cat_spend_time" to "Spend Time",
            "qc_cat_wrapping_up" to "Wrapping Up",
            "qc_translation_unclear"   to "Can you explain that? The translation was unclear.",
            "qc_say_more_simply"       to "Could you say that more simply?",
            "qc_slow_down"             to "Can we slow down a little?",
            "qc_still_learning"        to "I am still learning this language, please be patient.",
            "qc_what_language_prefer"  to "What language are you most comfortable in?",
            "qc_did_not_understand"    to "I did not understand that last message.",
            "qc_where_are_you_from"    to "Where are you from originally?",
            "qc_where_do_you_live"     to "Where do you live now?",
            "qc_how_long_been_here"    to "How long have you been here?",
            "qc_visiting_or_local"     to "Are you visiting or do you live here?",
            "qc_what_do_for_work"      to "What do you do for work?",
            "qc_what_are_hobbies"      to "What are your hobbies?",
            "qc_have_you_traveled"     to "Have you traveled much?",
            "qc_what_brings_you_here"  to "What brings you here today?",
            "qc_first_time_here"       to "Is this your first time here?",
            "qc_how_did_you_hear"      to "How did you hear about this place?",
            "qc_here_alone_friends"    to "Are you here alone or with friends?",
            "qc_how_long_staying"      to "How long are you staying?",
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

        val UI_STRINGS_KO = mapOf(
            "chat_title" to "채팅",
            "lang_title" to "언어를 선택하세요",
            "setup_username" to "사용자 이름 선택",
            "setup_username_hint" to "이름을 입력하세요",
            "setup_avatar" to "프로필 사진 (선택)",
            "btn_choose_photo" to "사진 선택",
            "btn_skip" to "건너뛰기",
            "btn_join" to "채팅 참여",
            "chat_placeholder" to "메시지를 입력하세요…",
            "btn_send" to "전송",
            "btn_export" to "채팅 내보내기",
            "translating" to "번역 중…",
            "retranslating" to "재번역 중…",
            "ctx_view_original" to "원문 보기",
            "ctx_view_translation" to "번역 보기",
            "ctx_retranslate" to "더 많은 맥락으로 재번역",
            "ctx_prev_translation" to "이전 번역 보기",
            "ctx_reply" to "답장",
            "ctx_cancel_reply" to "답장 취소",
            "connected_users" to "접속자",
            "connection_lost" to "연결 끊김. 재연결 중…",
            "connection_restored" to "연결됨",
            "you" to "나",
            "server_host" to "호스트",
            "welcome" to "채팅에 오신 것을 환영합니다!",
            "no_messages" to "아직 메시지가 없습니다.",
            "ctx_view_latest" to "최신 번역 보기",
            "ctx_copy_message" to "메시지 복사",
            "qc_title" to "퀵 채팅",
            "qc_cat_conversation" to "대화",
            "qc_cat_about_you" to "당신에 대해",
            "qc_cat_right_here" to "바로 여기",
            "qc_cat_common_ground" to "공통점",
            "qc_cat_spend_time" to "함께 시간 보내기",
            "qc_cat_wrapping_up" to "마무리",
            "qc_translation_unclear"    to "그게 무슨 뜻인지 설명해 주실 수 있나요? 번역이 불분명했습니다.",
            "qc_say_more_simply"        to "좀 더 간단하게 말씀해 주실 수 있나요?",
            "qc_slow_down"              to "조금 천천히 얘기할 수 있을까요?",
            "qc_still_learning"         to "저는 이 언어를 아직 배우고 있어요. 양해해 주세요.",
            "qc_what_language_prefer"   to "어떤 언어가 가장 편하신가요?",
            "qc_did_not_understand"     to "방금 메시지를 이해하지 못했습니다.",
            "qc_where_are_you_from"     to "원래 어디 출신이세요?",
            "qc_where_do_you_live"      to "지금은 어디에 사세요?",
            "qc_how_long_been_here"     to "여기 오신 지 얼마나 되셨나요?",
            "qc_visiting_or_local"      to "방문 중이세요, 아니면 여기 사세요?",
            "qc_what_do_for_work"       to "어떤 일을 하세요?",
            "qc_what_are_hobbies"       to "취미가 뭔가요?",
            "qc_have_you_traveled"      to "여행을 많이 다니셨나요?",
            "qc_what_brings_you_here"   to "오늘 여기는 어떤 일로 오셨나요?",
            "qc_first_time_here"        to "여기 처음 오셨나요?",
            "qc_how_did_you_hear"       to "이곳을 어떻게 알게 되셨나요?",
            "qc_here_alone_friends"     to "혼자 오셨나요, 아니면 친구들과 함께 오셨나요?",
            "qc_how_long_staying"       to "얼마나 계실 예정인가요?",
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
            "chat_title" to "チャット",
            "lang_title" to "言語を選択してください",
            "setup_username" to "ユーザー名を入力",
            "setup_username_hint" to "名前を入力してください",
            "setup_avatar" to "プロフィール画像（任意）",
            "btn_choose_photo" to "写真を選ぶ",
            "btn_skip" to "スキップ",
            "btn_join" to "チャットに参加",
            "chat_placeholder" to "メッセージを入力…",
            "btn_send" to "送信",
            "btn_export" to "チャットをエクスポート",
            "translating" to "翻訳中…",
            "retranslating" to "再翻訳中…",
            "ctx_view_original" to "原文を見る",
            "ctx_view_translation" to "翻訳を見る",
            "ctx_retranslate" to "より多くの文脈で再翻訳",
            "ctx_prev_translation" to "前の翻訳を見る",
            "ctx_reply" to "返信",
            "ctx_cancel_reply" to "返信をキャンセル",
            "connected_users" to "接続ユーザー",
            "connection_lost" to "接続が切れました。再接続中…",
            "connection_restored" to "接続済み",
            "you" to "あなた",
            "server_host" to "ホスト",
            "welcome" to "チャットへようこそ！",
            "no_messages" to "まだメッセージはありません。",
            "ctx_view_latest" to "最新の翻訳を表示",
            "ctx_copy_message" to "メッセージをコピー",
            "qc_title" to "クイックチャット",
            "qc_cat_conversation" to "会話",
            "qc_cat_about_you" to "あなたについて",
            "qc_cat_right_here" to "今ここで",
            "qc_cat_common_ground" to "共通の話題",
            "qc_cat_spend_time" to "一緒に過ごす",
            "qc_cat_wrapping_up" to "締めくくり",
            "qc_translation_unclear"    to "どういう意味ですか？翻訳がわかりにくかったです。",
            "qc_say_more_simply"        to "もっとシンプルに言っていただけますか？",
            "qc_slow_down"              to "少しゆっくり話せますか？",
            "qc_still_learning"         to "この言語をまだ勉強中です。ご辛抱ください。",
            "qc_what_language_prefer"   to "一番使いやすい言語はどれですか？",
            "qc_did_not_understand"     to "さきほどのメッセージが理解できませんでした。",
            "qc_where_are_you_from"     to "もともとどちらのご出身ですか？",
            "qc_where_do_you_live"      to "今はどこに住んでいますか？",
            "qc_how_long_been_here"     to "こちらに来てどのくらいになりますか？",
            "qc_visiting_or_local"      to "旅行中ですか、それともここに住んでいますか？",
            "qc_what_do_for_work"       to "どのようなお仕事をしていますか？",
            "qc_what_are_hobbies"       to "趣味は何ですか？",
            "qc_have_you_traveled"      to "旅行はよくされますか？",
            "qc_what_brings_you_here"   to "今日はどういったご用件でいらっしゃいましたか？",
            "qc_first_time_here"        to "ここに来るのは初めてですか？",
            "qc_how_did_you_hear"       to "こちらをどうやって知りましたか？",
            "qc_here_alone_friends"     to "おひとりですか、それともお友達と一緒ですか？",
            "qc_how_long_staying"       to "どのくらい滞在される予定ですか？",
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