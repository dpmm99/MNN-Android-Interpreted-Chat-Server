package com.alibaba.mnnllm.android.hotspot

import android.content.Context
import com.alibaba.mnnllm.android.R

/**
 * Builds the list of (label, durationMs) pairs consumed by [LabelCyclerView].
 *
 * Step-number prefixes
 * --------------------
 * Each supported language has its own digit/numeral for 1 and 2.  The mapping
 * is index-parallel to the string arrays (index 0 → English, index 1 → Mandarin,
 * …).  Only 1 and 2 are supported; any other value throws [IllegalArgumentException].
 *
 * English interleaving
 * --------------------
 * The cycler **interjects** an English entry after every 5 non-English entries
 * rather than relying on pre-placed English slots in the XML.  No language is
 * skipped.  The sequence looks like:
 *
 *   [EN] lang1 lang2 lang3 lang4 lang5 [EN] lang6 lang7 … (wrapping around)
 *
 * Durations
 * ---------
 * - Interjected English entries  → ENGLISH_MS (1 400 ms)
 * - Every other entry            → NORMAL_MS  (  700 ms)
 */
object HotspotLabelCycler {

    private const val NORMAL_MS  = 700L
    private const val ENGLISH_MS = 1_400L   // 2× normal

    /**
     * Per-language step prefixes, index-parallel to the string arrays.
     *
     * Arabic and Urdu use their own Eastern Arabic-Indic digits (١ / ٢ and ۱ / ۲).
     * Tibetan uses Tibetan digits (༡ / ༢).
     * All other scripts represented here use Western Arabic numerals (1 / 2).
     */
    private val STEP_PREFIXES: Array<Pair<String, String>> = arrayOf(
        /*  0 English    */ "1." to "2.",
        /*  1 Mandarin   */ "1." to "2.",
        /*  2 Hindi      */ "1." to "2.",
        /*  3 Portuguese */ "1." to "2.",
        /*  4 Russian    */ "1." to "2.",
        /*  5 Arabic     */ "١." to "٢.",
        /*  6 French     */ "1." to "2.",
        /*  7 German     */ "1." to "2.",
        /*  8 Marathi    */ "1." to "2.",
        /*  9 Bengali    */ "1." to "2.",
        /* 10 Spanish    */ "1." to "2.",
        /* 11 Indonesian */ "1." to "2.",
        /* 12 Vietnamese */ "1." to "2.",
        /* 13 Thai       */ "1." to "2.",
        /* 14 Korean     */ "1." to "2.",
        /* 15 Japanese   */ "1." to "2.",
        /* 16 Turkish    */ "1." to "2.",
        /* 17 Punjabi    */ "1." to "2.",
        /* 18 Tamil      */ "1." to "2.",
        /* 19 Swahili    */ "1." to "2.",
        /* 20 Italian    */ "1." to "2.",
        /* 21 Dutch      */ "1." to "2.",
        /* 22 Filipino   */ "1." to "2.",
        /* 23 Ukrainian  */ "1." to "2.",
        /* 24 Persian    */ "1." to "2.",
        /* 25 Urdu       */ "۱." to "۲.",
        /* 26 Norwegian  */ "1." to "2.",
        /* 27 Danish     */ "1." to "2.",
        /* 28 Swedish    */ "1." to "2.",
        /* 29 Finnish    */ "1." to "2.",
        /* 30 Hungarian  */ "1." to "2.",
        /* 31 Polish     */ "1." to "2.",
        /* 32 Czech      */ "1." to "2.",
        /* 33 Slovak     */ "1." to "2.",
        /* 34 Serbian    */ "1." to "2.",
        /* 35 Greek      */ "1." to "2.",
        /* 36 Georgian   */ "1." to "2.",
        /* 37 Uzbek      */ "1." to "2.",
        /* 38 Kazakh     */ "1." to "2.",
        /* 39 Amharic    */ "1." to "2.",
        /* 40 Hebrew     */ "1." to "2.",
        /* 41 Lao        */ "1." to "2.",
        /* 42 Burmese    */ "1." to "2.",
        /* 43 Khmer      */ "1." to "2.",
        /* 44 Sinhala    */ "1." to "2.",
        /* 45 Tibetan    */ "༡." to "༢.",
    )

    // ── Public helpers ──────────────────────────────────────────────────────

    fun buildWifiEntries(context: Context): List<Pair<String, Long>> =
        build(context, R.array.qr_wifi_labels, stepNumber = 1)

    fun buildUrlEntries(context: Context, stepNumber: Int = 2): List<Pair<String, Long>> =
        build(context, R.array.qr_url_labels, stepNumber)

    // ── Core builder ────────────────────────────────────────────────────────

    /**
     * @param stepNumber 1 or 2 – prepended to every label in the appropriate
     *                   script for that language.
     */
    private fun build(
        context: Context,
        arrayRes: Int,
        stepNumber: Int,
    ): List<Pair<String, Long>> {
        require(stepNumber == 1 || stepNumber == 2) {
            "stepNumber must be 1 or 2 (got $stepNumber)"
        }

        val rawLabels = context.resources.getStringArray(arrayRes)

        // Index 0 is English; indices 1..last are the non-English languages.
        val englishLabel = prefixed(rawLabels[0], langIndex = 0, stepNumber)
        val nonEnglish   = rawLabels.drop(1)   // keeps original language order

        val result = mutableListOf<Pair<String, Long>>()

        // Start with English
        result += englishLabel to ENGLISH_MS

        nonEnglish.forEachIndexed { offset, rawLabel ->
            // rawLabel comes from index (offset + 1) in the original array,
            // so its STEP_PREFIXES index is also (offset + 1).
            val langIndex = offset + 1
            result += prefixed(rawLabel, langIndex, stepNumber) to NORMAL_MS

            // After every 5th non-English entry, interject English
            val nonEnglishCount = offset + 1   // 1-based count so far
            if (nonEnglishCount % 5 == 0 && langIndex < rawLabels.lastIndex) {
                result += englishLabel to ENGLISH_MS
            }
        }

        return result
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private fun prefixed(label: String, langIndex: Int, stepNumber: Int): String {
        val (prefix1, prefix2) = STEP_PREFIXES.getOrElse(langIndex) { "1." to "2." }
        val prefix = if (stepNumber == 1) prefix1 else prefix2
        return "$prefix $label"
    }
}