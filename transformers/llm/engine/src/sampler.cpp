#include <random>
#include <queue>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "llm/llm.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#include "llmconfig.hpp"

namespace MNN{
namespace Transformer{

// sampler compute struct start
// a index and its corresponding score
struct IndexScore {
    int index;
    float score;
};
struct IndexScoreCmpLess{
    bool operator()(IndexScore a, IndexScore b) {
        return a.score < b.score;
    }
};
struct IndexScoreCmpGreater{
    bool operator()(IndexScore a, IndexScore b) {
        return a.score > b.score;
    }
};
// a series of index and their corresponding logits
struct SubsetLogits{
    std::vector<int> index;
    MNN::Express::VARP logits;
    bool is_subset;
};
// sampler compute struct end

// sampler compute functions start
Express::VARP _TempratureSoftmax(Express::VARP logits, float temperature, int axis = -1) {
    return Express::_Softmax(logits * Express::_Scalar<float>(1.0f / temperature), axis);
}

SubsetLogits createSubsetLogits(Express::VARP logits) {
    struct SubsetLogits subset;
    subset.logits = logits;
    subset.is_subset = false;
    return subset;
}

SubsetLogits createSubsetLogits(Express::VARP logits, const std::vector<int>& index) {
    struct SubsetLogits subset;
    subset.logits = logits;
    subset.index = index;
    subset.is_subset = true;
    return subset;
}

SubsetLogits createSubsetLogits(int size) {
    struct SubsetLogits subset;
    subset.logits = Express::_Input({size}, Express::NHWC);
    subset.index.resize(size);
    subset.is_subset = true;
    return subset;
}

SubsetLogits createSubsetLogits(const std::vector<float>& scores, const std::vector<int>& index) {
    int size = (int)(index.size());
    struct SubsetLogits subset;
    subset.logits = Express::_Input({size}, Express::NHWC);
    auto pointer = (float*)(subset.logits->writeMap<float>());
    for (int i = 0; i < size; ++i) {
        pointer[i] = scores[i];
    }
    subset.index = index;
    subset.is_subset = true;
    return subset;
}

void transformIndex(struct SubsetLogits& superset, struct SubsetLogits& subset) {
    if (!(superset.is_subset)) return;
    for (auto& id : subset.index) {
        id = superset.index[id];
    }
}

int select(struct SubsetLogits& subset, int id) {
    if (!(subset.is_subset)) {
        return id;
    }
    return subset.index[id];
}

int argmaxSelect(struct SubsetLogits superset) {
    auto scores = (float*)(superset.logits->readMap<float>());
    // get last dimension index
    int lastIndex = superset.logits->getInfo()->dim.size() - 1;
    // argmax size is last dimension size
    auto size = superset.logits->getInfo()->dim[lastIndex];
    float max_score = scores[0];
    int token_id = 0;
    for (int i = 0; i < size; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id = i;
        }
    }
    return select(superset, token_id);
}

int randomSelect(float* probs, size_t size) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float target = distribution(generator);
    float cumulative = 0.0;
    for (int i = 0; i < size; i++) {
        cumulative += probs[i];
        if (target < cumulative) {
            return i;
        }
    }
    return size - 1;
}

int randomSelect(Express::VARP probs) {
    return randomSelect((float*)(probs->readMap<float>()), probs->getInfo()->size);
}

int reSoftmaxSelect(struct SubsetLogits subset, float temperature) {
    int token_index_id = randomSelect(_TempratureSoftmax(subset.logits, temperature));
    return ((subset.is_subset) ? subset.index[token_index_id] : token_index_id);
}

int packSoftmax(Express::VARP logits, std::vector<IndexScore>& index_scores, float temperature) {
    auto prob_varp = _TempratureSoftmax(logits, temperature);
    auto probs = (float*)(prob_varp->readMap<float>());
    auto size = prob_varp->getInfo()->size;
    index_scores.resize(size);
    for (int i = 0; i < size; i++) {
        IndexScore m;
        m.index = i;
        m.score = probs[i];
        index_scores[i] = m;
    }
    return size;
}
// sampler compute functions end

Sampler* Sampler::createSampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, Tokenizer* tokenizer) {
    return new Sampler(context, config, tokenizer);
}

Sampler::Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, Tokenizer* tokenizer) {
    mContext = context;
    mTokenizer = tokenizer;
    // mConfig = getSamplerConfig(config);
    mConfig.max_all_tokens = config->max_all_tokens();
    mConfig.max_new_tokens = config->max_new_tokens();
    mConfig.type = config->sampler_type();
    mConfig.configSampler(mConfig.type, config);
}

/* ----------Sampler's members---------- */



/* ----------SamplerConfig---------- */
void Sampler::SamplerConfig::configSampler( std::string sampler_type, std::shared_ptr<LlmConfig> llmConfig) {
    if (sampler_type == "greedy"){
        this->configGreedy(llmConfig);
    } else if (sampler_type == "temperature"){
        this->configTemperature(llmConfig);
    } else if (sampler_type == "topK"){
        this->configTopK(llmConfig);
    } else if (sampler_type == "topP"){
        this->configTopP(llmConfig);
    } else if (sampler_type == "minP"){
        this->configMinP(llmConfig);
    } else if (sampler_type == "tfs"){
        this->configTFS(llmConfig);
    } else if (sampler_type == "typical"){
        this->configTypical(llmConfig);
    } else if (sampler_type == "penalty"){
        this->configPenalty(llmConfig);
    } else if (sampler_type == "mixed"){
        this->configMixed(llmConfig);
    }
}
void Sampler::SamplerConfig::configGreedy(std::shared_ptr<LlmConfig> llmConfig) {
    select_type = "greedy";
}
void Sampler::SamplerConfig::configTemperature(std::shared_ptr<LlmConfig> llmConfig) {
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTopK(std::shared_ptr<LlmConfig> llmConfig) {
    topK = llmConfig->topK();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTopP(std::shared_ptr<LlmConfig> llmConfig) {
    topP = llmConfig->topP();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configMinP(std::shared_ptr<LlmConfig> llmConfig) {
    minP = llmConfig->minP();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTFS(std::shared_ptr<LlmConfig> llmConfig) {
    tfsZ = llmConfig->tfsZ();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configTypical(std::shared_ptr<LlmConfig> llmConfig) {
    typical = llmConfig->typical();
    temperature = llmConfig->temperature();
    select_type = "temperature";
}
void Sampler::SamplerConfig::configPenalty(std::shared_ptr<LlmConfig> llmConfig) {
    penalty = llmConfig->penalty();
    ngram = llmConfig->ngram();
    ngram_factor = llmConfig->ngram_factor();
    sampler = llmConfig->penalty_sampler();
    select_type = sampler;
}
void Sampler::SamplerConfig::configMixed(std::shared_ptr<LlmConfig> llmConfig) {
    mixedSamplers = llmConfig->mixed_samplers();
    // std::cout << "Mixed Sampler Sequence: " << std::flush;
    for (auto samplerName : mixedSamplers) {
        this->configSampler(samplerName, llmConfig);
        // std::cout << samplerName << " " << std::flush;
    }
    // remove all "penalty", and add one to begin if presence.
    std::vector<std::string> newSamplers;
    bool hasPenalty = false;
    for (auto sampler:mixedSamplers) {
        if (sampler!="penalty") {
            newSamplers.push_back(sampler);
        } else {
            hasPenalty = true;
        }
    }
    if (hasPenalty) {
        newSamplers.insert(newSamplers.begin(), "penalty");
    }
    mixedSamplers = newSamplers;
    // std::cout << std::endl;
    // set select type
    // the final sampler select the token
    if (mixedSamplers.back() == "greedy") select_type = "greedy";
    else if(mixedSamplers.back()=="temperature") select_type = "temperature";
    else select_type = "temperature"; // By default temperature is used.
}

struct SubsetLogits Sampler::topK(struct SubsetLogits superset) {
    int K = mConfig.topK;
    auto scores = (float*)(superset.logits->readMap<float>());
    auto size = superset.logits->getInfo()->size;
    // 1. time complexity: O(nlogk)
    std::priority_queue<IndexScore, std::vector<IndexScore>, IndexScoreCmpGreater> heap;
    for (int i = 0; i < size; i++) {
        IndexScore m;
        m.index = i;
        m.score = scores[i];
        if (heap.size() < K) {
            heap.push(m);
        }
        else {
            if (heap.top().score < m.score) {
                heap.pop();
                heap.push(m);
            }
        }
    }
    // 2. store top K results
    auto subset = createSubsetLogits(K);
    float* topKscores = (float*)(subset.logits->writeMap<float>());
    for (int i = 0; i < K; i++) {
        subset.index[K-i-1] = heap.top().index;
        topKscores[K-i-1]  = heap.top().score;
        heap.pop();
    }
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::topP(struct SubsetLogits superset) {
    float p = mConfig.topP, temperature = mConfig.temperature;
    std::vector<IndexScore> index_scores;
    int size = packSoftmax(superset.logits, index_scores, temperature);
    // 1. make max heap
    std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
    // 2. top p algorithm
    auto scores = (float*)(superset.logits->readMap<float>());
    std::vector<int> index;
    std::vector<float> subset_logits;
    float cumulative = 0.0f;
    while (cumulative < p && !index_scores.empty()) {
        std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
        IndexScore m = index_scores.back();
        index_scores.pop_back();
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
        cumulative += m.score;
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::minP(struct SubsetLogits superset) {
    float p = mConfig.minP, temperature = mConfig.temperature;
    std::vector<IndexScore> index_scores;
    int size = packSoftmax(superset.logits, index_scores, temperature);
    // 1. make max heap
    std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
    // 2. min p algorithm
    auto scores = (float*)(superset.logits->readMap<float>());
    std::vector<int> index;
    std::vector<float> subset_logits;
    for (int i = 0; i < size; ++i) {
        std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
        IndexScore m = index_scores.back();
        if (m.score < p && !index.empty()) break;
        index_scores.pop_back();
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::tfs(struct SubsetLogits superset) {
    float z = mConfig.tfsZ, temperature = mConfig.temperature;
    // tfs algorithm
    // 1. softmax
    std::vector<IndexScore> index_scores;
    int size = packSoftmax(superset.logits, index_scores, temperature);
    // 2. sort
    std::sort(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
    auto scores = (float*)(superset.logits->readMap<float>());
    // 3. calculate derivatives
    std::vector<float> derivatives(size - 2, 0.0f);
    float first = index_scores[0].score - index_scores[1].score;
    float second = index_scores[1].score - index_scores[2].score;
    for (int i = 0; i < size - 2; ++i) {
        second = index_scores[i+1].score - index_scores[i+2].score;
        derivatives[i] = std::fabs(first - second);
        first = second;
    }
    // 4. normalize derivatives
    float derivatives_sum = 0.0;
    for (int i = 0; i < size - 2; ++i) derivatives_sum += derivatives[i];
    float derivatives_sum_rec = 1.0f / derivatives_sum;
    for (int i = 0; i < size - 2; ++i) derivatives[i] *= derivatives_sum_rec;
    // 5. cumulate, discard last 2 for sure.
    float cumulative = 0.0;
    std::vector<int> index;
    std::vector<float> subset_logits;
    for (int i = 0; i < size - 2; ++i) {
        IndexScore m = index_scores[i];
        cumulative += derivatives[i];
        if (cumulative >= z && !index.empty()) break;
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

struct SubsetLogits Sampler::typical(struct SubsetLogits superset) {
    float p = mConfig.typical, temperature = mConfig.temperature;
    auto prob_varp = _TempratureSoftmax(superset.logits, temperature);
    auto probs = (float*)(prob_varp->readMap<float>());
    auto size = prob_varp->getInfo()->size;
    std::vector<IndexScore> index_scores;
    index_scores.resize(size);
    // 1. calcaluate dist
    float entropy = 0.0f;
    for (int i = 0; i < size; i++) entropy -= probs[i] * std::log(probs[i]);
    for (int i = 0; i < size; i++) {
        IndexScore m;
        m.index = i;
        m.score = std::fabs(entropy + std::log(probs[i]));
        index_scores[i] = m;
    }
    // 2. make min heap for dist
    std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
    // 3. typical p algorithm
    auto scores = (float*)(superset.logits->readMap<float>());
    float cumulative = 0.0f;
    std::vector<int> index;
    std::vector<float> subset_logits;
    for (int i = 0; i < size; ++i) {
        std::pop_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
        IndexScore m = index_scores.back();
        cumulative += probs[m.index];
        if (cumulative >= p && !index.empty()) break;
        index_scores.pop_back();
        index.push_back(m.index);
        subset_logits.push_back(scores[m.index]);
    }
    auto subset = createSubsetLogits(subset_logits, index);
    transformIndex(superset, subset);
    return subset;
}

// presence penalty
// no frequency penalty now!
struct SubsetLogits Sampler::penalty(struct SubsetLogits subset) {
    float penalty = mConfig.penalty;
    int ngram = mConfig.ngram;
    float ngram_factor = mConfig.ngram_factor;
    float temperature = mConfig.temperature;
    bool penalizeNgram = (ngram_factor > 1.0f);
    if (penalty <= 1.0f) return subset; // no penalty!
    penalty = std::min(penalty, mConfig.max_penalty);
    // initialization
    std::vector<int>& prev = mContext->history_tokens;
    std::unordered_map<int, float> penalty_map;
    // 1. local ngram info, reversed order
    std::vector<int> ngram_info(ngram-1);
    if (penalizeNgram) {
        for (int n = 0; n < ngram_info.size(); ++n) {
            ngram_info[n] = prev[prev.size()-1-n];
        }
    }
    // 2. generate penalty map
    for (int i = 0; i < prev.size(); ++i) {
        if (penalty_map.count(prev[i]) == 0) penalty_map[prev[i]] = penalty;
        if (penalizeNgram) {
            float ngram_penalty = penalty;
            for (int j = i-1; i-j < ngram && j>=0; --j) {
                int idx = i-j-1;
                if (prev[j] != ngram_info[idx]) break;
                ngram_penalty *= ngram_factor;
                // no repeat larger than ngram!
                if (idx == ngram_info.size()-1) ngram_penalty = mConfig.max_penalty;
            }
            if (ngram_penalty > penalty_map[prev[i]]) penalty_map[prev[i]] = ngram_penalty;
        }
    }
    // 3. penalize logits according to penalty_map
    auto scoresMap = (float*)(subset.logits->readMap<float>());
    for (auto it = penalty_map.begin(); it != penalty_map.end(); ++it) {
        scoresMap[it->first] = (scoresMap[it->first] >= 0.0f) ? (scoresMap[it->first]/it->second) : (scoresMap[it->first]*it->second);
    }
    return subset;
}

struct SubsetLogits Sampler::mixed(struct SubsetLogits subset) {
    for (auto sampler : mConfig.mixedSamplers) {
        subset = subsetSampler(sampler, subset);
    }
    return subset;
}

struct SubsetLogits Sampler::subsetSampler(std::string sampler_type, struct SubsetLogits subset) {
    if (sampler_type == "penalty") subset = penalty(subset);
    if (sampler_type == "topK") subset = topK(subset);
    if (sampler_type == "topP") subset = topP(subset);
    if (sampler_type == "minP") subset = minP(subset);
    if (sampler_type == "tfs") subset = tfs(subset);
    if (sampler_type == "typical") subset = typical(subset);
    if (sampler_type == "mixed") subset = mixed(subset);
    // if greedy and temperate, just let the Selector handle it.
    return subset;
}

int Sampler::handleSelect(struct SubsetLogits subset) {
    if (mConfig.select_type == "greedy") {
        return argmaxSelect(subset);
    } else if(mConfig.select_type =="temperature") {
        return reSoftmaxSelect(subset, mConfig.temperature);
    }
    return 0;
}

int Sampler::sample(Express::VARP logits) {
    Timer _t;
    struct SubsetLogits subset = createSubsetLogits(logits);
    
    // Apply JSON constraint BEFORE other samplers if JSON mode is enabled
    if (mContext->json_mode) {
        applyJsonConstraint(subset.logits);
    }
    
    // Check for degenerate loop and force quote if detected
    int forced_token = -1;
    if (detectDegenerateLoop()) {
        MNN_PRINT("Degenerate loop detected! Forcing quote token.\n");
        forceQuoteToken(subset.logits, forced_token);
    }
    
    // process subsetSampler
    subset = subsetSampler(mConfig.type, subset);
    // select token from the subset
    int res = handleSelect(subset);
    
    // Override with forced token if loop was detected
    if (forced_token >= 0) {
        res = forced_token;
    }
    
    // Update state after token selection
    if (mContext->json_mode) {
        // Use schema-aware state update if schema loaded
        if (mContext->json_schema && mContext->json_schema->is_loaded()) {
            updateSchemaState(res);
        } else {
            updateJsonState(res);
        }
    }
    
    // Update recent tokens for loop detection
    updateRecentTokens(res);
    
    mContext->sample_us += _t.durationInUs();
    return res;
}

/* ----------JSON Constraint Decoding---------- */

// Initialize vocabulary mapping for JSON constraint
void Sampler::initializeJsonVocab() {
    if (mJsonVocabInitialized || !mTokenizer) return;
    
    // Build character to token ID map for JSON-critical characters
    std::vector<std::string> json_chars = {"{", "}", "[", "]", ":", ",", "\"", " ", "\n", "\t"};
    for (const auto& ch : json_chars) {
        auto tokens = mTokenizer->encode(ch);
        if (!tokens.empty()) {
            mCharToTokens[ch[0]].push_back(tokens[0]);
        }
    }
    
    // Find tokens that start with digits (for numbers)
    // We need to scan through the vocabulary - this requires tokenizer access to vocab
    // For now, we'll identify them by encoding sample numbers
    for (int i = 0; i <= 9; i++) {
        auto tokens = mTokenizer->encode(std::to_string(i));
        if (!tokens.empty()) {
            mNumberStartTokens.push_back(tokens[0]);
        }
    }
    
    // Also encode common JSON literals
    auto true_tokens = mTokenizer->encode("true");
    auto false_tokens = mTokenizer->encode("false");
    auto null_tokens = mTokenizer->encode("null");
    
    if (!true_tokens.empty()) mCharToTokens['t'].push_back(true_tokens[0]);
    if (!false_tokens.empty()) mCharToTokens['f'].push_back(false_tokens[0]);
    if (!null_tokens.empty()) mCharToTokens['n'].push_back(null_tokens[0]);
    
    mJsonVocabInitialized = true;
}

// Get all valid token IDs for current JSON state
std::vector<int> Sampler::getValidJsonTokens() {
    std::vector<int> valid_tokens;
    
    if (!mJsonVocabInitialized) {
        initializeJsonVocab();
    }
    
    const std::string& gen_str = mContext->generate_str;
    
    // Track current state
    int quote_count = 0;
    for (size_t i = 0; i < gen_str.size(); i++) {
        if (gen_str[i] == '"' && (i == 0 || gen_str[i-1] != '\\')) {
            quote_count++;
        }
    }
    bool in_string = (quote_count % 2 == 1);
    
    // Track bracket depth
    int open_braces = 0, close_braces = 0;
    int open_brackets = 0, close_brackets = 0;
    for (char c : gen_str) {
        if (c == '{') open_braces++;
        if (c == '}') close_braces++;
        if (c == '[') open_brackets++;
        if (c == ']') close_brackets++;
    }
    int bracket_depth = (open_braces - close_braces) + (open_brackets - close_brackets);
    
    // If we're at the very start, only allow '{'
    if (gen_str.empty() || gen_str.find_first_not_of(" \t\n\r") == std::string::npos) {
        auto it = mCharToTokens.find('{');
        if (it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), it->second.begin(), it->second.end());
        }
        return valid_tokens;
    }
    
    // If JSON is already complete, signal completion (return empty = let model decide to stop)
    if (bracket_depth <= 0 && !gen_str.empty()) {
        // Find last non-whitespace
        char last_char = 0;
        for (auto it = gen_str.rbegin(); it != gen_str.rend(); ++it) {
            if (!std::isspace(*it)) {
                last_char = *it;
                break;
            }
        }
        if (last_char == '}' || last_char == ']') {
            // JSON complete - signal by returning empty
            // The sampler will set json_complete flag after this token
            return valid_tokens;
        }
    }
    
    if (in_string) {
        // Inside a string: allow any token (too complex to constrain)
        // But we could block closing brackets if not escaped
        return valid_tokens;
    }
    
    // Outside strings - we can constrain
    // Find what we're expecting
    char last_char = 0;
    for (auto it = gen_str.rbegin(); it != gen_str.rend(); ++it) {
        if (!std::isspace(*it)) {
            last_char = *it;
            break;
        }
    }
    
    // After { or , - expect key (string starting with ") or }
    if (last_char == '{' || last_char == ',') {
        // Allow " for string key
        auto it = mCharToTokens.find('"');
        if (it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), it->second.begin(), it->second.end());
        }
        // Allow } to close empty object - ALWAYS ALLOW THIS
        auto close_it = mCharToTokens.find('}');
        if (close_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), close_it->second.begin(), close_it->second.end());
        }
    }
    // After : - expect value
    else if (last_char == ':') {
        // Allow " for string value
        auto quote_it = mCharToTokens.find('"');
        if (quote_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), quote_it->second.begin(), quote_it->second.end());
        }
        // Allow { for object
        auto open_brace_it = mCharToTokens.find('{');
        if (open_brace_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), open_brace_it->second.begin(), open_brace_it->second.end());
        }
        // Allow [ for array
        auto open_bracket_it = mCharToTokens.find('[');
        if (open_bracket_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), open_bracket_it->second.begin(), open_bracket_it->second.end());
        }
        // Allow numbers
        valid_tokens.insert(valid_tokens.end(), mNumberStartTokens.begin(), mNumberStartTokens.end());
        // Allow true, false, null
        auto t_it = mCharToTokens.find('t');
        if (t_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), t_it->second.begin(), t_it->second.end());
        }
        auto f_it = mCharToTokens.find('f');
        if (f_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), f_it->second.begin(), f_it->second.end());
        }
        auto n_it = mCharToTokens.find('n');
        if (n_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), n_it->second.begin(), n_it->second.end());
        }
    }
    // After a value - expect , or } (or ] for arrays)
    else {
        // Allow , for more items
        auto comma_it = mCharToTokens.find(',');
        if (comma_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), comma_it->second.begin(), comma_it->second.end());
        }
        // Allow } to close object - ALWAYS ALLOW CLOSING BRACKETS
        auto close_it = mCharToTokens.find('}');
        if (close_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), close_it->second.begin(), close_it->second.end());
        }
        // Allow ] to close array
        auto close_bracket_it = mCharToTokens.find(']');
        if (close_bracket_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), close_bracket_it->second.begin(), close_bracket_it->second.end());
        }
    }
    
    return valid_tokens;
}

// Apply JSON constraints by masking invalid tokens
void Sampler::applyJsonConstraint(Express::VARP logits) {
    if (!mTokenizer) return;
    
    // If schema is loaded, use schema-based constraints
    if (mContext->json_schema && mContext->json_schema->is_loaded()) {
        applySchemaConstraint(logits);
        return;
    }
    
    // Otherwise use basic JSON constraints
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    
    // Get valid tokens for current JSON state
    auto valid_tokens = getValidJsonTokens();
    
    // If no valid tokens specified (e.g., in string), allow everything
    if (valid_tokens.empty()) {
        return;
    }
    
    // Create a mask of valid tokens
    std::vector<bool> is_valid(size, false);
    for (int token_id : valid_tokens) {
        if (token_id >= 0 && token_id < size) {
            is_valid[token_id] = true;
        }
    }
    
    // Mask invalid tokens by setting their logits to a very negative value
    const float NEG_INF = -1e9;
    for (int i = 0; i < size; i++) {
        if (!is_valid[i]) {
            scores[i] = NEG_INF;
        }
    }
}

// Apply JSON Schema constraints
void Sampler::applySchemaConstraint(Express::VARP logits) {
    if (!mTokenizer) return;
    
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    
    // Get schema-valid tokens
    auto valid_tokens = getSchemaValidTokens();
    
    if (valid_tokens.empty()) {
        return;  // Allow all if no constraints
    }
    
    // Mask invalid tokens
    const float NEG_INF = -1e9;
    for (int i = 0; i < size; i++) {
        bool is_valid = false;
        for (int valid_token : valid_tokens) {
            if (i == valid_token) {
                is_valid = true;
                break;
            }
        }
        if (!is_valid) {
            scores[i] = NEG_INF;
        }
    }
}

// Get valid tokens based on JSON schema
std::vector<int> Sampler::getSchemaValidTokens() {
    std::vector<int> valid_tokens;
    
    if (!mTokenizer || !mContext->json_schema || !mContext->json_schema->is_loaded()) {
        return valid_tokens;
    }
    
    const std::string& gen_str = mContext->generate_str;
    
    // Track if we're in a string
    int quote_count = 0;
    for (size_t i = 0; i < gen_str.size(); i++) {
        if (gen_str[i] == '"' && (i == 0 || gen_str[i-1] != '\\')) {
            quote_count++;
        }
    }
    bool in_string = (quote_count % 2 == 1);
    
    // If inside a string value, allow any token (too complex to constrain)
    if (in_string) {
        return valid_tokens;
    }
    
    // Track bracket depth
    int open_braces = 0, close_braces = 0;
    for (char c : gen_str) {
        if (c == '{') open_braces++;
        if (c == '}') close_braces++;
    }
    int bracket_depth = open_braces - close_braces;
    
    // If at start, force '{'
    if (gen_str.empty() || gen_str.find_first_not_of(" \t\n\r") == std::string::npos) {
        auto it = mCharToTokens.find('{');
        if (it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), it->second.begin(), it->second.end());
        }
        return valid_tokens;
    }
    
    // If JSON complete, allow anything (model should stop)
    if (bracket_depth <= 0 && !gen_str.empty()) {
        char last_char = 0;
        for (auto it = gen_str.rbegin(); it != gen_str.rend(); ++it) {
            if (!std::isspace(*it)) {
                last_char = *it;
                break;
            }
        }
        if (last_char == '}') {
            return valid_tokens;
        }
    }
    
    // Find last non-whitespace character
    char last_char = 0;
    for (auto it = gen_str.rbegin(); it != gen_str.rend(); ++it) {
        if (!std::isspace(*it)) {
            last_char = *it;
            break;
        }
    }
    
    const JsonSchemaNode* root = mContext->json_schema->root();
    if (!root) return valid_tokens;
    
    // After '{' - expect schema keys or '}'
    if (last_char == '{') {
        // Add all required and optional keys from schema
        for (const auto& prop : root->properties()) {
            auto quote_it = mCharToTokens.find('"');
            if (quote_it != mCharToTokens.end()) {
                valid_tokens.insert(valid_tokens.end(), quote_it->second.begin(), quote_it->second.end());
            }
        }
        // Allow '}' for empty object or if all required fields present
        auto close_it = mCharToTokens.find('}');
        if (close_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), close_it->second.begin(), close_it->second.end());
        }
    }
    // After ',' - expect next key
    else if (last_char == ',') {
        // Allow keys that haven't been generated yet
        auto quote_it = mCharToTokens.find('"');
        if (quote_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), quote_it->second.begin(), quote_it->second.end());
        }
        auto close_it = mCharToTokens.find('}');
        if (close_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), close_it->second.begin(), close_it->second.end());
        }
    }
    // After ':' - expect value based on property type
    else if (last_char == ':') {
        std::string current_key = getCurrentKey();
        const SchemaProperty* prop = root->get_property(current_key);
        
        if (prop) {
            // Constrain based on expected type
            switch (prop->type) {
                case SchemaType::STRING: {
                    auto quote_it = mCharToTokens.find('"');
                    if (quote_it != mCharToTokens.end()) {
                        valid_tokens.insert(valid_tokens.end(), quote_it->second.begin(), quote_it->second.end());
                    }
                    break;
                }
                case SchemaType::NUMBER:
                case SchemaType::INTEGER: {
                    // Add number tokens
                    valid_tokens.insert(valid_tokens.end(), mNumberStartTokens.begin(), mNumberStartTokens.end());
                    break;
                }
                case SchemaType::BOOLEAN: {
                    // Allow "true" or "false"
                    auto t_it = mCharToTokens.find('t');
                    if (t_it != mCharToTokens.end()) {
                        valid_tokens.insert(valid_tokens.end(), t_it->second.begin(), t_it->second.end());
                    }
                    auto f_it = mCharToTokens.find('f');
                    if (f_it != mCharToTokens.end()) {
                        valid_tokens.insert(valid_tokens.end(), f_it->second.begin(), f_it->second.end());
                    }
                    break;
                }
                case SchemaType::NULL_TYPE: {
                    auto n_it = mCharToTokens.find('n');
                    if (n_it != mCharToTokens.end()) {
                        valid_tokens.insert(valid_tokens.end(), n_it->second.begin(), n_it->second.end());
                    }
                    break;
                }
                default:
                    // Allow any value
                    break;
            }
        }
    }
    // After a value - expect ',' or '}'
    else {
        auto comma_it = mCharToTokens.find(',');
        if (comma_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), comma_it->second.begin(), comma_it->second.end());
        }
        auto close_it = mCharToTokens.find('}');
        if (close_it != mCharToTokens.end()) {
            valid_tokens.insert(valid_tokens.end(), close_it->second.begin(), close_it->second.end());
        }
    }
    
    return valid_tokens;
}

// Get current key being generated (for schema validation)
std::string Sampler::getCurrentKey() const {
    const std::string& gen_str = mContext->generate_str;
    
    // Find last key by looking for pattern: "key":
    size_t colon_pos = gen_str.rfind(':');
    if (colon_pos == std::string::npos) {
        return "";
    }
    
    // Find opening quote before colon
    size_t quote_end = gen_str.rfind('"', colon_pos - 1);
    if (quote_end == std::string::npos || quote_end >= colon_pos) {
        return "";
    }
    
    // Find opening quote
    size_t quote_start = gen_str.rfind('"', quote_end - 1);
    if (quote_start == std::string::npos) {
        return "";
    }
    
    return gen_str.substr(quote_start + 1, quote_end - quote_start - 1);
}

// Check if currently in a string value according to schema
bool Sampler::isInSchemaString() const {
    std::string current_key = getCurrentKey();
    if (current_key.empty()) return false;
    
    if (!mContext->json_schema || !mContext->json_schema->is_loaded()) {
        return false;
    }
    
    const SchemaProperty* prop = mContext->json_schema->root()->get_property(current_key);
    return prop && prop->type == SchemaType::STRING;
}

// Update schema state after token
void Sampler::updateSchemaState(int token_id) {
    // Update basic JSON state
    updateJsonState(token_id);
    
    // Track generated keys
    if (!mContext->json_schema || !mContext->json_schema->is_loaded()) {
        return;
    }
    
    const std::string& gen_str = mContext->generate_str;
    
    // Check if we just completed a key (pattern: "key":)
    if (gen_str.size() > 2 && gen_str[gen_str.size()-1] == ':') {
        std::string key = getCurrentKey();
        if (!key.empty()) {
            // Check if key not already tracked
            bool found = false;
            for (const auto& k : mContext->generated_keys) {
                if (k == key) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                mContext->generated_keys.push_back(key);
            }
        }
    }
}

// Update JSON state machine after a token is selected
void Sampler::updateJsonState(int token_id) {
    if (!mTokenizer) return;
    
    // Decode the token to get its string representation
    std::string token_str = mTokenizer->decode(token_id);
    
    // Update state based on the generated string
    const std::string& gen_str = mContext->generate_str;
    
    // Update string state
    int quote_count = 0;
    for (size_t i = 0; i < gen_str.size(); i++) {
        if (gen_str[i] == '"' && (i == 0 || gen_str[i-1] != '\\')) {
            quote_count++;
        }
    }
    mContext->json_in_string = (quote_count % 2 == 1);
    
    // Update bracket depth
    int open_braces = 0, close_braces = 0;
    int open_brackets = 0, close_brackets = 0;
    for (char c : gen_str) {
        if (c == '{') open_braces++;
        if (c == '}') close_braces++;
        if (c == '[') open_brackets++;
        if (c == ']') close_brackets++;
    }
    mContext->json_bracket_depth = (open_braces - close_braces) + (open_brackets - close_brackets);
    
    // Determine if we're expecting a value (after : or , or [)
    if (!gen_str.empty()) {
        char last_char = gen_str.back();
        mContext->json_expect_value = (last_char == ':' || last_char == ',' || last_char == '[');
    }
    
    // Check if JSON is complete
    // JSON is complete when:
    // 1. All brackets are closed (depth = 0)
    // 2. We have at least one character (not empty)
    // 3. Last non-whitespace character is } or ]
    if (mContext->json_bracket_depth <= 0 && !gen_str.empty()) {
        char last_non_ws = 0;
        for (auto it = gen_str.rbegin(); it != gen_str.rend(); ++it) {
            if (!std::isspace(*it)) {
                last_non_ws = *it;
                break;
            }
        }
        if (last_non_ws == '}' || last_non_ws == ']') {
            mContext->json_complete = true;
        } else {
            mContext->json_complete = false;
        }
    } else {
        mContext->json_complete = false;
    }
}

// Check if a token is valid for JSON at the current state (used by applyJsonConstraint)
bool Sampler::isValidJsonToken(int token_id, bool expect_value, bool in_string, bool is_start) {
    // This is now handled by getValidJsonTokens() and applyJsonConstraint()
    // Kept for potential future use
    return true;
}

// Get token ID for a single character (helper for JSON constraint)
int Sampler::tokenizeChar(char c) {
    if (!mTokenizer) return -1;
    std::string s(1, c);
    auto tokens = mTokenizer->encode(s);
    return tokens.empty() ? -1 : tokens[0];
}

/* ----------Degenerate Loop Detection---------- */

// Update circular buffer with new token
void Sampler::updateRecentTokens(int token_id) {
    int idx = mContext->recent_token_count % 15;
    mContext->recent_tokens[idx] = token_id;
    mContext->recent_token_count++;
    
    // Reset loop detection flag
    mContext->loop_detected = false;
}

// Detect if last 15 tokens are cycling through same 3 tokens
bool Sampler::detectDegenerateLoop() {
    // Need at least 15 tokens to check
    if (mContext->recent_token_count < 15) {
        return false;
    }

    // Get last 15 tokens and count unique
    std::unordered_map<int, int> token_counts;
    for (int i = 0; i < 15; i++) {
        int idx = (mContext->recent_token_count - 15 + i) % 15;
        int token = mContext->recent_tokens[idx];
        token_counts[token]++;
    }

    // Check if only 1-3 unique tokens in last 15 (indicates looping)
    if (token_counts.size() <= 3 && token_counts.size() >= 1) {
        MNN_PRINT("Loop detected: %zu unique tokens in last 15. Clearing buffer.\n", token_counts.size());
        mContext->loop_detected = true;

        // Clear the circular buffer to prevent immediate re-triggering
        for (int i = 0; i < 15; i++) {
            mContext->recent_tokens[i] = 0;
        }
        mContext->recent_token_count = 0;

        return true;
    }

    return false;
}

// Force selection of quotation mark token
void Sampler::forceQuoteToken(Express::VARP logits, int& selected_token) {
    if (!mTokenizer) {
        selected_token = -1;
        return;
    }
    
    // Get token ID for quotation mark
    auto quote_tokens = mTokenizer->encode("\"");
    if (quote_tokens.empty()) {
        selected_token = -1;
        return;
    }
    
    int quote_token = quote_tokens[0];
    auto scores = (float*)(logits->readMap<float>());
    auto size = logits->getInfo()->size;
    
    if (quote_token >= 0 && quote_token < size) {
        // Set quote token to highest logit
        float max_score = scores[quote_token];
        for (int i = 0; i < size; i++) {
            if (i != quote_token && scores[i] > max_score) {
                max_score = scores[i];
            }
        }
        // Make quote token the clear winner
        scores[quote_token] = max_score + 10.0f;
        selected_token = quote_token;
    } else {
        selected_token = -1;
    }
}

} // Transformer
} // MNN
