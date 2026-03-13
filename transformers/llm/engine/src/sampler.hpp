#ifndef SAMPLER_hpp
#define SAMPLER_hpp

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <streambuf>
#include <functional>
#include <unordered_map>
#include <utility>

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#include "llmconfig.hpp"
#include "llm/llm.hpp"
#include "json_schema.hpp"


namespace MNN {
namespace Transformer {

class Sampler {
public:
    class SamplerConfig {
    public:
        int max_new_tokens = 512;
        int max_all_tokens = 2048;
        std::string type = "temperature";
        std::string select_type = "temperature";
        float temperature = 0.8;
        int topK = 40;
        float topP = 0.9;
        float minP = 0.05;
        float tfsZ = 1.0;
        float typical = 0.95;
        // penalty
        float penalty = 1.05;
        int ngram = 8;
        float ngram_factor = 1.02; // panalize repeated ngram with a multiplied ngram_factor.
        float max_penalty = 10.;
        std::string sampler = "temperature"; // "greedy", "temperature".
        std::vector<std::string> mixedSamplers= {"topK", "tfs", "typical", "topP", "min_p", "temperature"};
        void configSampler(std::string sampler_type, std::shared_ptr<LlmConfig> llmConfig);
        void configGreedy(std::shared_ptr<LlmConfig> llmConfig);
        void configTemperature(std::shared_ptr<LlmConfig> llmConfig);
        void configTopK(std::shared_ptr<LlmConfig> llmConfig);
        void configTopP(std::shared_ptr<LlmConfig> llmConfig);
        void configMinP(std::shared_ptr<LlmConfig> llmConfig);
        void configTFS(std::shared_ptr<LlmConfig> llmConfig);
        void configTypical(std::shared_ptr<LlmConfig> llmConfig);
        void configPenalty(std::shared_ptr<LlmConfig> llmConfig);
        void configMixed(std::shared_ptr<LlmConfig> llmConfig);
    };
public:
    static Sampler* createSampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, Tokenizer* tokenizer = nullptr);
    Sampler(std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, Tokenizer* tokenizer = nullptr);
    int sample(MNN::Express::VARP logits);
private:
    std::shared_ptr<LlmContext> mContext;
    SamplerConfig mConfig;
    Tokenizer* mTokenizer;  // Tokenizer for token-level constraints
    
    // JSON constraint state
    std::unordered_map<char, std::vector<int>> mCharToTokens;  // Map characters to token IDs
    std::vector<int> mNumberStartTokens;  // Tokens that start with digits
    bool mJsonVocabInitialized = false;
    
    // Initialize vocabulary for JSON constraint
    void initializeJsonVocab();
    std::vector<int> getValidJsonTokens();
    
    struct SubsetLogits penalty(struct SubsetLogits superset);
    struct SubsetLogits topK(struct SubsetLogits superset);
    struct SubsetLogits topP(struct SubsetLogits superset);
    struct SubsetLogits minP(struct SubsetLogits superset);
    struct SubsetLogits tfs(struct SubsetLogits superset);
    struct SubsetLogits typical(struct SubsetLogits superset);
    struct SubsetLogits mixed(struct SubsetLogits subset);
    struct SubsetLogits subsetSampler(std::string sampler_type, struct SubsetLogits subset);
    int handleSelect(struct SubsetLogits subset);
    
    // JSON-constrained decoding helpers
    void applyJsonConstraint(Express::VARP logits);
    void updateJsonState(int token_id);
    bool isValidJsonToken(int token_id, bool expect_value, bool in_string, bool is_start);
    int tokenizeChar(char c);  // Get token ID for a single character
    
    // JSON Schema-constrained decoding
    void applySchemaConstraint(Express::VARP logits);
    std::vector<int> getSchemaValidTokens();
    void updateSchemaState(int token_id);
    bool isInSchemaString() const;
    std::string getCurrentKey() const;
    SchemaType getExpectedTypeForCurrentKey() const;
    
    // Degenerate loop detection
    void updateRecentTokens(int token_id);
    bool detectDegenerateLoop();
    void forceQuoteToken(Express::VARP logits, int& selected_token);
};


} // Transformer
} // MNN


#endif // SAMPLER_hpp