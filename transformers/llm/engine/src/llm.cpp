//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unordered_set>

#include <MNN/AutoTime.hpp>
#include "cpp/ExprDebug.hpp"
#include "llm/llm.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "prompt.hpp"
#include "tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#include "omni.hpp"
#include "speculative_decoding/generate.hpp"
#include "core/MNNFileUtils.h"
#include "json_schema.hpp"

// 0: no debug, 1: test op time, 2: print tensor info, 3: print tensor in output
#define DEBUG_MODE 0
//#define DEBUG_IMAGE

namespace MNN {
using namespace Express;
namespace Transformer {

void KVMeta::sync() {
    int revertNumber = 0;
    for (int i=0; i<n_reserve; ++i) {
        revertNumber += reserve[2*i+1];
    }
    previous = previous - remove + add + revertNumber;
    n_reserve = 0;
    reserve = nullptr;
    remove = 0;
    add = 0;
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

Llm* Llm::createLLM(const std::string& config_path) {
    std::shared_ptr<LlmConfig> config(new LlmConfig(config_path));
    Llm* llm = nullptr;
    if (config->is_visual() || config->is_audio() || config->has_talker()) {
        llm = new Omni(config);
    } else {
        llm = new Llm(config);
    }
    return llm;
}
void Llm::destroy(Llm* llm) {
    delete llm;
}

std::string Llm::dump_config() {
    return mConfig->config_.dump();
}

bool Llm::set_config(const std::string& content) {
    auto res = mConfig->config_.merge(content.c_str());
    // update prompt
    if(mPrompt != nullptr) {
        mPrompt->setParams(mConfig);
    } else {
        mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    }
    mAsync = mConfig->config_.document.HasMember("async") ? mConfig->config_.document["async"].GetBool() : true;
    mValidBlockSize.clear();
    mBlockSize = 0;
    if (mConfig->config_.document.HasMember("chunk")) {
        mBlockSize = mConfig->config_.document["chunk"].GetInt();
    }
    if (mConfig->config_.document.HasMember("chunk_limits")) {
        auto& size_limit = mConfig->config_.document["chunk_limits"];
        do {
            if (!size_limit.IsArray()) {
                MNN_ERROR("size_limit must be array, eg: [128, 1]\n");
                break;
            }
            for (auto iter = size_limit.GetArray().begin(); iter != size_limit.GetArray().end(); iter++) {
                mValidBlockSize.emplace_back(iter->GetInt());
            }
            if (mValidBlockSize.size() < 2) {
                MNN_ERROR("size_limit must be array larger than 1, eg: [128, 1]\n");
                mValidBlockSize.clear();
                break;
            }
            std::sort(mValidBlockSize.begin(), mValidBlockSize.end());
            mBlockSize = mValidBlockSize[mValidBlockSize.size()-1];
        } while (false);
    }
    return res;
}

void Llm::setRuntimeHint(std::shared_ptr<Express::Executor::RuntimeManager> &rtg) {
    rtg->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);

    rtg->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);

    /* 'quant_qkv' is deprecated, use 'attention_mode '*/
    int legacyAttentionMode = mConfig->config_.value("quant_qkv", 8); // compatibility
    int attentionMode = mConfig->config_.value("attention_mode", legacyAttentionMode); // try to read 'attention_mode'

    // 3. 设置 Hint
    rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    if (mConfig->reuse_kv() && attentionMode == 10) {
        rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, 9);
    }
    if (mConfig->use_cached_mmap()) {
        rtg->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }
    std::string tmpPath = mConfig->tmp_path();
    if (mConfig->kvcache_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    auto cachePath = mConfig->prefix_cache_path();
    rtg->setExternalPath(cachePath, MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
    if (mConfig->use_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    // set npu model dir
    rtg->setExternalPath(mConfig->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
    rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, mConfig->config_.value("dynamic_option", 0));

    rtg->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
    if (backend_type_convert(mConfig->backend_type()) != 0) { // not cpu
        std::string cacheFilePath = tmpPath.length() != 0 ? tmpPath : ".";
        rtg->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
    rtg->setHint(MNN::Interpreter::CPU_SME2_NEON_DIVISION_RATIO, mConfig->config_.value("cpu_sme2_neon_division_ratio", 41));
    rtg->setHint(MNN::Interpreter::CPU_SME_CORES, mConfig->config_.value("cpu_sme_core_num", 2));
}

void Llm::initRuntime() {
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type      = backend_type_convert(mConfig->backend_type());
    config.numThread = mConfig->thread_num();
    if(config.type == 3){
        // opencl need set numThread = 64(buffer mode)
        config.numThread |= 64;
    }
    if (mConfig->power() == "high") {
        cpuBackendConfig.power = BackendConfig::Power_High;
    } else if (mConfig->power() == "low") {
        cpuBackendConfig.power = BackendConfig::Power_Low;
    }
    if (mConfig->memory() == "high") {
        cpuBackendConfig.memory = BackendConfig::Memory_High;
    } else if (mConfig->memory() == "low") {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }
    if (mConfig->precision() == "high") {
        cpuBackendConfig.precision = BackendConfig::Precision_High;
    } else if (mConfig->precision() == "low") {
        cpuBackendConfig.precision = BackendConfig::Precision_Low;
    }
    config.backendConfig = &cpuBackendConfig;

    mRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
    setRuntimeHint(mRuntimeManager);

#if DEBUG_MODE == 1
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTimeTrace();
#endif
#if DEBUG_MODE == 2
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initTensorStatic();
#endif
#if DEBUG_MODE == 3
    mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    _initDebug();
#endif
    // get linear input thresholds and max values
    if (mConfig->config_.value("enable_debug", false)) {
        mRuntimeManager->setMode(MNN::Interpreter::Session_Debug);
    }
}

static bool canSpecDecode(std::shared_ptr<Express::Module> module) {
    bool canSpec = false;
    auto info = module->getInfo();
    // check from mnn model
    for (int i=0; i<info->inputNames.size(); ++i) {
        auto& varInfo = info->inputs[i];
        if(info->inputNames[i] == "logits_index") {
            if (varInfo.dim.size() > 0) {
                canSpec = true;
            }
        }
    }
    return canSpec;
}
void Llm::setSpeculativeConfig() {
    auto specultive_type = mConfig->speculative_type();
    if(!specultive_type.empty()) {
        if(!canSpecDecode(mModule)) {
            mInSpec = false;
            return;
        }
        mDraftLength = mConfig->draft_predict_length();
        mInSpec = true;
    }
}

bool Llm::load() {
    MNN::Express::ExecutorScope s(mExecutor);
    Timer _t;
    initRuntime();
    // init module status
    // 1. load vocab
    mTokenizer.reset(Tokenizer::createTokenizer(mConfig->tokenizer_file()));
    // 2. load context
    {
        std::ifstream contextFile(mConfig->context_file());
        if (contextFile.is_open()) {
            std::ostringstream contextStream;
            contextStream << contextFile.rdbuf();
            auto contextStr = contextStream.str();
            // check valid json
            rapidjson::Document contextDoc;
            contextDoc.Parse(contextStr.c_str());
            if (!contextDoc.HasParseError()) {
                std::string config_json = R"({
                    "jinja": {
                        "context": )" + contextStr + R"(
                    }
                })";
                mConfig->config_.merge(config_json.c_str());
            }
        }
    }
    mDiskEmbedding.reset(new DiskEmbedding(mConfig));
    mPrompt.reset(Prompt::createPrompt(mContext, mConfig));
    mSampler.reset(Sampler::createSampler(mContext, mConfig, mTokenizer.get()));
    // 3. load model
    Module::Config module_config;
    if (mConfig->backend_type() == "opencl" || mConfig->backend_type() == "vulkan" || mConfig->backend_type() == "npu") {
        module_config.shapeMutable = false;
    } else {
        module_config.shapeMutable = true;
    }
    module_config.rearrange    = true;
    // using base module for lora module
    if (mBaseModule != nullptr) {
        module_config.base = mBaseModule;
    }
    // load single model
    std::string model_path = mConfig->llm_model();

    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits"};
    if (mConfig->has_talker()) {
        outputNames.emplace_back("talker_embeds");
    }
    bool needHiddenState = false;
    if (mConfig->config_.document.HasMember("hidden_states")) {
        needHiddenState = mConfig->config_.document["hidden_states"].GetBool();
    }
    if(mConfig->speculative_type() == "mtp") {
        needHiddenState = true;
    }
    if (needHiddenState) {
        outputNames.emplace_back("hidden_states");
    }

    mRuntimeManager->setExternalFile(mConfig->llm_weight());
    if (mConfig->has_deepstack()) {
        inputNames.emplace_back("deepstack_embeds");
    }
    mModule.reset(Module::load(inputNames, outputNames, model_path.c_str(), mRuntimeManager, &module_config));
    mRuntimeManager->setExternalFile("");
    if(nullptr == mModule) {
        MNN_ERROR("[Error]: Load module failed, please check model.\n");
        if(outputNames.size() > 1) {
            MNN_ERROR("[Warning]: Set module multi outputs, please double check.\n");
        }
        return false;
    }
    // set speculative decoding params
    setSpeculativeConfig();
    // create generation strategy
    mGenerationStrategy = GenerationStrategyFactory::create(this, mContext, mConfig, mInSpec);

    int decode_type_num = 1;
    int verify_length = 1;
    if(mInSpec) {
        // decode one token or mDraftLength token
        decode_type_num = 2;
        verify_length = mDraftLength + 1;
        // speculative decode module
        mModulePool[std::make_pair(verify_length, true)].reset(Module::clone(mModule.get()));
    }

    // autoregressive decode module
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()));
    // prefill module
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;

    // module input varp setting
    logitsLastIdx = _var<int>({-1}, {1});
    logitsAllIdx = _var<int>({0}, {1});
    // index match with seq_len
    mAttentionMaskVarVec.resize(decode_type_num);
    mPositionIdsVarVec.resize(decode_type_num);
    for(int i = 0; i < decode_type_num; i++) {
        int index = 1;
        if(i > 0) {
            index = verify_length;
        }
        // attentiion mask var
        {
            // Mask: lower triangular
           if (mConfig->backend_type() == "cpu" && mValidBlockSize.empty()) {
               mAttentionMaskVarVec[i] = _Input({}, NCHW, halide_type_of<float>());
               auto ptr = mAttentionMaskVarVec[i]->writeMap<float>();
               ptr[0] = 0;
           } else {
                mAttentionMaskVarVec[i] = _Input({1, 1, index, index}, NCHW, halide_type_of<float>());
                auto ptr = mAttentionMaskVarVec[i]->writeMap<float>();
                for (int i = 0; i < index; i++) {
                    for (int j = 0; j < index; j++) {
                        ptr[index * i + j] = (j > i) * std::numeric_limits<float>::lowest();
                    }
                }
           }
        }

        if (mConfig->is_mrope()) {
            mPositionIdsVarVec[i] = _Input({3, index}, NCHW, halide_type_of<int>());
        } else {
            mPositionIdsVarVec[i] = _Input({1, index}, NCHW, halide_type_of<int>());
        }
    }

    // MTP model load
    mGenerationStrategy->load(module_config);
    mContext->load_us += _t.durationInUs();
    return true;
}

Llm* Llm::create_lora(const std::string& lora_path) {
    auto llm = new Llm(std::make_shared<LlmConfig>(*mConfig));
    llm->set_config("{\"llm_model\": \"" + lora_path + "\", \"use_mmap\": false, \"use_cached_mmap\": false}");
    llm->mBaseModule = mModule.get();
    auto res = llm->load();
    if (!res) {
        MNN_ERROR("[MNN:LLM] Load Lora error\n");
        delete llm;
        return nullptr;
    }
    return llm;
}

void Llm::tuning(TuneType type, std::vector<int> candidates) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (type != OP_ENCODER_NUMBER) {
        MNN_ERROR("tuning type not supported\n");
        return;
    }
    // FIXME: Currently OpenCL Don't support KVMeta
    if (mConfig->backend_type() == "opencl") {
        return;
    }
    int decode_seq = 1;
    // Set to decode mode
    mContext->gen_seq_len = 1;
    if(mInSpec) {
        // start autoregressive decoding
        std::vector<int> input_ids = {0};
        auto logits = forwardVec(input_ids);
        int verify_length = mDraftLength + 1;
        decode_seq = verify_length;
    }
    int64_t min_time     = INT64_MAX;
    int prefer_candidate = 10;
    for (auto& candidate : candidates) {
        mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, candidate);
        Timer _t;
        std::vector<int> input_ids(decode_seq, 0);
        auto outputs = forwardVec(input_ids);
        if(outputs.empty()) {
            return;
        }
        auto logits = outputs[0];
        if (nullptr == logits.get()) {
            return;
        }
        if (logits->getInfo()->size == 0) {
            return;
        }
        auto token   = sample(logits);
        auto time = _t.durationInUs();
        if (time < min_time) {
            prefer_candidate = candidate;
            min_time         = time;
            // MNN_PRINT("op encode number:%d, decode time: %lld us\n", candidate, time);
        }
    }
    mRuntimeManager->setHint(MNN::Interpreter::OP_ENCODER_NUMBER_FOR_COMMIT, prefer_candidate);
    // clear dirty tuning kv history
    setKVCacheInfo(0, getCurrentHistory());
    reset();
}

void Llm::switchMode(Llm::Stage stage) {
    // do nothing, only reserve api
    return;
}

void Llm::setKVCacheInfo(size_t add, size_t remove, int* reserve, int n_reserve) {
    if (remove > mMeta->previous) {
        remove = mMeta->previous;
    }

    mMeta->remove = remove;
    mMeta->reserve = reserve;
    mMeta->n_reserve = n_reserve;
    mMeta->add = add;
}

std::vector<Express::VARP> Llm::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    MNN::Express::ExecutorScope s(mExecutor);
    Express::VARP logitsIndex;
    bool inDecode = mContext->gen_seq_len > 0;
    bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
    auto seqLen = hiddenState->getInfo()->dim[mSeqLenIndex];
    int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
    isAllLogists = seqLenKey == 1 ? false : isAllLogists;
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
    std::shared_ptr<Module> selectModule = mModule;
    if (mValidBlockSize.empty()) {
        if(mModulePool.find(moduleKey) == mModulePool.end()) {
            MNN_PRINT("Warning: module need new clone, cloning now.\n");
            mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMeta.get());
            mModulePool[moduleKey].reset(Module::clone(mModule.get()));
        }
        selectModule = mModulePool[moduleKey];
    }

    if (isAllLogists) {
        logitsIndex = logitsAllIdx;
    } else {
        logitsIndex = logitsLastIdx;
    }
    if (mMeta->add != seqLen) {
        // Has Pad, need all logits
        logitsIndex = logitsAllIdx;
    }

    mGenerateParam->input_embeds = nullptr;
    mGenerateParam->outputs.clear();
    mGenerateParam->validLogitSize = 0;
    mGenerateParam->validLogitStart = 0;
    std::vector<Express::VARP> inputs {hiddenState, mask, inputPos, logitsIndex};
    inputs.insert(inputs.end(), extraArgs.begin(), extraArgs.end());
    std::vector<Express::VARP> outputs = selectModule->onForward(inputs);

    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return outputs;
    }
    if (!mAsync) {
        ((MNN::Tensor*)(outputs[0]->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
    }
    mGenerateParam->input_embeds = hiddenState;
    mGenerateParam->outputs = outputs;

#if DEBUG_MODE == 3
    VARP logits = outputs[0];
    if(logits->getInfo()->dim[1] < 10 && logits->getInfo()->dim[1] >= 1) {
        for (int j = 0; j < logits->getInfo()->dim[1]; j++) {
            {
                int length = hiddenState->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = hiddenState->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("\nhiddenState statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }

            {
                int length = mask->getInfo()->dim[3];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = mask->readMap<float>()[index];
                    total += (temp / length);
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                MNN_PRINT("mask statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
            MNN_PRINT("position statistic value:%d\n", inputPos->readMap<int>()[j]);
            {
                int length = logits->getInfo()->dim[2];
                float total = 0.0;
                float max_ = std::numeric_limits<float>::lowest();
                float min_ = std::numeric_limits<float>::max();
                for (int i = 0; i < length; i++) {
                    int index = j * length + i;
                    float temp = logits->readMap<float>()[index];
                    total += temp;
                    max_ = fmax(max_, temp);
                    min_ = fmin(min_, temp);
                }
                auto ptr = logits->readMap<float>() + j * logits->getInfo()->dim[2];
                //            MNN_PRINT("\noutput data value:%6f %6f %6f %6f %6f\n", ptr[0], ptr[length/5], ptr[length/10], ptr[length/20], ptr[length/100]);
                MNN_PRINT("output statistic value:%6f, %6f, %6f\n", total, max_, min_);
            }
        }
    }
#endif
    mMeta->sync();
    return outputs;
}

VARP Llm::forward(const std::vector<int>& input_ids, bool is_prefill) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto hidden_states = embedding(input_ids);
    return forward(hidden_states);
}
VARP Llm::forward(MNN::Express::VARP input_embeds) {
    MNN::Express::ExecutorScope s(mExecutor);
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    auto out = forwardVec(input_embeds);
    if (out.empty()) {
        return nullptr;
    }
    auto logits = out[0];
    updateContext(seq_len, 1);
    return logits;
}

std::vector<VARP> Llm::forwardVec(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto input_embeds = embedding(input_ids);
    auto outputs = forwardVec(input_embeds);
    return outputs;
}

std::vector<VARP> Llm::forwardVec(MNN::Express::VARP input_embeds) {
    MNN::Express::ExecutorScope s(mExecutor);
    int seq_len         = input_embeds->getInfo()->dim[mSeqLenIndex];
    if (0 == mBlockSize) {
        mMeta->add = seq_len;
        auto attention_mask = gen_attention_mask(seq_len);
        auto position_ids = gen_position_ids(seq_len);
        auto res = forwardRaw(input_embeds, attention_mask, position_ids);
        return res;
    }
    // For decode can't support seq_len <= mBlockSize
    MNN_ASSERT(mContext->gen_seq_len <= 0 || seq_len <= mBlockSize);
    auto blockNumber = seq_len / mBlockSize;
    auto blockRemain = seq_len % mBlockSize;
    std::vector<VARP> logits;
    std::vector<VARP> embeddings;
    auto blockSize = mBlockSize;
    INTS sizeSplits;
    if (0 < blockNumber) {
        sizeSplits.resize(blockNumber);
        for (int i=0; i<blockNumber; ++i) {
            sizeSplits[i] = blockSize;
        }
        if (blockRemain > 0) {
            sizeSplits.emplace_back(blockRemain);
        }
    }
    if (sizeSplits.size() > 1) {
        embeddings = MNN::Express::_Split(input_embeds, sizeSplits);
    } else {
        embeddings = {input_embeds};
    }
    int addSize = blockSize;
    for (int i=0; i<blockNumber; ++i) {
        logits.clear();
        mMeta->add = blockSize;
        auto embed = embeddings[i];
        auto attention_mask = gen_attention_mask(blockSize);
        auto position_ids = gen_position_ids(blockSize);
        logits = forwardRaw(embed, attention_mask, position_ids);
        if(logits.empty()) {
            return logits;
        }
        updateContext(blockSize, 0);
    }
    bool hasPad = false;
    if (blockRemain != 0) {
        logits.clear();
        mMeta->add = blockRemain;
        addSize = blockRemain;
        int forwardSize = blockRemain;
        input_embeds = embeddings[embeddings.size()-1];
        if (!mValidBlockSize.empty()) {
            forwardSize = mValidBlockSize[mValidBlockSize.size()-1];
            for (int j=mValidBlockSize.size()-2; j>=0; --j) {
                if (mValidBlockSize[j] < blockRemain) {
                    break;
                }
                forwardSize = mValidBlockSize[j];
            }
            if (blockRemain < forwardSize) {
                // Pad
                hasPad = true;
                auto dim = input_embeds->getInfo()->dim;
                dim[mSeqLenIndex] = forwardSize;
                auto newEmbed = _Input(dim, NCHW);
                ::memcpy(newEmbed->writeMap<void>(), input_embeds->readMap<void>(), input_embeds->getInfo()->size * sizeof(float));
                ::memset(newEmbed->writeMap<float>() + input_embeds->getInfo()->size, 0, (newEmbed->getInfo()->size - input_embeds->getInfo()->size) * sizeof(float));
                input_embeds = newEmbed;
            }
        }
        auto attention_mask = gen_attention_mask(forwardSize);
        auto position_ids = gen_position_ids(forwardSize);
        logits = forwardRaw(input_embeds, attention_mask, position_ids);
        if(logits.empty()) {
            return logits;
        }
    }
    updateContext(-blockSize * blockNumber, 0);
    if (hasPad) {
        auto logitSize = logits[0]->getInfo()->dim[2];
        // encode
        mGenerateParam->validLogitStart = ((int)addSize - 1) * logitSize;
        mGenerateParam->validLogitSize = logitSize;
    }

    return logits;
}

void Llm::updateContext(int seq_len, int gen_len) {
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len += gen_len;
}

int Llm::sample(VARP logits, int offset, int size) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto logitsShape = logits->getInfo()->dim;
    if (offset && size) {
        MNN_ASSERT(logits->getInfo()->size >= offset + size);
        logits = _Const(logits->readMap<float>() + offset, {size}, NHWC, halide_type_of<float>());
    }
    auto token_id = mSampler->sample(logits);
    return token_id;
}

void Llm::reset() {
    mContext->output_tokens.clear();
    mContext->history_tokens.clear();
    mContext->all_seq_len = 0;
    mContext->gen_seq_len = 0;
    mContext->vision_us = 0;
    mContext->pixels_mp = 0.0f;
    mContext->audio_us = 0;
    mContext->audio_input_s = 0.0f;
    mMeta->remove = mMeta->previous;
}

void Llm::generate_init(std::ostream* os, const char* end_with) {
    // init status
    mContext->os = os;
    if (nullptr != end_with) {
        mContext->end_with = end_with;
    }
    if (!mContext->generate_str.empty()) {
        mContext->generate_str.clear();
    }
    mContext->gen_seq_len = 0;
    mContext->prefill_us  = 0;
    mContext->decode_us   = 0;
    mContext->current_token = -1;
    mContext->sample_us = 0;
    mContext->status = LlmStatus::RUNNING;
    if (!mConfig->reuse_kv()) {
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
    }
    mContext->output_tokens.clear();
    
    // Reset JSON state for constrained decoding
    if (mContext->json_mode) {
        reset_json_state();
    }
}

size_t Llm::getCurrentHistory() const {
    return mMeta->previous;
}
void Llm::eraseHistory(size_t begin, size_t end) {
    if (0 == end) {
        end = mMeta->previous;
    }
    if (end > mMeta->previous || begin >= end) {
        MNN_ERROR("Invalid erase range history larger than current\n");
        return;
    }
    if (mMeta->remove != 0) {
        MNN_ERROR("MNN-LLM: erase history hasn't been executed by response, override erase info\n");
    }
    mMeta->remove = mMeta->previous - begin;
    int revertNumber = 0;
    if (end != mMeta->previous) {
        mMeta->reserveHost.resize(2);
        mMeta->reserve = mMeta->reserveHost.data();
        mMeta->n_reserve = 1;
        mMeta->reserve[0] = end - begin;
        mMeta->reserve[1] = mMeta->previous - end;
        revertNumber = mMeta->reserve[1];
    }
    mContext->all_seq_len = mMeta->previous - mMeta->remove + revertNumber;
    // FIXME: support history_tokens erease the tokens with correct position
    if(revertNumber == 0 && mMeta->remove <  mContext->history_tokens.size()){
        mContext->history_tokens.resize(mContext->history_tokens.size() - mMeta->remove);
    }
}

bool Llm::stoped() {
    return is_stop(mContext->current_token);
}

void Llm::generate(int max_token) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (is_stop(mContext->current_token)) {
        return;
    }
    mGenerateParam->max_new_tokens = max_token;
    mGenerationStrategy->generate(*mGenerateParam);
}

std::vector<int> Llm::generate(const std::vector<int>& input_ids, int max_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }

    bool passExecute = false;
    if(mPrefixCacheMode) {
        mCallIndex++;

        // first time execute generate function
        if(mCallIndex == 1) {
            passExecute = mIsPrefixFileExist;

            if(!mIsPrefixFileExist) {
                // save prefix kvcache file
                mMeta->file_name = mPrefixCacheFileName;
                mMeta->file_flag = KVMeta::PendingWrite; // write
            } else {
                // first time and cachefile exist, pass this time
            }
            mPrefixLength = input_ids.size();
        }
        // second time execute generate function
        else if(mCallIndex == 2) {
            // second time and cachefile exist, load prefix file
            if(mIsPrefixFileExist) {
                mMeta->file_name = mPrefixCacheFileName;
                mMeta->file_flag = KVMeta::PendingRead; // read
                mMeta->seqlen_in_disk = mPrefixLength; // set_length
            }
        }
    }

    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end()); // push to history_ids_
    if(!passExecute) {
        if (0 == mBlockSize || input_ids.size() <= mBlockSize) {
            auto hidden_states = embedding(input_ids);
            return generate(hidden_states, max_tokens);
        }
        int total_size = (int)input_ids.size();
        int loop_size = UP_DIV(total_size, mBlockSize);
        for (int i = 0; i < loop_size; i++) {
            auto start = i * mBlockSize;
            auto end = (i+1) * mBlockSize;
            if (end >= total_size) {
                end = total_size;
            }
            std::vector<int> chunk_ids(input_ids.begin() + start, input_ids.begin() + end);
            auto input_embeds = embedding(chunk_ids);
            generate(input_embeds, 0);
        }
    } else {
        // update states
        updateContext((int)input_ids.size(), 0);
    }

    generate(max_tokens);
    mContext->prompt_len = static_cast<int>(input_ids.size());
    return mContext->output_tokens;
}

std::string Llm::apply_chat_template(const std::string& user_content) const {
    return mPrompt->applyTemplate(user_content, true);
}

std::string Llm::apply_chat_template(const ChatMessages& chat_prompts) const {
    return mPrompt->applyTemplate(chat_prompts, true);
}

std::vector<int> Llm::tokenizer_encode(const std::string& user_content) {
    return mTokenizer->encode(user_content);
}

std::vector<int> Llm::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    return mTokenizer->encode(multimodal_input.prompt_template);
}

void Llm::response(const MultimodalPrompt& multimodal_input,
                   std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto multimodal_input_copy = multimodal_input;
    if (mConfig->use_template()) {
        multimodal_input_copy.prompt_template = mPrompt->applyTemplate(multimodal_input_copy.prompt_template, true);
    }
    std::vector<int> input_ids = tokenizer_encode(multimodal_input_copy);
    response(input_ids, os, end_with, max_new_tokens);
}

std::vector<int> Llm::generate(MNN::Express::VARP input_embeds, int max_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (max_tokens < 0) {
        max_tokens = mConfig->max_new_tokens();
    }
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    mContext->prompt_len = seqLen;

    Timer _t;
    forwardVec(input_embeds);
    if(mGenerateParam->outputs.size() < 1) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    updateContext(seqLen, 0);
    mContext->prefill_us += _t.durationInUs();
    MNN::Express::ExecutorScope::Current()->gc(); // after prefill

    // prefix cache mode and response second time
    if(mPrefixCacheMode && mCallIndex == 2) {
        if(mIsPrefixFileExist) {
            // when cachefile exist, after second time prefill, updata previous length
            mMeta->previous += mMeta->seqlen_in_disk;
        }
        // recover meta status
        mMeta->seqlen_in_disk = 0;
        mMeta->file_name = "";
        mMeta->file_flag = KVMeta::NoChange;
        mMeta->layer_index = 0;
        // recover normal mode
        mPrefixCacheMode = false;
    }


#if DEBUG_MODE == 3
    {
        std::ofstream outFile("input_embeds.txt");
        auto temp = input_embeds->readMap<float>();
        for (size_t i = 0; i < input_embeds->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("logits.txt");
        auto temp = mGenerateParam->outputs[0]->readMap<float>();
        for (size_t i = 0; i < mGenerateParam->outputs[0]->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
#endif

    // call generation function
    if (0 < max_tokens) {
        mGenerateParam->max_new_tokens = max_tokens;
        mGenerationStrategy->generate(*mGenerateParam);
    }
    return mContext->output_tokens;
}

void Llm::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_ids, max_new_tokens);
}

void Llm::response(MNN::Express::VARP input_embeds, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    generate(input_embeds, max_new_tokens);
}

void Llm::response(const std::string& user_content, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto prompt = user_content;
    if (mConfig->use_template()) {
        prompt = mPrompt->applyTemplate(user_content, true);
        if (prompt.empty()) {
            prompt = user_content;
        }
    }
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}

void Llm::response(const ChatMessages& chat_prompts, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (chat_prompts.empty()) {
        return;
    }
    auto prompt = mPrompt->applyTemplate(chat_prompts);
    std::vector<int> input_ids = tokenizer_encode(prompt);
    response(input_ids, os, end_with, max_new_tokens);
}

Llm::Llm(std::shared_ptr<LlmConfig> config) : mConfig(config) {
    MNN::BackendConfig backendConfig;
    mExecutor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    mContext.reset(new LlmContext);
    mMeta.reset(new KVMeta);
    mMeta->layer_nums = mConfig->layer_nums();
    mGenerateParam.reset(new GenerationParams);
}

Llm::~Llm() {
#if DEBUG_MODE == 1
    if (nullptr != gTimeTraceInfo) {
        gTimeTraceInfo->dump();
    }
#endif
    mGenerateParam.reset();
    mModule.reset();
    mRuntimeManager.reset();
    mProcessorRuntimeManager.reset();
    mExecutor.reset();
}
int Llm::getOutputIndex(const std::string& name) const {
    if (mModulePool.empty()) {
        return -1;
    }
    auto info = mModulePool.begin()->second->getInfo();
    for (int i=0; i<info->outputNames.size(); ++i) {
        if (info->outputNames[i] == name) {
            return i;
        }
    }
    return -1;
}
std::vector<Express::VARP> Llm::getOutputs() const {
    return mGenerateParam->outputs;
}

bool Llm::setPrefixCacheFile(const std::string& filename, int flag) {
    mPrefixCacheFileName = filename;
    mCallIndex = 0;
    mPrefixCacheMode = true;


    mIsPrefixFileExist = true;
    // check kvcache, validate file existence
    for(int i = 0; i < mConfig->layer_nums(); i++) {
        auto k_file = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i) + "_sync.k";
        if(!MNNFileExist(k_file.c_str())) {
            mIsPrefixFileExist = false;
            break;
        }
        auto v_file = MNNFilePathConcat(mConfig->prefix_cache_path(), mPrefixCacheFileName) + "_" + std::to_string(i) + "_sync.v";
        if(!MNNFileExist(v_file.c_str())) {
            mIsPrefixFileExist = false;
            break;
        }
    }
    return mIsPrefixFileExist;
}

bool Llm::reuse_kv() { return mConfig->reuse_kv(); }

static inline bool needNewVar(VARP var, int axis, int seq_len, int kv_seq_len = 0) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    if (kv_seq_len != 0 && var->getInfo()->dim[axis + 1] != kv_seq_len) {
        return true;
    }
    return false;
}

VARP Llm::embedding(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    AUTOTIME;
    int hidden_size = mConfig->hidden_size();
    int seq_len = static_cast<int>(input_ids.size());

    VARP res = _Input({seq_len, 1, hidden_size}, NCHW);
    // disk embedding to save memory
    mDiskEmbedding->embedding(input_ids, res->writeMap<float>());
    return res;
}

std::string Llm::tokenizer_decode(int id) {
    std::string word = mTokenizer->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length() - 1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word    = static_cast<char>(num);
    }
    return word;
}

VARP Llm::gen_attention_mask(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    int kv_seq_len = mContext->all_seq_len + seq_len;
    if (mConfig->attention_mask() == "float") {
        // full and sliding mix, using normal mask
        if (mConfig->attention_type() == "mix") {
            const int sliding_window = mConfig->sliding_window();
            // mix attention mask
            attentionMask = _Input({2, 1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
            auto full_attn_ptr = attentionMask->writeMap<float>();
            // full attn mask
            for (int i = 0; i < seq_len; i++) {
                const int query_pos = i + (kv_seq_len - seq_len);
                for (int j = 0; j < kv_seq_len; j++) {
                    if (j > query_pos) {
                        full_attn_ptr[kv_seq_len * i + j] = std::numeric_limits<float>::lowest();
                    } else {
                        full_attn_ptr[kv_seq_len * i + j] = 0.0f;
                    }
                }
            }
            // sliding attn mask
            auto sliding_attn_ptr = full_attn_ptr + seq_len * kv_seq_len;
            const int query_pos_offset = kv_seq_len - seq_len;
            for (int i = 0; i < seq_len; i++) {
                const int query_pos = i + query_pos_offset;
                for (int j = 0; j < kv_seq_len; j++) {
                    const int key_pos = j;
                    bool is_allowed = (key_pos <= query_pos) && (key_pos > query_pos - sliding_window);
                    if (is_allowed) {
                        sliding_attn_ptr[kv_seq_len * i + j] = 0.0f;
                    } else {
                        sliding_attn_ptr[kv_seq_len * i + j] = std::numeric_limits<float>::lowest();
                    }
                }
            }
            return attentionMask;
        }
        // Use square mask just for new generation token, save memory of attention mask
        kv_seq_len = seq_len;
        if (mAttentionMaskVarVec.size() > 0) {
            if(seq_len == 1) {
                return mAttentionMaskVarVec[0];
            }
            if (mAttentionMaskVarVec.size() > 1 && seq_len == mDraftLength) {
                return mAttentionMaskVarVec[1];
            }
        }

        // Mask: lower triangular
       if (mConfig->backend_type() == "cpu" && mValidBlockSize.empty()) { // Now only cpu supports using lower triangular to opt the attention performance
           attentionMask = _Input({}, NCHW, halide_type_of<float>());
           auto ptr = attentionMask->writeMap<float>();
           ptr[0] = 0;
       } else {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<float>());
            auto ptr = attentionMask->writeMap<float>();
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    ptr[kv_seq_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
                }
            }
       }
        return attentionMask;
    } else {
        if (needNewVar(attentionMask, 2, seq_len, kv_seq_len)) {
            attentionMask = _Input({1, 1, seq_len, kv_seq_len}, NCHW, halide_type_of<int>());
        } else {
            return attentionMask;
        }
        auto ptr = attentionMask->writeMap<int>();
        if (mConfig->attention_mask() == "glm") {
            // chatglm
            for (int i = 0; i < seq_len * kv_seq_len; i++) {
                ptr[i] = 0;
            }
            if (seq_len > 1) {
                for (int i = 1; i < seq_len; i++) {
                    ptr[seq_len * i - 1] = 1;
                }
            }
        } else {
            bool is_glm2 = mConfig->attention_mask() == "glm2";
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    int row              = i + mContext->all_seq_len;
                    ptr[seq_len * i + j] = is_glm2 ? j > row : j <= row;
                }
            }
        }
        return attentionMask;
    }
}

VARP Llm::gen_position_ids(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (mConfig->attention_mask() == "glm") {
        // chatglm
        if (needNewVar(positionIds, 2, seq_len)) {
            positionIds = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
        }
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = mContext->all_seq_len - mContext->gen_seq_len - 2;
            ptr[1] = mContext->gen_seq_len + 1;
        } else {
            for (int i = 0; i < seq_len - 1; i++) {
                ptr[i]           = i;
                ptr[seq_len + i] = 0;
            }
            ptr[seq_len - 1]     = seq_len - 2;
            ptr[2 * seq_len - 1] = 1;
        }
        return positionIds;
    } else {
        bool is_glm2 = mConfig->attention_mask() == "glm2";
        if (seq_len == 1) {
            auto ptr = mPositionIdsVarVec[0]->writeMap<int>();
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
            if (mConfig->is_mrope()) {
                ptr[1] = ptr[0];
                ptr[2] = ptr[0];
            }
            return mPositionIdsVarVec[0];
        }
        if(mPositionIdsVarVec.size() > 1 && seq_len == mDraftLength) {
            auto ptr = mPositionIdsVarVec[1]->writeMap<int>();
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
            return mPositionIdsVarVec[1];
        }

        if (mConfig->is_mrope()) {
            positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
            auto ptr = positionIds->writeMap<int>();
            for (int i = 0; i < seq_len; i++) {
                ptr[0 * seq_len + i] = i + mContext->all_seq_len;
                ptr[1 * seq_len + i] = i + mContext->all_seq_len;
                ptr[2 * seq_len + i] = i + mContext->all_seq_len;
            }
            return positionIds;
        }

        positionIds = _Input({1, seq_len}, NCHW, halide_type_of<int>());
        auto ptr = positionIds->writeMap<int>();
        if (seq_len == 1) {
            ptr[0] = is_glm2 ? mContext->gen_seq_len : mContext->all_seq_len;
        } else {
            for (int i = 0; i < seq_len; i++) {
                ptr[i] = i + mContext->all_seq_len;
            }
        }
        return positionIds;
    }
}

bool Llm::is_stop(int token_id) {
    if (mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
        return true;
    }
    bool stop = mTokenizer->is_stop(token_id);
    if (stop) {
        mContext->status = LlmStatus::NORMAL_FINISHED;
    }
    
    // Check for JSON completion (sampler sets json_complete flag)
    if (mContext->json_mode && mContext->json_complete) {
        MNN_PRINT("JSON complete - stopping generation\n");
        mContext->status = LlmStatus::NORMAL_FINISHED;
        return true;
    }
    
    return stop;
}

void Llm::set_json_mode(bool enabled) {
    mContext->json_mode = enabled;
    if (enabled) {
        reset_json_state();
    }
}

bool Llm::is_json_mode() const {
    return mContext->json_mode;
}

void Llm::reset_json_state() {
    mContext->json_state = 0;
    mContext->json_expect_value = false;
    mContext->json_in_string = false;
    mContext->json_bracket_depth = 0;
    mContext->json_complete = false;
    mContext->ff_value_start_pos = 0;

    // Reset loop detection
    for (int i = 0; i < 15; i++) {
        mContext->recent_tokens[i] = 0;
    }
    mContext->recent_token_count = 0;
    mContext->loop_detected = false;
    
    // Clear schema tracking
    mContext->expected_keys.clear();
    mContext->generated_keys.clear();
    
    // Clear fast-forward cache
    mContext->fast_forward_tokens.clear();
    mContext->fast_forward_index = 0;
    
    // Clear pre-computed structure
    mContext->json_structure_computed = false;
    mContext->json_structure_tokens.clear();
}

void Llm::set_json_schema(const std::string& schema_json) {
    if (!mContext->json_schema) {
        mContext->json_schema = std::make_shared<JsonSchemaParser>();
    }
    
    if (mContext->json_schema->initialize(schema_json)) {
        mContext->json_schema_string = schema_json;
        mContext->json_mode = true;  // Enable JSON mode automatically
        
        // Extract expected keys from root schema
        if (mContext->json_schema->root()) {
            mContext->expected_keys.clear();
            for (const auto& prop : mContext->json_schema->root()->properties()) {
                mContext->expected_keys.push_back(prop.name);
            }
        }
        
        MNN_PRINT("JSON schema loaded with %zu properties\n", mContext->expected_keys.size());
    } else {
        MNN_PRINT("Failed to parse JSON schema\n");
        mContext->json_schema = nullptr;
    }
}

bool Llm::has_json_schema() const {
    return mContext->json_schema != nullptr && mContext->json_schema->is_loaded();
}

void Llm::clear_json_schema() {
    mContext->json_schema = nullptr;
    mContext->json_schema_string.clear();
    mContext->expected_keys.clear();
    mContext->generated_keys.clear();
    // Keep json_mode enabled for basic JSON constraint
}

// Predict next token in JSON mode without LLM inference
int Llm::predict_next_json_token() const {
    if (!mContext->json_mode) {
        return -1;
    }
    
    const std::string& gen_str = mContext->generate_str;
    if (gen_str.empty()) {
        return -1;  // Let LLM generate opening '{'
    }
    
    // Find last non-whitespace character
    char last_char = 0;
    for (auto it = gen_str.rbegin(); it != gen_str.rend(); ++it) {
        if (!std::isspace(*it)) {
            last_char = *it;
            break;
        }
    }
    
    // Count quotes to determine if we're in a string
    int quote_count = 0;
    for (size_t i = 0; i < gen_str.size(); i++) {
        if (gen_str[i] == '"' && (i == 0 || gen_str[i-1] != '\\')) {
            quote_count++;
        }
    }
    bool in_string = (quote_count % 2 == 1);
    
    // If we just closed a string (even quote count, last char was ")
    if (!in_string && last_char == '"') {
        // Next must be ':' - this is PREDICTABLE
        if (mTokenizer) {
            auto tokens = mTokenizer->encode(":");
            if (!tokens.empty()) {
                MNN_PRINT("Fast-forward: predicting ':' after string\n");
                return tokens[0];
            }
        }
    }
    
    // If we just had ':' 
    if (last_char == ':') {
        // Next should be '"' for string values - PREDICTABLE if schema says string
        if (mContext->json_schema && mContext->json_schema->is_loaded()) {
            // Could check schema here for type
        }
        // For now, let LLM decide value type
    }
    
    // If we just closed a value (not in string, last char not special)
    if (!in_string && last_char != ':' && last_char != '{' && last_char != ',' && 
        last_char != '[' && last_char != '}') {
        // Check if we need ',' or '}'
        // Count bracket depth
        int open_braces = 0, close_braces = 0;
        for (char c : gen_str) {
            if (c == '{') open_braces++;
            if (c == '}') close_braces++;
        }
        
        // If brackets are balanced, JSON is complete
        if (open_braces == close_braces && open_braces > 0) {
            // Don't predict - let model stop naturally
            return -1;
        }
        
        // Otherwise next is ',' or '}' - let LLM decide based on schema
    }
    
    // After ',' in object, next key starts with '"' - PREDICTABLE
    if (last_char == ',' && !in_string) {
        if (mTokenizer) {
            auto tokens = mTokenizer->encode("\"");
            if (!tokens.empty()) {
                MNN_PRINT("Fast-forward: predicting '\"' after ','\n");
                return tokens[0];
            }
        }
    }
    
    return -1;  // Let LLM sample normally
}

bool Llm::can_fast_forward() const {
    return !get_fast_forward_tokens().empty();
}

int Llm::get_fast_forward_count() const {
    return get_fast_forward_tokens().size();
}

void Llm::compute_json_structure() const {
    if (mContext->json_structure_computed) return;
    mContext->json_structure_computed = true;
    mContext->json_structure_tokens.clear();

    if (!mContext->json_schema || !mContext->json_schema->is_loaded()) return;
    const JsonSchemaNode* root = mContext->json_schema->root();
    if (!root) return;

    const auto& props = root->properties();
    auto required = root->required_properties();

    bool all_required = (required.size() == props.size());
    if (all_required) {
        for (const auto& prop : props) {
            bool found = false;
            for (const auto& req : required)
                if (req == prop.name) { found = true; break; }
            if (!found) { all_required = false; break; }
        }
    }
    MNN_PRINT("JSON FF: %zu properties, all_required=%d\n",
        props.size(), (int)all_required);
}

std::vector<int> Llm::get_fast_forward_tokens() const {
    std::vector<int> tokens;
    if (!mContext->json_mode || !mTokenizer) return tokens;

    if (mContext->json_schema && mContext->json_schema->is_loaded() &&
        !mContext->json_structure_computed)
        compute_json_structure();

    bool has_schema = mContext->json_schema &&
        mContext->json_schema->is_loaded() &&
        mContext->json_schema->root() &&
        !mContext->json_schema->root()->properties().empty();

    // Virtual generation string containing the imminent current_token ensuring accuracy
    std::string gen = mContext->generate_str;
    if (mContext->current_token >= 0 && mTokenizer) {
        gen += mTokenizer->decode(mContext->current_token);
    }

    if (!has_schema) {
        if (mContext->generate_str.empty()) {
            std::string full = "{\"";
            std::string s = full;
            for (int strip = (int)gen.size(); strip > 0; --strip) {
                if (strip <= (int)full.size() && full.substr(0, strip) == gen.substr(gen.size() - strip)) {
                    s = full.substr(strip);
                    break;
                }
            }
            if (!s.empty()) {
                tokens = mTokenizer->encode(s);
            }
        }
        return tokens;
    }

    const auto& props = mContext->json_schema->root()->properties();
    const int   N = static_cast<int>(props.size());

    int& ff_phase = mContext->fast_forward_index;  // 0 by reset_json_state

    // ---- Odd phase: waiting for LLM to finish writing a value ------------
    if (ff_phase % 2 == 1) {
        int boundary = mContext->ff_value_start_pos;

        // No new characters since the burst completed
        if ((int)gen.size() <= boundary) return tokens;

        std::string tail;
        for (int k = (int)gen.size() - 1; k >= boundary && tail.size() < 4; --k) {
            if (!std::isspace(static_cast<unsigned char>(gen[k])))
                tail = gen[k] + tail;
        }

        bool value_closed = !tail.empty() && (
            tail.back() == '"' ||
            tail.back() == ',' ||
            tail.back() == '}'
            );

        if (!value_closed) return tokens;  // value still open

        ff_phase += 1;
        MNN_PRINT("JSON FF: value %d closed (tail=\"%s\") → phase %d\n",
            ff_phase / 2 - 1, tail.c_str(), ff_phase);
    }

    // ---- Even phase: emit structural tokens for key[i] -------------------
    int i = ff_phase / 2;
    if (i > N) return tokens;

    std::string already;
    {
        for (int k = (int)gen.size() - 1; k >= mContext->ff_value_start_pos; --k) {
            if (!std::isspace(static_cast<unsigned char>(gen[k])))
                already = gen[k] + already;
        }
    }

    if (i == N || (!already.empty() && already.back() == '}')) {
        // All pairs done (either by schema count or LLM closed early).
        // Only emit '}' if the LLM hasn't already written it.
        if (already.empty() || already.back() != '}') {
            tokens = mTokenizer->encode("}");
            MNN_PRINT("JSON FF: phase %d → closing '}' (%zu token(s))\n",
                ff_phase, tokens.size());
        }
        else {
            MNN_PRINT("JSON FF: phase %d → LLM already wrote '}', nothing to emit\n",
                ff_phase);
        }
        ff_phase = 2 * N + 2;
        mContext->fast_forward_tokens = tokens;
        return tokens;
    }

    std::string full = (i == 0)
        ? "{\"" + props[i].name + "\":\""
        : ",\"" + props[i].name + "\":\"";

    std::string s = full;
    for (int strip = (int)already.size(); strip > 0; --strip) {
        if (strip <= (int)full.size() && full.substr(0, strip) == already.substr(already.size() - strip)) {
            s = full.substr(strip);
            MNN_PRINT("JSON FF: stripping %d already-written char(s) from burst\n", strip);
            break;
        }
    }

    if (s.empty()) {
        // Everything in the structural prefix was already written by the LLM.
        ff_phase = 2 * i + 1;
        mContext->fast_forward_tokens.clear();
        MNN_PRINT("JSON FF: phase %d → full prefix already written, deferring to LLM\n",
            2 * i);
        return tokens;
    }

    tokens = mTokenizer->encode(s);
    MNN_PRINT("JSON FF: phase %d → \"%s\" (%zu token(s))\n",
        ff_phase, s.c_str(), tokens.size());

    ff_phase = 2 * i + 1;

    mContext->fast_forward_tokens = tokens;
    return tokens;
}

} // namespace Transformer
} // namespace MNN
