//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#include "generate.hpp"
#include <MNN/AutoTime.hpp>
#include "llm/llm.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"
#include "lookahead.hpp"

using namespace MNN::Express;

namespace MNN {
namespace Transformer {

std::shared_ptr<Generation> GenerationStrategyFactory::create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec) {
    std::shared_ptr<Generation> res;
    if(canSpec) {
        if(config->speculative_type() == "lookahead") {
            res.reset(new LookaheadGeneration(llm, context, config));
        } else if(config->speculative_type() == "mtp") {
            res.reset(new MtpGeneration(llm, context, config));
        } else if(config->speculative_type() == "eagle") {
            res.reset(new EagleGeneration(llm, context, config));
        } else {
            // autoregressive generation
            res.reset(new ArGeneration(llm, context, config));
        }
    } else {
        // autoregressive generation
        res.reset(new ArGeneration(llm, context, config));
    }
    return res;
}

ArGeneration::ArGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    // do nothing
}

// Returns true if a FF burst was consumed, false if the caller should do normal AR.
// On true, param.outputs is NOT updated — current_token is set for the next AR step.
bool ArGeneration::fastForwardStep(GenerationParams& param, int& len, int& ff_total_skipped) {
    if (!mContext->json_mode) return false;

    if (mContext->fast_forward_tokens.empty())
        mContext->fast_forward_tokens = mLlm->get_fast_forward_tokens();

    if (mContext->fast_forward_tokens.empty()) return false;

    MNN::Timer _t;

    std::vector<int> drafts;
    drafts.push_back(mContext->current_token);
    drafts.insert(drafts.end(),
        mContext->fast_forward_tokens.begin(),
        mContext->fast_forward_tokens.end());

    emitToken(mContext->current_token);

    // No live VARPs in param.outputs at this point — the caller consumed
    // and cleared them before entering the loop. Safe to forward.
    mLlm->mMeta->add = drafts.size();
    auto outputs = mLlm->forwardVec(drafts);

    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return true;
    }
    for (auto& o : outputs) {
        if (nullptr == o->readMap<float>()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return true;
        }
    }

    for (int i = 1; i < (int)drafts.size(); i++)
        emitToken(drafts[i]);

    auto logits = outputs[0];
    auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
    auto sample_offset = logits->getInfo()->size - sample_size;
    mContext->current_token = mLlm->sample(logits, sample_offset, sample_size);

    int accepted = (int)drafts.size();
    mContext->history_tokens.insert(mContext->history_tokens.end(), drafts.begin(), drafts.end());
    mContext->output_tokens.insert(mContext->output_tokens.end(), drafts.begin(), drafts.end());

    mLlm->mMeta->remove = 0;
    mLlm->updateContext(accepted, accepted - 1);

    ff_total_skipped += accepted - 1;
    len += accepted;
    mContext->decode_us += _t.durationInUs();

    mContext->ff_value_start_pos = (int)mContext->generate_str.size();
    mContext->recent_token_count = 0;
    for (int k = 0; k < 15; k++) mContext->recent_tokens[k] = 0;
    mContext->fast_forward_tokens.clear();

    // outputs goes out of scope here — destructed after forwardVec is
    // fully done and readMap has materialized all tensors. Safe.
    MNN_PRINT("FF batch: %zu tokens in 1 forward, next token %d\n",
        drafts.size(), mContext->current_token);
    return true;
}

// Decodes, appends to generate_str, and streams to os. Optionally commits to KV.
void ArGeneration::emitToken(int token) {
    auto s = mLlm->tokenizer_decode(token);
    mContext->generate_str += s;
    if (nullptr != mContext->os)
        *mContext->os << s << std::flush;
}

void ArGeneration::emitAndCommitToken(int token, bool updateKV) {
    emitToken(token);
    if (updateKV)
        mLlm->updateContext(0, 1);
}

void ArGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;
    int len = 0;
    int ff_total_skipped = 0;

    // Sample the first token from prefill outputs BEFORE the loop,
    // matching the pattern used by Lookahead, MTP, and Eagle.
    // This ensures param.outputs is consumed before any forwardVec call.
    mContext->current_token = mLlm->sample(
        param.outputs[0], param.validLogitStart, param.validLogitSize);
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    mLlm->updateContext(0, 1);

    // Explicitly null out param.outputs — the prefill VARPs must not
    // outlive this point. MNN's executor reuses internal session state
    // and having stale output VARPs alive across forwardVec corrupts it.
    param.outputs.clear();

    if (mLlm->is_stop(mContext->current_token)) {
        if (nullptr != mContext->os)
            *mContext->os << mContext->end_with << std::flush;
        return;
    }

    while (len < max_token) {
        if (mContext->status == LlmStatus::USER_CANCEL) break;
        AUTOTIME;

        if (fastForwardStep(param, len, ff_total_skipped)) {
            if (mContext->status == LlmStatus::INTERNAL_ERROR) break;
            if (mLlm->is_stop(mContext->current_token)) {
                if (nullptr != mContext->os)
                    *mContext->os << mContext->end_with << std::flush;
                break;
            }
            continue;
        }

        // Normal AR step — current_token is already set (either from
        // pre-loop sample above, or from previous iteration's forwardVec).
        emitToken(mContext->current_token);

        MNN::Timer _t;
        auto outputs = mLlm->forwardVec({ mContext->current_token });
        for (auto& o : outputs) {
            if (nullptr == o->readMap<float>()) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                break;
            }
        }
        if (outputs.empty() || mContext->status == LlmStatus::INTERNAL_ERROR) break;

        mLlm->updateContext(1, 0);
        mContext->decode_us += _t.durationInUs();

        // Sample next token immediately, then let outputs go out of scope
        // (or be overwritten next iteration) — never carry them across
        // a forwardVec boundary.
        mContext->current_token = mLlm->sample(
            outputs[0], param.validLogitStart, param.validLogitSize);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
        mLlm->updateContext(0, 1);

        if (mLlm->is_stop(mContext->current_token)) {
            if (nullptr != mContext->os)
                *mContext->os << mContext->end_with << std::flush;
            break;
        }

        param.outputs = outputs;
        len++;
    }

    if (ff_total_skipped > 0)
        MNN_PRINT("FF summary: %d/%d tokens skipped sampling this call\n",
            ff_total_skipped, len);
    if (len >= max_token)
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
}

int Generation::draftVerify(VARP logits, const std::vector<int> &drafts, bool& stop) {
    // verify draft token whether be accepted
    int i_dft = 1;
    {
        //AUTOTIME;
        for(; i_dft < drafts.size(); i_dft++) {
            auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
            auto sample_offset = logits->getInfo()->size - (drafts.size() - i_dft + 1) * sample_size;

            auto predict = mLlm->sample(logits, sample_offset, sample_size);

            // stop token just break the process
            if (mLlm->is_stop(predict)) {
                mContext->current_token = predict;
                if (nullptr != mContext->os) {
                    *mContext->os << mContext->end_with << std::flush;
                }
                stop = true;
                break;
            }
            // draft token id not match
            if(predict != drafts[i_dft]) {
                mContext->current_token = predict;
                break;
            }

            if (nullptr != mContext->os) {
                *mContext->os << mLlm->tokenizer_decode(predict);
                *mContext->os << std::flush;
            }
        }
        // all drafts are corrcet!
        if(i_dft == drafts.size()) {
            auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
            auto sample_offset = logits->getInfo()->size -  sample_size;

            auto predict = mLlm->sample(logits, sample_offset, sample_size);
            mContext->current_token = predict;
        }
    }

    return i_dft;
}



} // namespace Transformer
} // namespace MNN
