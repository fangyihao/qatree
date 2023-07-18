'''
Created on Jul. 11, 2023

@author: Yihao Fang
'''
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import generation_utils
'''
# Traditional Beam Search
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to German: How old are you?"

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

outputs = model.generate(
    input_ids,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# With Constrained Beam Search
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

encoder_input_str = "translate English to German: How old are you?"

force_words = ["Sie"]

input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

outputs = model.generate(
    input_ids,
    force_words_ids=force_words_ids,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)


print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


'''


from transformers import (
             LogitsProcessorList,
             MinLengthLogitsProcessor,
             BeamSearchScorer,
         )
import torch
from torch import nn
from transformers.pytorch_utils import torch_int_div
from transformers.generation_stopping_criteria import StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")


encoder_input_str = "translate English to German: How old are you?"
encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
num_beams = 3
input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
input_ids = input_ids * model.config.decoder_start_token_id


model_kwargs = {
            "encoder_outputs": model.get_encoder()(
                encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
            )
        }

beam_scorer = BeamSearchScorer(
             batch_size=1,
             num_beams=num_beams,
             device=model.device,
        )

logits_processor = LogitsProcessorList(
             [
                 MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
             ]
         )

stopping_criteria = StoppingCriteriaList()

batch_size = len(beam_scorer._beam_hyps)

beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
beam_scores[:, 1:] = -1e9
beam_scores = beam_scores.view((batch_size * num_beams,))


model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

outputs = model(
    **model_inputs,
    return_dict=True,
    output_attentions=model.config.output_attentions,
    output_hidden_states=model.config.output_hidden_states,
)

print("outputs:", outputs.logits.shape)

next_token_logits = outputs.logits[:, -1, :]

next_token_scores = nn.functional.log_softmax(
    next_token_logits, dim=-1
)  # (batch_size * num_beams, vocab_size)

print("next_token_scores:", next_token_scores.shape)

next_token_scores_processed = logits_processor(input_ids, next_token_scores)
next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

vocab_size = next_token_scores.shape[-1]
next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

next_token_scores, next_tokens = torch.topk(
    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
)

next_indices = torch_int_div(next_tokens, vocab_size)
next_tokens = next_tokens % vocab_size
beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )
beam_scores = beam_outputs["next_beam_scores"]
beam_next_tokens = beam_outputs["next_beam_tokens"]
beam_idx = beam_outputs["next_beam_indices"]

print(beam_next_tokens.shape)

print(beam_idx.shape)

print(input_ids)

input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

print(input_ids)

sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_length=stopping_criteria.max_length,
        )

print(sequence_outputs["sequences"])

print(tokenizer.batch_decode(sequence_outputs["sequences"], skip_special_tokens=True))
