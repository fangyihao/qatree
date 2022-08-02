'''
Created on Jul. 15, 2022

@author: yfang
'''
import torch
from util.tokenization import make_tokenizer
import mpu
from util.util import Dict2Class

def prepare_tokenizer(args):
    add_sentinel_token = 0
    if args.sentinel_token:
        add_sentinel_token = args.max_position_embeddings
    tokenizer = make_tokenizer(args.tokenizer_type, None, args.tokenizer_path, args.vocab_size,
                               args.tokenizer_model_type, add_block_symbols=args.block_lm, cache_dir=args.cache_dir,
                               add_sentinel_token=add_sentinel_token, add_task_mask=args.task_mask,
                               add_decoder_mask=args.block_mask_prob > 0.0 or args.context_mask_ratio > 0.0,
                               fix_command_token=args.fix_command_token)
    
    num_tokens = tokenizer.num_tokens
    eod_token = tokenizer.get_command('eos').Id
    assert eod_token == tokenizer.get_command('pad').Id
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by
    while (after % multiple) != 0:
        after += 1
    print('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(before, after - before, after))
    print('> found end-of-document token: {}'.format(eod_token))
    token_counts = torch.cuda.LongTensor([after, eod_token])

    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()
    args.vocab_size, args.eod_token = num_tokens, eod_token
    return tokenizer

args={}
args["tokenizer_type"] = "GPT2BPETokenizer"
args["tokenizer_path"] = "tokenizer.model"
args["vocab_size"] = 50304#30522
args["tokenizer_model_type"] = "roberta"
args["block_lm"] = True
args["cache_dir"] = None
args["task_mask"] = None
args["block_mask_prob"] = 0.0
args["context_mask_ratio"] = 0.0
args["fix_command_token"] = True
args["make_vocab_size_divisible_by"] = 128
args["sentinel_token"] = False
args = Dict2Class(args)

tokenizer = prepare_tokenizer(args)

print(tokenizer, args.vocab_size, args.eod_token)


