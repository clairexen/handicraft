# Agent Notes

- **Project focus:** Gradient-less Recurrent Context Encoding (GRCE) atop a picoGPT-style SimpleWiki language model.
- **Model defaults:** `n_layer=8`, `n_head=8`, `n_embd=192`, `n_grce=96`, `block_size=64`, `dropout=0.05`, `vocab_size=2000`.
- **GRCE geometry:** `(n_layer+1)*n_embd → 4*n_embd → n_grce → 4*n_embd → n_embd` with stop-gradient at the concatenation of token input + previous FFN outputs.
- **Tokenizer workflow:** Byte-level BPE trained on up to `--vocab-chars` of the training text; saved under `model/<trainstem>_<limit>_<vocab>.json` to stay colocated with checkpoints.
- **Chunks per cycle:** Train chunk size `(block_size+1)*batch_size*steps`; test chunk size `(block_size+1)*batch_size*eval_iters*eval_calls`, where `eval_calls` counts step=1, every `eval_interval`, and the final step.
- **Persistence:** Checkpoints (`model/*.pt`) store weights, dataset offsets, total step counter, and loss table; log files mirror console output and include `[tokenizer]` timing only on fresh tokenizer training.
- **Testing:** Console prints only test loss per eval step; samples show prompt in plain text and completion in bold white.
