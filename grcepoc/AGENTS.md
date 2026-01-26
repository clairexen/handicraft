## Dev notes & agent cheat sheet

(Pretty much all code in this repo is AI-generated—but under strong human supervision. ~Claire)

- **Project focus:** Gradient-limited Recurrent Context Encoding (GRCE) end Extended Context (XCTX) atop a picoGPT-style SimpleWiki language model.
- **Model defaults:** `n_layer=8`, `n_head=8`, `n_embd=192`, `n_grce=96`, `block_size=64`, `dropout=0.05`, `vocab_size=2000`.
- **GRCE geometry:** each layer owns a sampler `LayerNorm → n_embd → n_grce`; sampled vectors are summed, passed through a shared MLP `n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`, then per-layer decoders `n_grce → n_embd` inject the biases.
- **XCTX geometry:** identical workflow, just swapping `n_grce` for `n_xctx` so the channel has more headroom. The sampler projects down to `n_xctx // n_layer` for every layer before projecting onto `n_xctx`, the bias-injectors project onto `n_xctx // n_layer` for every layer before projecting onto `n_embd`, and the shared MLP uses a reduced hidden width `4*n_xctx//n_layer` so the wide channel keeps its parameter cost in check.
- **Parameter dominance:** ignoring embeddings, the stack costs `~12 * n_layer * n_embd^2`, and the GRCE path adds `~2 * n_layer * n_embd * n_grce + 4 * n_grce^2` when enabled.
- **Tokenizer workflow:** Byte-level BPE trained on up to `--vocab-chars` characters; saved to `model/<trainstem>_<limit>_<vocab>.json` alongside checkpoints.
- **Chunks per cycle:** Train chunk size `(block_size+1)*batch_size*steps`; test chunk size `(block_size+1)*batch_size*eval_iters*eval_calls` (where `eval_calls` covers step 1, every `eval_interval`, and the final step). Each chunk records its absolute corpus offset so the loader can recover the `block_size` tokens immediately before every sample and run the recurrent prefill.
- **Persistence:** Checkpoints store weights, dataset offsets, total step counter, cumulative wall-clock training seconds, and the loss table; log files mirror console output and capture tokenizer timing when the tokenizer is retrained.
- **Testing:** Eval prints test/train losses plus a colorized sample; prompts are cyan/green, completions yellow/magenta, and the GRCE-disabled loss is shown for comparison.
- **Local CPU sanity checks:** run a tiny model to keep turnaround fast:
  ```bash
  .venv/bin/python3 grce.py --device cpu --cycles 2 --steps 20 --block-size 16 --batch-size 4 --n-layer 2 --n-head 2 --n-embd 64 --n-grce 16 --detach-span 4 --eval-interval 10 --eval-iters 1 --generate 5
  ```
  This fits in RAM and exercises the context dropout path without a GPU.
- **Detaching parts of the stack:** `--detach-layer K` severs gradients after Transformer layer `K` (1-based), letting you freeze the lower stack while training fresh layers on top.
- **Model evaluation:** Each evaluation pass prints two loss groups: the main columns (`target`, `noprev`) described above and the six-column stress test (`plain`, `normal`, `noctx`, `noatt`, `none`, `encode`). The `plain` column runs the entire batch as a pure Transformer (both GRCE and XCTX disabled), `normal` tracks the baseline context-enabled run, and `encode` repeats that baseline with bidirectional (encoder-style) attention so you can gauge how much the causal mask costs.
- **Environment note:** always run tooling via `.venv/bin/python3` (and related entrypoints) so the local dependencies are available; the system python may lack the required packages, or there even may be no system python.
