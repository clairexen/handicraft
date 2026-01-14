# Gradient-limited Recurrent Context Encoding (GRCE)

This repo extends a tiny picoGPT-style language model with a recurrent context channel that keeps a lightweight “role/focus” state alongside the usual token stream. Training and sampling logic lives in `grce.py`.

## How the context channel works
1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each block input goes through LayerNorm + a linear bottleneck `n_embd → n_grce`. The `n_layer` message vectors share weights across timesteps but are detached according to `--context-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step).
3. **Context propagation.** All per-layer messages are summed and run through a shared MLP `n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`. The LayerNormed output is the sole context vector for the next position—nothing else persists across time.
4. **Bias injection.** At the next position every block applies a single linear decoder `n_grce → n_embd`. The generated biases are added only to the newest token row of each block input, so the rest of the sequence remains untouched while the context acts as an additive steering signal.

In other words, this channel is literally the recurrent shortcut that classic RNNs tried to build, but it is implemented as a clean add-on to the Transformer stack: Each block, while computing logits for token N+1, already contains every piece of context needed to describe the prefix. The GRCE path just samples that information, compresses it into `n_grce` scalars, mixes them with a single hidden layer in the time domain, and feeds the signal into the very next step. Nothing else has to travel across time. Training stays stable because gradients do not need to propagate across multiple positions; the heavy lifting is still performed inside the per-token Transformer layers.

Plotting defaults:
- `plot.py` shows both train/test traces when no flags are provided.
- `--no-nogrce` hides the GRCE-disabled comparisons.
- `--avg-span` averages runs with the same configuration label (span stripped).
- `--store span_avg.json --store-only` writes exactly what you see, so you can feed it into analysis scripts.

This mechanism creates an explicit channel for time-domain (i.e. recurrent) signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable patterns** that act like on/off switches describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing patterns** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

Think of it this way: the original “Attention Is All You Need” insight was to rotate an interleaved recurrent stack by 90°, replace the fixed unit selector ("use the same layer from the previous token") with attention, and thereby remove the long gradient paths that made RNNs hard to train. GRCE rotates us back over depth, but instead of letting layers browse earlier positions, every layer at position N sends a small learned message to position N+1. A shared bottleneck MLP mixes those messages, and each layer in the next position adds a linear bias from the shared vector. That gives us the benefits of a depth-wise recurrent shortcut (context and role persistence) without the hard-to-train time-domain gradients—everything needed to predict the next token is already inside the previous stack, so GRCE just samples and forwards it.

This means any layer at position N can send a context-related message to any layer at position N+1, and the training signal never has to cross the position boundary: by the time we emit the message, the previous stack has already computed everything it needs to predict the next token. In practice (see the sweeps in this repo), even `--context-span 1`—which suppresses cross-position gradients entirely—matches the default span: the channel just learns how to sample the information that already exists inside the previous position’s layers.

One way to view this is: attention killed recurrence by rotating the computation over depth and letting every position look backwards. GRCE rotates part of that structure back, but with the same philosophy—keep the bottleneck tight, use learned linear emitters/decoders, and place the only nonlinear mixing in a single shared layer. The result is a recurrent path that is just as easy to train as the rest of the Transformer because gradients never have to walk through time.

## Parameter count (dominant terms)
Remember: when the model computes logits for position N+1, it already synthesized every feature it needs about the prefix—that’s what autoregressive prediction is. GRCE simply taps into that already-available context and moves it forward; it does not have to learn new facts across the boundary. That’s why gradients from position N+1 flowing back into position N via the GRCE channel are largely unnecessary: the previous stack has already computed the relevant summary while predicting the token. Training just has to learn which latent features to sample and forward through the n_grce bottleneck.

Ignoring embeddings and other lower-order pieces, two terms dominate:

- Position-domain Transformer stack: `~ 12 * n_layer * n_embd^2`
- Time-domain GRCE network (only if `n_grce > 0`): `~ 2 * n_layer * n_embd * n_grce + 4 * n_grce^2`

## Think tokens

The optional `--think N` mode lets the model reserve a special `<think>` token for internal reasoning bursts. Each batch is first processed without thinking tokens so we can score where the model most wants to emit `<think>`, then we splice the top `T≤N` positions back into the sequence. Three loss terms are combined:

1. **Plain CE (the “nothink/ce” numbers):** standard next-token loss with `<think>` and undo fillers ignored; this stays comparable with non-think runs.
2. **Think correctness penalty:** a binary loss that rewards thinking only when the following token is already predicted correctly, discouraging “think after a mistake” patterns.
3. **Plan head CE:** an auxiliary decoder (with the `<think>` column masked out) that must emit the displaced token at every think slot, so the model learns to plan the next word before it appears.

Console logs now show `train loss (learned)` and `nogrce (learned)` where the value outside parentheses is the plain CE and the parenthesised number is the aggregate objective (CE + think penalties). When a column is pure CE (for example the `nothink` column), only a single number is shown. Checkpoint histories mirror this convention by storing both `*_loss` (plain CE) and `*_loss_learned` entries for every metric, so downstream analysis scripts can pick whichever view they need. Sampling/reporting highlight `<think>` as a `❔`, and `--no-think` suppresses both prompt expansion and completion of those markers when you want clean output or to measure downstream effects.

## Undo tokens

`--undo N` injects up to `U≤N` *undo pairs* into every training block. Each pair contributes a random filler token immediately followed by a dedicated `<undo>` marker (rendered as `↩` in the logs). We shorten the base chunk to `block_size - T - 2U` tokens so the augmented sample still fits the configured block size, splice the undo pairs in sequence (allowing nesting when a later pair lands inside an earlier one), and only then insert the `T` think tokens. During loss computation the filler tokens are ignored entirely, while the `<undo>` tokens are enforced like any other label so the model learns to clean up after each random detour. Undo pairs stay in-place for all evaluations, keeping the reported losses comparable to standard runs while giving the sampler a reversible scratch pad it can lean on during training.

## Running it
Use `python grce.py --help` for CLI options. Main experiment:
- `--think N` enables the above thinking-token workflow (set `--no-think` to keep sampling clean while still training with thinking tokens).
- `--undo N` inserts up to `N` random+undo pairs per block (filler loss ignored, undo enforced).
- `--report-count N` skips training entirely, loads the latest checkpoint (if any), and prints `N` completions of the configured prompt.
- `--no-newlines` keeps the sampler from emitting newline tokens so completions stay on one line.
- `--grce-dropout M` randomly disables the context channel per position (for `M=1`, each block selects a fixed set of drop points; for `M>1`, every position drops independently with probability `1/M`).
You can reproduce the sweeps below; notice how even the `--context-span 1` run (which detaches the recurrent gradients entirely) tracks all other spans almost perfectly, confirming that the channel only needs to learn what to sample, not how to backpropagate across positions.

Every evaluation logs both GRCE-enabled and GRCE-disabled losses, and `plot.py` draws both traces for quick comparison.

For postprocessing, run for example `plot.py --avg-span --no-nogrce --store span_avg.json --store-only` and feed that JSON into your analysis scripts.

```
time bash -exc '
for cy in 2 3 5 10 10; do
	python grce.py --cycles $cy --context-span 0
	python grce.py --cycles $cy --context-span 1
	python grce.py --cycles $cy --context-span 2
	python grce.py --cycles $cy --context-span 3
	python grce.py --cycles $cy --grce-dropout 20
	python grce.py --cycles $cy --n-grce 0
done
for cy in 20 50; do
	python grce.py --cycles $cy
	python grce.py --cycles $cy --grce-dropout 20
	python grce.py --cycles $cy --n-grce 0
done
'
```

(pretty much all code in this repo is ai-generated. but of course only under my strong supervision.. ~Claire ;)

## Spare notes
- Remember: when `--think` is on, run at least one evaluation with `--no-think` to capture clean completions alongside the highlighted reasoning traces.
- Undo filler tokens are ignored for CE loss but their paired `↩` markers are enforced—watch the console to ensure the model learns to undo immediately after each random insert.

## Dev notes & agent cheat sheet
- **Project focus:** Gradient-limited Recurrent Context Encoding (GRCE) atop a picoGPT-style SimpleWiki language model.
- **Model defaults:** `n_layer=8`, `n_head=8`, `n_embd=192`, `n_grce=96`, `block_size=64`, `dropout=0.05`, `vocab_size=2000`.
- **GRCE geometry:** each layer owns a sampler `LayerNorm → n_embd → n_grce`; sampled vectors are summed, passed through a shared MLP `n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`, then per-layer decoders `n_grce → n_embd` inject the biases.
- **Parameter dominance:** ignoring embeddings, the stack costs `~12 * n_layer * n_embd^2`, and the GRCE path adds `~2 * n_layer * n_embd * n_grce + 4 * n_grce^2` when enabled.
- **Tokenizer workflow:** Byte-level BPE trained on up to `--vocab-chars` characters; saved to `model/<trainstem>_<limit>_<vocab>.json` alongside checkpoints.
- **Chunks per cycle:** Train chunk size `(block_size+1)*batch_size*steps`; test chunk size `(block_size+1)*batch_size*eval_iters*eval_calls` (where `eval_calls` covers step 1, every `eval_interval`, and the final step).
- **Persistence:** Checkpoints store weights, dataset offsets, total step counter, and the loss table; log files mirror console output and capture tokenizer timing when the tokenizer is retrained.
- **Testing:** Eval prints test/train losses plus a colorized sample; prompts are cyan/green, completions yellow/magenta, and the GRCE-disabled loss is shown for comparison.
- **Local CPU sanity checks:** run a tiny model to keep turnaround fast:
  ```bash
  .venv/bin/python grce.py --device cpu --cycles 2 --steps 20 --block-size 16 --batch-size 4 --n-layer 2 --n-head 2 --n-embd 64 --n-grce 16 --context-span 4 --grce-dropout 20 --eval-interval 10 --eval-iters 1 --generate 5
  ```
  This fits in RAM and exercises the GRCE dropout path without a GPU.
