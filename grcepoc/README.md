# GPT with Gradient-limited Recurrent Context Encoding (GRCE)

This repo extends a tiny picoGPT-style language model with a recurrent context channel that keeps a lightweight “role/focus” state alongside the usual token stream. Training and sampling logic lives in `grce.py`.

## How the context channel works
1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each block input goes through LayerNorm + a linear bottleneck `n_embd → n_grce`. The `n_layer` message vectors share weights across timesteps but are detached according to `--context-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step).
3. **Context propagation.** All per-layer messages are summed, the (optionally detach-controlled) previous context is added into that fused vector, and the result is run through a shared MLP `n_grce → 4*n_grce → ReLU → n_grce`. The MLP output then adds the same (optionally detach-controlled) previous context again—just like a Transformer residual—before a final LayerNorm produces the next-step context. That LayerNormed output is the sole signal that crosses positions—nothing else persists across time.
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

The optional `--think N` mode lets the model reserve a special `<think>` token for internal reasoning bursts. Each batch is first processed without thinking tokens so we can score where the model most wants to emit `<think>`, then we grab the top `T+1` candidates, randomly remove *two* of them, and finally add a random position that wasn’t already in the set so we end up with `T≤N` insertions. This mix means the model still practices missing an “expected” think event, but now also experiences an “unexpected extra” think slot somewhere off the high-score path. When a `<think>` slot is inserted at position `N`, we keep the *same* token embedding and positional encoding for both the original stack and the follow-up stack at `N+1`; the second pass simply adds the `<think>` embedding on top so the block knows it is re-evaluating that position. Downstream positional encodings shift right (tokens that fall off the block window are dropped), which means the remainder of the sequence retains its original token↔position mapping. Conceptually this turns a single slot into two stacked evaluations of the same token, where the second stack gets to reuse everything the first stack already computed plus the GRCE message fired during the thinking step. The tokenizer always includes `<think>` and `<undo>` support regardless of the CLI flags, so you can freely copy checkpoints between think/non-think or undo/non-undo runs; disabling `--think` or `--undo` simply masks those tokens and enforces that they never appear in prompts/datasets. During training we intentionally reserve one random sequence per batch for “plain” mode (no `<think>` or `<undo>`), ensuring the model regularly sees non-thinking trajectories while the rest of the batch uses the full think workflow. Three loss terms are combined:

The “plain” losses reported in the training logs refer to a think/undo-disabled evaluation pass that blocks both `<think>` and `<undo>` tokens.

GRCE is what makes thinking tokens worthwhile. Without the cross-layer shortcut, inserting `<think>` at position `N` would mostly force position `N+1` to redo whatever the same layer index already computed at `N`—the stack above layer `M` would still have no way to see the work layer `M` performed one time step earlier. With GRCE, every layer at `N` can influence every layer at `N+1`, so a `<think>` token effectively chains two full layer stacks together before emitting token `N+1`. Thinking tokens therefore act as a compute-slowdown knob only in the presence of GRCE; otherwise they would add latency without increasing the network’s ability to reuse the extra work.

1. **Plain CE (the “nothink/ce” numbers):** standard next-token loss with `<think>` and undo fillers ignored; this stays comparable with non-think runs.
2. **Think correctness penalty:** a binary loss that rewards thinking only when the following token is already predicted correctly, discouraging “think after a mistake” patterns.
3. **Plan head CE:** an auxiliary decoder (with the `<think>` column masked out) that must emit the displaced token at every think slot, so the model learns to plan the next word before it appears.

Console logs now show `train loss (target)` and `nogrce (target)` where the value outside parentheses is the plain CE loss and the parenthesised number is the aggregate objective (CE + think penalties). When a column is pure CE (for example the `plain`/`nothink` column that appears when think mode is enabled), only a single number is shown. Checkpoint histories mirror this convention by storing both `*_loss` (plain CE) and `*_target` entries for every metric, plus `train_wall_seconds`, `unix_time`, and the current dataset cursors (`train_cursor` / `test_cursor`) so downstream analysis scripts can correlate metrics with real time and corpus coverage. The “plain” column is preserved as `train_loss_plain`/`test_loss_plain` alongside their GRCE-enabled and GRCE-disabled counterparts. Sampling/reporting highlight `<think>` as a `❔`; `--no-think` suppresses thinking tokens entirely, `--no-think-prompt` specifically disables thinking-token insertion inside the prompt, and `--think-hard` goes the other way, forcing a thinking token after every prompt token the model mispredicts so you can stress-test completions with dense reasoning bursts.

## Undo tokens

`--undo N` injects up to `U≤N` *undo pairs* into every training block. Each pair contributes a random filler token immediately followed by a dedicated `<undo>` marker (rendered as `↩` in the logs). We shorten the base chunk to `block_size - T - 2U` tokens so the augmented sample still fits the configured block size, splice the undo pairs in sequence (allowing nesting when a later pair lands inside an earlier one), and only then insert the `T` think tokens. During loss computation the filler tokens are ignored entirely, while the `<undo>` tokens are enforced like any other label so the model learns to clean up after each random detour. Undo pairs stay in-place for all evaluations, keeping the reported losses comparable to standard runs while giving the sampler a reversible scratch pad it can lean on during training.

## A mental model for large ReLU networks and transformer-style networks

Think of a Transformer stack as a long sequence of “mix → detect → remix” stages operating on a shared channel. The **mix** stage is completely linear: attention builds queries/keys, computes the weighted sum of value vectors, and hands that mixture to the downstream detector (GRCE’s mix stage is just the per-layer sampler sums). The **detect** stage pushes that linear combination through non-linearities (GELU/ReLU) to decide which motifs are active, and the **remix** stage projects those non-negative activations back onto the channel. We normalize twice: once between mix and detect (LayerNorm before attention/MLP so the detector sees a stable magnitude), and once between remix and the next block (either by placing LayerNorm at the start of the next block—pre-norm, as in this repo—or at the end of the current block—post-norm). Both normalizations exist to keep the channel’s signal-to-noise ratio consistent.

### Messages riding on an orthogonal basis
An N-dimensional vector can simultaneously carry N independent “messages” if each one aligns with an orthogonal basis vector. In practice the basis is not literally orthogonal—the model learns whatever mixture best represents the training distribution—but it helps to imagine that each neuron owns a direction in embedding space. The attention “mix” stage is just a matrix multiply (queries·keys) that selects which of those directions should contribute to the weighted sum, and the GRCE “mix” stage is even simpler: each block samples `n_embd → n_grce` features via a linear map and sums them. When a block wants to detect a pattern, it creates a probe vector aligned with the relevant positive directions and computes a dot product with the incoming channel. The result is a scalar score saying “how loudly is this pattern currently active?”

The probe can also subtract evidence: append a few antipattern directions (negative encodings) for features that should *not* coincide with the target pattern. Because dot products are linear, this contrastive construction still fits inside a single matrix multiply. If we collect M such probes we end up with an `N×M` projection, run its outputs through a ReLU/GELU (zeroing negative scores), and interpret the non-negative scalars as detected message strengths.

### Remixing back into the channel
The second linear map (`M×N`) takes those non-negative activations and re-encodes them as broadcastable updates on the channel. Positive activations add energy to their preferred directions; zeroed (formerly negative) activations mean “leave that direction alone.” Residual connections ensure that if a block decides “I don’t have anything new to say,” the channel simply flows through unchanged.

LayerNorm (or RMSNorm) between blocks keeps the energy roughly constant. If a downstream block still needs an upstream feature, backpropagation increases the weight on the residual path or trains the block to recreate that feature explicitly. If the signal is no longer useful, gradients push the block to cancel it and amplify something more relevant. Over depth this continual attenuation-and-refresh process makes the loss surface much smoother: early layers learn broad principles quickly, while later layers spend their capacity on increasingly specific refinements.

### Attention and GRCE in this picture
Self-attention is just another message mixer: the `QK^T` term selects which source tokens contribute to the message, and the value projection determines which directions get added to the channel. GRCE fits into the same mental model but rotates it over time instead of over tokens. Each layer emits a compressed “message” about its local view, the shared GRCE bottleneck mixes those across depth, and the next time step injects the decoded biases back into the channel. Because we only pass around additive messages (no full activations) the system stays easy to reason about.

When you picture the network this way, debugging becomes simpler: norm spikes mean “too much energy in the channel,” flat probes mean “no block bothered to speak,” and GRCE provides a narrow, well-defined lane where long-range biases can hitch a ride without polluting the main attention stream.

## Running grce.py

Call `grce.py --help` for the full CLI. In all examples below we assume a virtual environment at `.venv/`; overriding the interpreter is as simple as exporting `$PYTHON`, since every snippet uses `${PYTHON:-.venv/bin/python3}`.

Key switches:
- `--think N` enables the above thinking-token workflow (set `--no-think` to keep sampling clean while still training with thinking tokens). Think tokens (and undo tokens) always live in the tokenizer/embedding space, so you can import/export checkpoints between think/non-think runs without remapping vocabularies. One row per batch is automatically left in “plain” mode so the model keeps a steady diet of non-thinking updates.
- `--reward-relu SCALE` enables an experimental auxiliary update that periodically inspects the top/bottom 25% token predictions per sequence: active neurons that helped the good predictions and inactive neurons that hurt the bad ones accumulate credit, and at scheduled checkpoints the union of those counts is used to select roughly the top 20% of neurons for a tiny bias boost. The increment is `SCALE * std(bias_vector)` (with `SCALE=0` disabling the mechanism), so every nudge is relative to the layer’s own bias magnitude. Updates fire mid-cycle and at the end, and no extra state needs to be saved in checkpoints.
- `--corpus NAME` chooses which `<NAME>-train.txt.gz` / `<NAME>-test.txt.gz` split to load from the `--data DIR` directory (default `data/`).
- `--model DIR` selects where checkpoints, logs, and tokenizer caches live (default `model/`).
- `--import-model some.pt` seeds a new run from an existing checkpoint. Use `--drop-layers i,j,...` to delete specific source layers (1-indexed) and `--add-layers i,j,...` to specify where new randomly initialized layers should be inserted so the total matches the new `--n-layer`. `--trim-model` lets you shrink other tensor dimensions (embedding width, vocab, etc.) while copying whatever fits. The importer enforces that the number of attention heads (`--n-head`) stays the same and that every overlapping tensor slice lines up, carries over the total step counter, and writes a fresh `.pt` with an empty loss history.
- `--undo N` inserts up to `N` random+undo pairs per block (filler loss ignored, undo enforced).
- `--report-count N` skips training entirely, loads the latest checkpoint (if any), and prints `N` completions of the configured prompt.
- `--train` explicitly requests the standard training loop (this is the default when no other action is given).
- `--report N` skips training entirely, loads the latest checkpoint (if any), and prints `N` completions of the configured prompt.
- `--test N` dumps `block_size` tokens from the test split starting at cursor `N`; if the model supports thinking tokens it also runs the model over that window and inserts predicted `<think>` tokens in-line so you can inspect where the network wants to branch into reasoning mode.
- `--no-newlines` keeps the sampler from emitting newline tokens so completions stay on one line.
You can reproduce the sweeps below; notice how even the `--context-span 1` run (which detaches the recurrent gradients entirely) tracks all other spans almost perfectly, confirming that the channel only needs to learn what to sample, not how to backpropagate across positions.

Every evaluation logs both GRCE-enabled and GRCE-disabled losses, and `plot.py` draws both traces for quick comparison.

For postprocessing, run for example `plot.py --avg-span --no-nogrce --store span_avg.json --store-only` and feed that JSON into your analysis scripts.

## Example training sweeps

Below are two quick sweeps you can adapt. Both snippets assume a shell where `${PYTHON:-.venv/bin/python3}` resolves to your preferred interpreter.

1. **GRCE vs span variants.** Demonstrates the benefit of the GRCE path and the weak dependence on `--context-span`.

    ```bash
    time bash -exc '
    for cy in 2 3 5 10 10; do
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --n-grce 0
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --tag span0 --context-span 0
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --tag span1 --context-span 1
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --tag span2 --context-span 2
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --tag span3 --context-span 3
    done
    for cy in 20 20 30; do
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --n-grce 0
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --tag span2 --context-span 2
    done'
    ```

2. **Think vs non-think.** Compares the base model to a run with thinking tokens enabled.

    ```bash
    time bash -exc '
    for cy in 2 3 5 10 10 20 20 30; do
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy
	${PYTHON:-.venv/bin/python3} grce.py --cycles $cy --think 10
    done'
    ```

## Dev notes & agent cheat sheet

(Pretty much all code in this repo is AI-generated—but under strong human supervision. ~Claire)

- **Project focus:** Gradient-limited Recurrent Context Encoding (GRCE) atop a picoGPT-style SimpleWiki language model.
- **Model defaults:** `n_layer=8`, `n_head=8`, `n_embd=192`, `n_grce=96`, `block_size=64`, `dropout=0.05`, `vocab_size=2000`.
- **GRCE geometry:** each layer owns a sampler `LayerNorm → n_embd → n_grce`; sampled vectors are summed, passed through a shared MLP `n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`, then per-layer decoders `n_grce → n_embd` inject the biases.
- **Parameter dominance:** ignoring embeddings, the stack costs `~12 * n_layer * n_embd^2`, and the GRCE path adds `~2 * n_layer * n_embd * n_grce + 4 * n_grce^2` when enabled.
- **Tokenizer workflow:** Byte-level BPE trained on up to `--vocab-chars` characters; saved to `model/<trainstem>_<limit>_<vocab>.json` alongside checkpoints.
- **Chunks per cycle:** Train chunk size `(block_size+1)*batch_size*steps`; test chunk size `(block_size+1)*batch_size*eval_iters*eval_calls` (where `eval_calls` covers step 1, every `eval_interval`, and the final step).
- **Persistence:** Checkpoints store weights, dataset offsets, total step counter, cumulative wall-clock training seconds, and the loss table; log files mirror console output and capture tokenizer timing when the tokenizer is retrained.
- **Testing:** Eval prints test/train losses plus a colorized sample; prompts are cyan/green, completions yellow/magenta, and the GRCE-disabled loss is shown for comparison.
- **Local CPU sanity checks:** run a tiny model to keep turnaround fast:
  ```bash
  .venv/bin/python3 grce.py --device cpu --cycles 2 --steps 20 --block-size 16 --batch-size 4 --n-layer 2 --n-head 2 --n-embd 64 --n-grce 16 --context-span 4 --eval-interval 10 --eval-iters 1 --generate 5
  ```
  This fits in RAM and exercises the GRCE dropout path without a GPU.
**Environment note:** always run tooling via `.venv/bin/python3` (and related entrypoints) so the local dependencies are available; the system python may lack the required packages, or there even may be no system python.
