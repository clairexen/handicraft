# GPT with Gradient-limited Recurrent Context Encoding and Extended Context (GPT+GRCE+XCTX)

This repo extends a tiny picoGPT-style language model with two recurrent context channels: A lightweight low-bandwith “role/focus” state alongside the usual token stream (GRCE), and a high-bandwidth "short-term-memory"-style channel, to send messages forward in time (XCTX), parallel to the multi-head attention mechanism, that looks backward. "GPT+GRCE+XCTX" is pronounced "GPT with grace and extended context".

This adds the following benefits:
- The low-bandwith "GRCE" channel mostly adds stability, debugability, and interpretability.
- The high-bandwith "XCTX" channel is meant to be functionally equivalent to the multi-head attention mechanism. It's a (simplistic and thus probably worse ;) recurrent re-implementation of the same functinality, that we can only learn because we use the transformer stack and its multi-head attention as "scaffolding". Instead of sending queries into the past we are shuting the things worth remembering for a little while into the future. We use a dropout-like mechanism during learning to encourage the network to learn that functionality, that is redundant within a token block. And then we use that learned functionality to both pass messages forward in time from one block to the next in inference, and prevent the network from doing weird things at the same time. The attention mechanism is great, when you know what you want to know from the past. Context is a way for the past to let the future know what to query.

Training and sampling logic all lives in `grce.py`.

## How the context channel works
1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each block input goes through LayerNorm + a linear bottleneck `n_embd → n_grce`. The `n_layer` message vectors share weights across timesteps but are detached according to `--detach-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step).
3. **Context propagation.** All per-layer messages are summed, the (optionally detach-controlled) previous context is added into that fused vector, and the result is run through a shared MLP `n_grce → 4*n_grce → ReLU → n_grce`. The MLP output then adds the same (optionally detach-controlled) previous context again—just like a Transformer residual—before a final LayerNorm produces the next-step context. That LayerNormed output is the sole signal that crosses positions—nothing else persists across time.
4. **Bias injection.** At the next position every block applies a single linear decoder `n_grce → n_embd`. The generated biases are added only to the newest token row of each block input, so the rest of the sequence remains untouched while the context acts as an additive steering signal.

In other words, this channel is literally the recurrent shortcut that classic RNNs tried to build, but it is implemented as a clean add-on to the Transformer stack: Each block, while computing logits for token N+1, already contains every piece of context needed to describe the prefix. The GRCE path just samples that information, compresses it into `n_grce` scalars, mixes them with a single hidden layer in the time domain, and feeds the signal into the very next step. Nothing else has to travel across time. Training stays stable because gradients do not need to propagate across multiple positions; the heavy lifting is still performed inside the per-token Transformer layers.

This mechanism creates an explicit channel for time-domain (i.e. recurrent) signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable patterns** that act like on/off switches describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing patterns** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

Think of it this way: the original “Attention Is All You Need” insight was to rotate an interleaved recurrent stack by 90°, replace the fixed unit selector ("use the same layer from the previous token") with attention, and thereby remove the long gradient paths that made RNNs hard to train. GRCE rotates us back over depth, but instead of letting layers browse earlier positions, every layer at position N sends a small learned message to position N+1. A shared bottleneck MLP mixes those messages, and each layer in the next position adds a linear bias from the shared vector. That gives us the benefits of a depth-wise recurrent shortcut (context and role persistence) without the hard-to-train time-domain gradients—everything needed to predict the next token is already inside the previous stack, so GRCE just samples and forwards it.

The extended context (XCTX) variant follows the same sampler, residual, and per-layer bias workflow, but it removes the intermediate MLP entirely. XCTX simply sums the projected messages (plus the optional residual context), performs a LayerNorm, and feeds that normalized vector directly into the bias generators. That keeps the signal high-bandwidth while still constraining it to `n_xctx` scalars.

This means any layer at position N can send a context-related message to any layer at position N+1, and the training signal never has to cross the position boundary: by the time we emit the message, the previous stack has already computed everything it needs to predict the next token. In practice (see the sweeps in this repo), even `--detach-span 1`—which suppresses cross-position gradients entirely—matches the default span: the channel just learns how to sample the information that already exists inside the previous position’s layers.

One way to view this is: attention killed recurrence by rotating the computation over depth and letting every position look backwards. GRCE rotates part of that structure back, but with the same philosophy—keep the bottleneck tight, use learned linear emitters/decoders, and place the only nonlinear mixing in a single shared layer. The result is a recurrent path that is just as easy to train as the rest of the Transformer because gradients never have to walk through time.

## Parameter count (dominant terms)
Remember: when the model computes logits for position N+1, it already synthesized every feature it needs about the prefix—that’s what autoregressive prediction is. GRCE simply taps into that already-available context and moves it forward; it does not have to learn new facts across the boundary. That’s why gradients from position N+1 flowing back into position N via the GRCE channel are largely unnecessary: the previous stack has already computed the relevant summary while predicting the token. Training just has to learn which latent features to sample and forward through the n_grce bottleneck.

Ignoring embeddings and other lower-order pieces, two terms dominate:

- Position-domain Transformer stack: `~ 12 * n_layer * n_embd^2`
- Time-domain GRCE network (only if `n_grce > 0`): `~ 2 * n_layer * n_embd * n_grce + 4 * n_grce^2`

## Think tokens

`--think N` means “activate thinking on exactly `N` sequences per batch” in training. The training loop first evaluates the untouched batch to collect logits, then chooses the first `N` eligible rows (plus one extra when context dropout fires). For each chosen row we sample three non-negative integers `R`, `T`, and `H` by drawing `u ~ U[0,1)`, squaring it, multiplying by `block_size/4`, and flooring. These act as simple compute budgets:

1. **Repeat phase (`R`).** After every base token we insert `R` `<think>` tokens, truncating any overflow. When `R>0` this dramatically shrinks the effective context and forces the model to reuse the same slot multiple times.
2. **Targeted inserts (`T`).** We look at the logits we saved earlier and insert `T` more `<think>` tokens, sampling positions proportionally to their probability of emitting `<think>`. This gives the model practice placing thought where it already “wants” it.
3. **Tail padding (`H`).** Finally we append `<think>` tokens just before the block boundary until we have inserted at least `H` trailing planners, ensuring the network also learns to finish a thought explicitly.

Undo fillers (when `--undo` is active) are spliced in before all of the above, and one sequence per batch is always left plain so the model keeps calibrating on non-thinking data. When `--context-dropout-interval` fires (and at least one context channel is enabled) we carve out a suite of special rows: (1) a pure-Transformer baseline with GRCE/XCTX fully disabled, (2) a “punctured context” row where the recurrent state is zeroed at a single random timestep, (3) a “random think” row (only when `<think>` is enabled) where context flows normally but a random number of `<think>` tokens are sprinkled arbitrarily, (4) an attention-disabled row whose multi-head attention is muted so it must rely entirely on the recurrent channels, and (5) an attention-punctured row whose single random timestep cannot send information forward via attention (its value stream is masked for all later positions). Every dropout step therefore trains the model on “no context,” “punctured context,” “random think,” “no attention,” and “punctured attention” scenarios simultaneously, and the training log mirrors those experiments via the `noctx` and `noatt` columns.

For every `<think>` slot we compute the normal CE using a decoder with the `<think>` logit disabled (so the loss remains comparable to non-thinking runs). We also compute an **alignment loss**: let `A` be that CE at position `N`, `B` the CE at `N+1`, and `ratio = A/(A+B+ε)`. We blend the next-token embedding with the `<think>` embedding by `ratio`, decode it, and compare that soft distribution against the *unmodified* decoder output. Conceptually this enforces the intended behavior: early `<think>` evaluations still carry some mixture of “target vs. reasoning”, while the final stack converges on the exact target token. The alignment loss replaces the old plan head/penalty system.

During sampling/reporting we flip a coin for each completion: heads means argmax, tails means sampling from the predicted distribution. Prompts that have only been solved via the random path stay in the queue and are retried with argmax until the top candidate alone satisfies the goal. The progress header shows `sample (random-only/argmax/total)` so you can track both counts at a glance, and completions generated via argmax are the only ones rendered in bold.

The recurrent context has two width knobs. `--n-grce` controls a narrow, low-bandwidth GRCE bottleneck, while `--n-xctx` enables a wider channel that uses the same overall method with minor modifications and a much larger context vector space. Use either on its own or enable both—their projected biases simply add before each Transformer block—so you can mix a steady recurrent signal with a higher-bandwidth shortcut.

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

Call `grce.py --help` for the full CLI.

Key switches:
- `--think N` enables the above thinking-token workflow (set `--no-think` to keep sampling clean while still training with thinking tokens). Think tokens (and undo tokens) always live in the tokenizer/embedding space, so you can import/export checkpoints between think/non-think runs without remapping vocabularies. One row per batch is automatically left in “plain” mode so the model keeps a steady diet of non-thinking updates.
- `--reward-relu VALUE` enables an experimental auxiliary update that periodically inspects the top/bottom 25% token predictions per sequence: active neurons that helped the good predictions and inactive neurons that hurt the bad ones accumulate credit, and at scheduled checkpoints the union of those counts is used to select roughly the top 20% of neurons for a tiny bias boost. The increment is `10^{-VALUE} * std(bias_vector)` (VALUE ≤ 0 disables the mechanism), so every nudge is relative to the layer’s own bias magnitude while letting you specify the multiplier in log10 form. Updates fire mid-cycle and at the end, and no extra state needs to be saved in checkpoints.
- `--corpus NAME` chooses which `<NAME>-train.txt.gz` / `<NAME>-test.txt.gz` split to load from the `--data DIR` directory (default `data/`).
- `--model DIR` selects where checkpoints, logs, and tokenizer caches live (default `model/`).
- `--import-model some.pt` seeds a new run from an existing checkpoint. Use `--drop-layers i,j,...` to delete specific source layers (1-indexed) and `--add-layers i,j,...` to specify where new randomly initialized layers should be inserted so the total matches the new `--n-layer`. `--trim-model` lets you shrink other tensor dimensions (embedding width, vocab, etc.) while copying whatever fits. The importer enforces that the number of attention heads (`--n-head`) stays the same and that every overlapping tensor slice lines up, carries over the total step counter, and writes a fresh `.pt` with an empty loss history.
- `--undo N` inserts up to `N` random+undo pairs per block (filler loss ignored, undo enforced).
- `train` is the main command and runs the standard training loop with the configured cycles/steps.
- `report -n N` (or `report --count N`) loads the latest checkpoint and prints `N` prompt completions without running another training cycle.
- `test --start N` dumps `block_size` tokens from the test split starting at cursor `N`; if the model supports thinking tokens it also runs the model over that window and inserts predicted `<think>` tokens in-line so you can inspect where the network wants to branch into reasoning mode.
- `size` prints the configured model’s parameter breakdown (respecting the usual geometry flags) and exits.
- `print-train START-END` / `print-test START-END` emit token ranges from the respective corpus splits so you can debug the raw data without kicking off a training run.
- `init` builds (or refreshes) the tokenizer cache if needed and writes a brand-new checkpoint with zeroed training counters so you can stage experiments or clone configs without running a cycle.
- `--pt some_checkpoint.pt` lets you run any of the non-training commands directly against an explicit checkpoint file; all geometry parameters are lifted from the file so you don’t have to mirror the original CLI flags (and with `init` it simply names the destination checkpoint).
- `--no-newlines` keeps the sampler from emitting newline tokens so completions stay on one line.
You can reproduce the sweeps below; notice how even the `--detach-span 1` run (which detaches the recurrent gradients entirely) tracks all other spans almost perfectly, confirming that the channel only needs to learn what to sample, not how to backpropagate across positions.

## Example training sweeps

Below are two quick sweeps you can adapt.

1. **GRCE vs span variants.** Demonstrates the benefit of the GRCE path and the weak dependence on `--detach-span`.

    ```bash
    time bash -exc '
    for cy in 2 3 5 10 10; do
	python3 grce.py --cycles $cy --n-grce 0
	python3 grce.py --cycles $cy --tag span0 --detach-span 0
	python3 grce.py --cycles $cy --tag span1 --detach-span 1
	python3 grce.py --cycles $cy --tag span2 --detach-span 2
	python3 grce.py --cycles $cy --tag span3 --detach-span 3
    done
    for cy in 20 20 30; do
	python3 grce.py --cycles $cy --n-grce 0
	python3 grce.py --cycles $cy --tag span2 --detach-span 2
    done
    ```

    ```bash
    # python hist.py --write-json-dir sweep1
    python sweep1.py
    ```

2. **GRCE small vs wide.**

    ```bash
    time bash -exc '
    for cy in 2 3 5 10 10; do
	python3 grce.py --cycles $cy --n-grce 32
	python3 grce.py --cycles $cy --n-grce 256
    done'
    ```

3. **Think vs non-think.** Compares the base model to a run with thinking tokens enabled.

    ```bash
    time bash -exc '
    for cy in 2 3 5 10 10 20 20 30; do
	python3 grce.py --cycles $cy
	python3 grce.py --cycles $cy --think 10
    done'
    ```

## Dev notes & agent cheat sheet

(Pretty much all code in this repo is AI-generated—but under strong human supervision. ~Claire)

- **Project focus:** Gradient-limited Recurrent Context Encoding (GRCE) atop a picoGPT-style SimpleWiki language model.
- **Model defaults:** `n_layer=8`, `n_head=8`, `n_embd=192`, `n_grce=96`, `block_size=64`, `dropout=0.05`, `vocab_size=2000`.
- **GRCE geometry:** each layer owns a sampler `LayerNorm → n_embd → n_grce`; sampled vectors are summed, passed through a shared MLP `n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`, then per-layer decoders `n_grce → n_embd` inject the biases.
- **XCTX geometry:** identical workflow, just swapping `n_grce` for `n_xctx` so the channel has more headroom; the sampler projects down to `n_xctx // n_layer` for every layer before projecting onto `n_xctx`, the bias-injectors project onto `n_xctx // n_layer` for every layer before projecting onto `n_embd`, and the shared MLP is removed so the summed messages (plus any residual context) flow straight into LayerNorm.
- **Parameter dominance:** ignoring embeddings, the stack costs `~12 * n_layer * n_embd^2`, and the GRCE path adds `~2 * n_layer * n_embd * n_grce + 4 * n_grce^2` when enabled.
- **Tokenizer workflow:** Byte-level BPE trained on up to `--vocab-chars` characters; saved to `model/<trainstem>_<limit>_<vocab>.json` alongside checkpoints.
- **Chunks per cycle:** Train chunk size `(block_size+1)*batch_size*steps`; test chunk size `(block_size+1)*batch_size*eval_iters*eval_calls` (where `eval_calls` covers step 1, every `eval_interval`, and the final step).
- **Persistence:** Checkpoints store weights, dataset offsets, total step counter, cumulative wall-clock training seconds, and the loss table; log files mirror console output and capture tokenizer timing when the tokenizer is retrained.
- **Testing:** Eval prints test/train losses plus a colorized sample; prompts are cyan/green, completions yellow/magenta, and the GRCE-disabled loss is shown for comparison.
- **Local CPU sanity checks:** run a tiny model to keep turnaround fast:
  ```bash
  .venv/bin/python3 grce.py --device cpu --cycles 2 --steps 20 --block-size 16 --batch-size 4 --n-layer 2 --n-head 2 --n-embd 64 --n-grce 16 --detach-span 4 --eval-interval 10 --eval-iters 1 --generate 5
  ```
  This fits in RAM and exercises the context dropout path without a GPU.
- **Detaching parts of the stack:** `--detach-layer K` severs gradients after Transformer layer `K` (1-based), letting you freeze the lower stack while training fresh layers on top.
- **Context dropout:** `--context-dropout-interval N` (default `1`) reserves five diagnostic rows every `N` steps when a context channel is active—pure Transformer, context puncture, random-think (when `<think>` is enabled), attention-disabled, and attention-punctured—so the model continuously practices each failure mode.
**Environment note:** always run tooling via `.venv/bin/python3` (and related entrypoints) so the local dependencies are available; the system python may lack the required packages, or there even may be no system python.
