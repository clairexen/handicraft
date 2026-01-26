# GPT with Gradient-limited Recurrent Context Encoding and Extended Context (GPT+GRCE+XCTX)

This repo extends a tiny picoGPT-style language model with two recurrent context channels: A lightweight low-bandwith “role/focus” state alongside the usual token stream (GRCE), and a high-bandwidth "short-term-memory"-style channel, to send messages forward in time (XCTX), parallel to the multi-head attention mechanism, that looks backward. "GPT+GRCE+XCTX" is pronounced "GPT with grace and extended context".

This adds the following benefits:
- The low-bandwith "GRCE" channel mostly adds stability, debugability, and interpretability.
- The high-bandwith "XCTX" channel is meant to be functionally equivalent to the multi-head attention mechanism. It's a (simplistic and thus probably worse ;) recurrent re-implementation of the same functinality, that we can only learn because we use the transformer stack and its multi-head attention as "scaffolding". Instead of sending queries into the past we are shuting the things worth remembering for a little while into the future. We use a dropout-like mechanism during learning to encourage the network to learn that functionality, that is redundant within a token block. And then we use that learned functionality to both pass messages forward in time from one block to the next in inference, and prevent the network from doing weird things at the same time. The attention mechanism is great, when you know what you want to know from the past. Context is a way for the past to let the future know what to query.

Training and sampling logic all lives in `grce.py`.

## How the context channel works

1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each block input goes through LayerNorm + a linear bottleneck `n_embd → n_grce`. The `n_layer` message vectors share weights across timesteps but are detached according to `--detach-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step).
3. **Context propagation.** All per-layer messages are summed, the (optionally detach-controlled) previous context is added once, and that combined vector is *normalized again* before entering the shared MLP `n_grce → 4*n_grce → ReLU → n_grce`. The MLP output then adds the same previous context again—just like a Transformer residual—before a final LayerNorm produces the next-step context. That LayerNormed output is the sole signal that crosses positions—nothing else persists across time.
4. **Bias injection.** At the next position every block applies a single linear decoder `n_grce → n_embd`. The generated biases are added only to the newest token row of each block input, so the rest of the sequence remains untouched while the context acts as an additive steering signal.

In other words, this channel is literally the recurrent shortcut that classic RNNs tried to build, but it is implemented as a clean add-on to the Transformer stack: Each block, while computing logits for token N+1, already contains every piece of context needed to describe the prefix. The GRCE path just samples that information, compresses it into `n_grce` scalars, mixes them with a single hidden layer in the time domain, and feeds the signal into the very next step. Nothing else has to travel across time. Training stays stable because gradients do not need to propagate across multiple positions; the heavy lifting is still performed inside the per-token Transformer layers.

This mechanism creates an explicit channel for time-domain (i.e. recurrent) signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable patterns** that act like on/off switches describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing patterns** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

Think of it this way: the original “Attention Is All You Need” insight was to rotate an interleaved recurrent stack by 90°, replace the fixed unit selector ("use the same layer from the previous token") with attention, and thereby remove the long gradient paths that made RNNs hard to train. GRCE rotates us back over depth, but instead of letting layers browse earlier positions, every layer at position N sends a small learned message to position N+1. A shared bottleneck MLP mixes those messages, and each layer in the next position adds a linear bias from the shared vector. That gives us the benefits of a depth-wise recurrent shortcut (context and role persistence) without the hard-to-train time-domain gradients—everything needed to predict the next token is already inside the previous stack, so GRCE just samples and forwards it.

The extended context (XCTX) variant follows the same sampler, residual, and per-layer bias workflow; its shared MLP just runs narrower (`n_xctx → 4*n_xctx//n_layer → ReLU → n_xctx`) so the wide channel stays parameter-efficient even though it carries more activations. That keeps the signal high-bandwidth while still constraining it to `n_xctx` scalars.

One way to view this geometry is that attention “killed” classic recurrence by rotating the computation over depth and letting every position look backward. GRCE rotates a slim slice of that structure back into the time axis, but it keeps the Transformer philosophy—tight bottlenecks, shared emitters/decoders, and a single shared nonlinear mix—so gradients never have to walk through time. The heavy lifting stays inside the per-token Transformer blocks; the recurrent shortcut simply recycles whatever features those blocks already distilled.

## Sequence boundaries and recurrent prefill

The corpus is stored with a trailing record separator so it is always safe to treat it as a looped scroll. All low-level slice helpers accept windows that cross the file boundary (even negative offsets) and quietly wrap the indices. Every sampled batch therefore knows both its `(block_size + 1)` training window and the `block_size` tokens that immediately preceded it in the corpus.

Before running the “real” block we feed that preceding window through the model exactly once (with the usual settings) and capture the raw GRCE/XCTX vectors emitted by the last position—these are the fused, pre-LayerNorm values that would otherwise become the residual input to the next step. Those raw vectors are saved per channel and injected into the actual block so each layer’s bias generator sees the same recurrent state it would have seen if we had streamed tokens through continuously. All later special-row experiments (context dropout, attention punctures, additional evaluation modes, etc.) reuse the cached vectors, so the prefill happens just once per batch regardless of how many forward passes we run.

This means any layer at position N can send a context-related message to any layer at position N+1, and the training signal never has to cross the position boundary: by the time we emit the message, the previous stack has already computed everything it needs to predict the next token. In practice (see the sweeps in this repo), even `--detach-span 1`—which suppresses cross-position gradients entirely—matches the default span: the channel just learns how to sample the information that already exists inside the previous position’s layers.

## GRCE row types, training target, and losses reported during training

### Row types in GRCE batches

Training and evaluation revolve around *row types*—deterministic ways of mutating a batch so the model practices specific failure modes:

- **`plain`** – disables both GRCE and XCTX entirely so the GPT core behaves like a pure Transformer. Used as one of the stress rows and for the dedicated “plain” evaluation column.
- **`normal`** – the baseline path: GRCE, XCTX, and attention all enabled with the standard causal mask.
- **`encode`** – identical to `plain` in that both GRCE and XCTX are disabled; retained as a dedicated diagnostic row so logs remain comparable to older runs.
- **`noxctx` / `puxctx`** – remove the wide XCTX signal either for the entire row (`noxctx`) or by puncturing it at a single random timestep (`puxctx`) while GRCE remains active.
- **`noattn` / `puattn`** – shut attention off entirely (`noattn`) or mask a single timestep’s ability to transmit forward (`puattn`) so the recurrent channels have to carry the load.

### How these row types shape a training batch

Let `n_total` be the batch size. Each batch is a fixed mixture of row types:

- `n_noxctx = n_puxctx = n_noattn = n_puattn = n_encode = 1`
- The remaining rows are split between `plain` (roughly half of what remains) and standard `normal` rows so the optimizer keeps seeing baseline sequences.

That means every batch carries exactly one copy of each structural ablation (pure Transformer, no-XCTX, punctured-XCTX, no-attention, punctured-attention) plus a healthy mix of `normal` rows. The ordering varies per batch because we randomly assign row indices when building the masks, but the counts above remain fixed.

### Loss reporting

The live log and checkpoint history capture two views of the training objective:

1. **Training target (`target`).** This is the exact loss the optimizer just saw on the mixed batch of row types.
2. **Row-type diagnostics.** During evaluation we reuse that same batch and aggregate the per-row losses to expose conditional metrics for `normal`, `plain`, `noxctx`, `puxctx`, `noatt`, `none` (the attention-punctured row), and `encode`. Because every value comes from a single forward pass, all reported losses are directly comparable slices of the training objective.

## Parameter count (dominant terms)

Remember: when the model computes logits for position N+1, it already synthesized every feature it needs about the prefix—that’s what autoregressive prediction is. GRCE simply taps into that already-available context and moves it forward; it does not have to learn new facts across the boundary. That’s why gradients from position N+1 flowing back into position N via the GRCE channel are largely unnecessary: the previous stack has already computed the relevant summary while predicting the token. Training just has to learn which latent features to sample and forward through the n_grce bottleneck.

Ignoring embeddings and other lower-order pieces, two terms dominate:

- Position-domain Transformer stack: `~ 12 * n_layer * n_embd^2`
- Time-domain GRCE network (only if `n_grce > 0`): `~ 2 * n_layer * n_embd * n_grce + 8 * n_grce^2`
- Time-domain XCTX network (only if `n_xctx > 0`): `~ 2 * n_embd * n_xctx + 2 * n_xctx^2 + 8 * n_xctx^2 / n_layer`

For exact counts (including the XCTX channel and bias/sampler splits) run `python grce.py size [--check]` with your chosen hyperparameters—the report prints every contribution with its closed-form formula and can optionally instantiate a model to verify the arithmetic.

Thinking about the stack from a geometric point of view helps explain why the recurrent shortcut is viable: attention “killed” classical recurrence by rotating the computation over depth, letting every position look backwards instead of pushing state forward. GRCE rotates a slim slice of that structure back into the time axis, but it keeps the same design philosophy—tight bottlenecks, shared samplers/decoders, and a single shared nonlinearity—so gradients never have to march through time. The heavy lifting still happens in the standard Transformer layers; the recurrent channels just recycle whatever features those layers already extracted.

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

## Dev notes & agent cheat sheet

See the AGENTS.md and notes.txt files for more information.
