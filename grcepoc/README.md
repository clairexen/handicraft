# GPT with Gradient-limited Recurrent Context Encoding and Extended Context (GPT+GRCE+XCTX)

This repo extends a tiny picoGPT-style language model with two recurrent context channels: A lightweight low-bandwith “role/focus” state alongside the usual token stream (GRCE), and a high-bandwidth "short-term-memory"-style channel, to send messages forward in time (XCTX), parallel to the multi-head attention mechanism, that looks backward. "GPT+GRCE+XCTX" is pronounced "GPT with grace and extended context".

This adds the following benefits:
- The low-bandwith "GRCE" channel mostly adds stability, debugability, and interpretability.
- The high-bandwith "XCTX" channel is meant to be functionally equivalent to the multi-head attention mechanism. It's a (simplistic and thus probably worse ;) recurrent re-implementation of the same functinality, that we can only learn because we use the transformer stack and its multi-head attention as "scaffolding". Instead of sending queries into the past we are shuting the things worth remembering for a little while into the future. We use a dropout-like mechanism during learning to encourage the network to learn that functionality, that is redundant within a token block. And then we use that learned functionality to both pass messages forward in time from one block to the next in inference, and prevent the network from doing weird things at the same time. The attention mechanism is great, when you know what you want to know from the past. Context is a way for the past to let the future know what to query.

Training and sampling logic all lives in `grce.py`. The CLI separates the model's
maximum positional range (`--n-pos`) from the runtime window (`--block-size`),
so you can keep the checkpoint's full set of positional embeddings while training on
shorter slices that randomly slide across the corpus.

## Batch layout mini-language

`--layout` replaces the old maze of `--rows-*`/`--cols-*` flags and describes the
entire training/eval batch with a compact string. Each term is `ROWS[SEGMENTS]`
where `SEGMENTS` is a `/`-delimited list of `COLS+MODE` tokens (`e`, `d`, `f`, `n`
for encode/decode/forward/noattn). Ranges use `A-B`, optional `*` prefixes mark
segments that can expand/shrink to fill up to the configured `--batch-size`, and
parenthetical `(a|b|c)` choices are expanded before parsing. The `+` prefix behaves
like `*` but caps growth at the specified maximum (`+12` is shorthand for `+1-12`).
Rows separated by `+` belong to the same micro-batch, while commas split the layout
into gradient-accumulation micro-batches. Everything between two commas must fit on
the GPU; we run forward/backward on that slice, accumulate gradients, then advance to
the next slice before calling `optimizer.step()`. For example

```
128[64f](,1[1024f]|||)
```

produces a 128×64 forward micro-batch, and with 25% probability appends a second
micro-batch of size 1×1024 before the optimizer update. Another sample layout,
`2[*d]+2[*f]+*[*1-2e/*1-4d/*1-4f/*1-2n]`, keeps two full decode and forward rows
while filling the remainder with a randomized encode/decode/forward/no-attention
pattern.

## How the context channel works

1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each block input goes through LayerNorm plus a learned projection `n_embd → n_grce` to produce per-layer messages `V_i`. Messages are detached according to `--detach-span`, so gradients never travel through time for more than a few steps.
3. **Context propagation.** The previous recurrent state `G` is added to the sum of sampled messages, LayerNorm'ed, pushed through the shared MLP `n_grce → 4*n_grce → ReLU → n_grce`, and then passed through a LayerDampening residual `G' = LD(G + Δ, with_gain=False)` instead of a plain LayerNorm. LD softly shrinks large updates while letting small features travel untouched, so the recurrent signal stays stable without being forced onto a unit sphere.
4. **Bias injection.** Before processing the next token each block consumes both context channels via `LN(S_i + α ⊙ X_i) + LD(G_i)` where `α` is a learned per-feature gain, injecting the result only into the newest token row so the GRCE/XCTX biases steer attention and MLP work without disturbing the rest of the window.

In other words, this channel is literally the recurrent shortcut that classic RNNs tried to build, but it is implemented as a clean add-on to the Transformer stack: Each block, while computing logits for token N+1, already contains every piece of context needed to describe the prefix. The GRCE path just samples that information, compresses it into `n_grce` scalars, mixes them with a single hidden layer in the time domain, and feeds the signal into the very next step. Nothing else has to travel across time. Training stays stable because gradients do not need to propagate across multiple positions; the heavy lifting is still performed inside the per-token Transformer layers.

This mechanism creates an explicit channel for time-domain (i.e. recurrent) signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable patterns** that act like on/off switches describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing patterns** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

Think of it this way: the original “Attention Is All You Need” insight was to rotate an interleaved recurrent stack by 90°, replace the fixed unit selector ("use the same layer from the previous token") with attention, and thereby remove the long gradient paths that made RNNs hard to train. GRCE rotates us back over depth, but instead of letting layers browse earlier positions, every layer at position N sends a small learned message to position N+1. A shared bottleneck MLP mixes those messages, and each layer in the next position adds a linear bias from the shared vector. That gives us the benefits of a depth-wise recurrent shortcut (context and role persistence) without the hard-to-train time-domain gradients—everything needed to predict the next token is already inside the previous stack, so GRCE just samples and forwards it.

### Extended context (XCTX)

XCTX follows the same capture/mix/remix template while widening the channel so it can behave more like a short-term memory buffer:

1. Each block input is LayerNorm'ed, projected down to an intermediate width `u`, re-normalized, and expanded back to `n_xctx`. A matching decoder maps `LN(X)` back to token-space biases via `X2U → U2E`.
2. The recurrent state `X` is added to the sum of sampled messages, passed through a learned down-projection `DnX`, and LayerNorm'ed before it enters the shared mixer.
3. The mixed vector is expanded with `UpX`, pushed through a learned `W` projection with ReLU, and added to the previous state before a final RMSNorm produces `X'`. RMSNorm keeps the direction of the recurrent signal stable so the channel can accumulate information over many steps.

Because both channels bolt onto the same Transformer spine, we can vary their widths or disable them entirely with CLI flags while leaving the core model unchanged.

One way to view this geometry is that attention “killed” classic recurrence by rotating the computation over depth and letting every position look backward. GRCE rotates a slim slice of that structure back into the time axis, but it keeps the Transformer philosophy—tight bottlenecks, shared emitters/decoders, and a single shared nonlinear mix—so gradients never have to walk through time. The heavy lifting stays inside the per-token Transformer blocks; the recurrent shortcut simply recycles whatever features those blocks already distilled.

## Loss reporting

The live log and checkpoint history capture two views of the training objective:

1. **Training target (`target`).** This is the exact loss the optimizer just saw on the mixed batch of row types.
2. **normal.** Loss on the model in "fully functional" mode with both GRCE and XCTX enabled.
2. **decode.** Loss on the model in "standard transformer" mode with GRCE and XCTX disabled.

## Parameter count (dominant terms)

Remember: when the model computes logits for position N+1, it already synthesized every feature it needs about the prefix—that’s what autoregressive prediction is. GRCE simply taps into that already-available context and moves it forward; it does not have to learn new facts across the boundary. That’s why gradients from position N+1 flowing back into position N via the GRCE channel are largely unnecessary: the previous stack has already computed the relevant summary while predicting the token. Training just has to learn which latent features to sample and forward through the n_grce bottleneck.

Ignoring embeddings and other lower-order pieces, two terms dominate:

- Position-domain Transformer stack: `~ 12 * n_layer * n_embd^2`
- Time-domain GRCE network (only if `n_grce > 0`): `~ 2 * n_layer * n_embd * n_grce + 8 * n_grce^2`
- Time-domain XCTX network (only if `n_xctx > 0`): `~ 2 * n_embd * n_xctx + 2 * n_xctx^2 + 8 * n_xctx^2 / n_layer`

For exact counts (including the XCTX channel and bias/sampler splits) run `python grce.py size [--check]` with your chosen hyperparameters—the report prints every contribution with its closed-form formula and can optionally instantiate a model to verify the arithmetic.

Thinking about the stack from a geometric point of view helps explain why the recurrent shortcut is viable: attention “killed” classical recurrence by rotating the computation over depth, letting every position look backwards instead of pushing state forward. GRCE rotates a slim slice of that structure back into the time axis, but it keeps the same design philosophy—tight bottlenecks, shared samplers/decoders, and a single shared nonlinearity—so gradients never have to march through time. The heavy lifting still happens in the standard Transformer layers; the recurrent channels just recycle whatever features those layers already extracted.

## Getting Started

Setting up and testing the toolchain:

    ```bash
    # setup python env
    python3 -m venv .venv
    source activate
    pip install -r requirements.txt

    # tokenize 'simplestwiki' corpus
    cd data
    bash simplestwiki-setup.sh
    cd ..

    # create model and run a training loop
    grce --tiny create
    grce --tiny corpus --set simplestwiki
    grce --tiny --log-step-details train
    ```

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
