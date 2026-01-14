# Gradient-limited Recurrent Context Encoding (GRCE)

This repo extends a tiny picoGPT-style language model with a recurrent context channel that keeps a lightweight “role/focus” state alongside the usual token stream. Training and sampling logic lives in `grce.py`.

## How the context channel works
1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each block input goes through LayerNorm + a linear bottleneck `n_embd → n_grce`. The `n_layer` message vectors share weights across timesteps but are detached according to `--context-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step).
3. **Context propagation.** All per-layer messages are summed and run through a shared MLP `n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`. The LayerNormed output is the sole context vector for the next position—nothing else persists across time.
4. **Bias injection.** At the next position every block applies a single linear decoder `n_grce → n_embd`. The generated biases are added only to the newest token row of each block input, so the rest of the sequence remains untouched while the context acts as an additive steering signal.

This mechanism creates an explicit channel for time-domain (i.e. recurrent) signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable patterns** that act like on/off switches describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing patterns** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

GRCE acts like attention rotated over depth: each layer emits a fixed linear summary, the summaries are combined through a shared bottleneck MLP, and every layer decodes the shared message with a linear bias. No query/key routing is needed, so the channel stays bottlenecked at `n_grce` scalars while still steering the next time step.

## Parameter count (dominant terms)
Ignoring embeddings and other lower-order pieces, two terms dominate:

- Position-domain Transformer stack: `~ 12 * n_layer * n_embd^2`
- Time-domain GRCE network (only if `n_grce > 0`): `~ 2 * n_layer * n_embd * n_grce + 4 * n_grce^2`

## Running it
Use `python grce.py --help` for CLI options. Main experiment:
Every evaluation logs both GRCE-enabled and GRCE-disabled losses, and `plot.py` draws both traces for quick comparison.

```
time bash -exc '
for cy in 2 3 5 10 10; do
	python grce.py --cycles $cy --context-span 0
	python grce.py --cycles $cy --context-span 1
	python grce.py --cycles $cy --context-span 2
	python grce.py --cycles $cy --context-span 3
	python grce.py --cycles $cy --n-grce 0
done
for cy in 20 50; do
	python grce.py --cycles $cy
	python grce.py --cycles $cy --n-grce 0
done
'
```

(pretty much all code in this repo is ai-generated. but of course only under my strong supervision.. ~Claire ;)
