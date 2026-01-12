# Gradient-limited Recurrent Context Encoding (GRCE)

This repo extends a tiny picoGPT-style language model with a recurrent context channel that keeps a lightweight “role/focus” state alongside the usual token stream. Training and sampling logic lives in `grce.py`.

## How the context channel works
1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each captured input goes through a block-local sampler `LayerNorm → n_embd → 4*n_grce → ReLU → Dropout → n_grce`. The sampler outputs share weights across timesteps but are detached according to `--context-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step).
3. **Context propagation.** The `n_layer` sampler outputs are averaged, squashed with `tanh`, and forwarded as the single context vector for the next position—nothing else persists across time.
4. **Bias injection.** At the next position every block applies its own generator `LayerNorm → n_grce → 4*n_grce → ReLU → Dropout → n_embd`. The generated biases are added only to the newest token row of each block input, so the rest of the sequence remains untouched while the context acts as an additive steering signal.

This mechanism creates an explicit channel for time-domain (i.e. recurrent) signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable patterns** that act like on/off switches describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing patterns** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

## Parameter count (dominant terms)
Ignoring embeddings and other lower-order pieces, the learned weights are dominated by two expressions:

- Position-domain Transformer stack: `~ 12 * n_layer * n_embd^2`
- Time-domain GRCE network (only if `n_grce > 0`): `~ 8 * n_layer * (n_embd * n_grce + n_grce^2)`

These come directly from the quadratic QKV/FFN projections and the two GRCE MLPs; everything else is small compared to the squared terms.

## Running it
Use `python grce.py --help` for CLI options. Main experiment:

```
set -x
for cy in 2 3 5 10 20; do
	python grce.py --cycles $cy --n-grce 0
	python grce.py --cycles $cy --context-span 0
	python grce.py --cycles $cy --context-span 1
	python grce.py --cycles $cy --context-span 2
	python grce.py --cycles $cy --context-span 3
done
```

(pretty much all code in this repo is ai-generated. but of course only under my strong supervision.. ~Claire ;)
