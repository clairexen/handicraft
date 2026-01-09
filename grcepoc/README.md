# Gradient-limited Recurrent Context Encoding (GRCE)

This repo extends a tiny picoGPT-style language model with a recurrent context channel that keeps a lightweight “role/focus” state alongside the usual token stream. Training and sampling logic lives in `grce.py`.

## How the context channel works
1. **Per-position capture.** For every position we collect the inputs to each Transformer block before self-attention/FFN work on them. Those vectors are the only items allowed to leak information across time.
2. **Gradient-limited sampling.** Each captured input passes through a small sampler MLP; the outputs share weights across timesteps but are detached according to `--context-span` (span 0 keeps gradients, span 1 detaches every step, span N>1 detaches every Nth step). The sampled vectors are summed to form a “pre-link” context sketch.
3. **Context propagation.** A context-link MLP transforms the sketch into the recurrent context vector for the next position. This vector is carried forward exactly once per time step; no other hidden state persists.
4. **Bias injection.** At the next position the context vector is fed into block-specific bias generators. Each generator emits a bias that is added only to the newest token row of its block input before attention/FFN. The rest of the sequence stays untouched, so the context acts like an additive, per-layer steering signal.

This mechanism creates an explicit channel for slow-changing control signals without interfering with self-attention capacity. Because the context vector is the sole cross-time carrier, we expect it to split into two behavioral bands:
- **Long-term, semi-stable slots** that act like on/off embeddings describing style, role, or tone and remain active across many positions.
- **Short-term, rapidly changing slots** that behave like token-to-token controllers (grammar states, agreement markers, etc.) and flicker as the model advances.

Tuning `--context-span` lets you decide how much of the vector learns long vs. short horizon behavior: larger spans enforce more stability, while span 0 leaves everything plastic.

## Running it
Use `python grce.py --help` for CLI options. Typical runs specify dataset (`--data`), model size (`--n-layer`, `--n-head`, `--n-embd`, `--n-grce`), and optional knobs such as `--special` (injecting dissonance markers) or `--context-span`.

(pretty much all code in this repo is ai-generated. but of course only under my strong supervision.. ~Claire ;)
