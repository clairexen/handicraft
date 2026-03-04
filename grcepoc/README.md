# GPT with Gradient-limited Recurrent Context Encoding and Extended Context (GPT+GRCE+XCTX)

See `notes.txt` for the full specification and `grce.py` for the (mostly AI-generated) implementation.

This repo extends a tiny picoGPT-style language model with two recurrent context channels: a lightweight low-bandwidth “role/focus” state alongside the usual token stream (GRCE) plus a high-bandwidth “short-term memory” channel for pushing information forward in time (XCTX) in parallel to multi-head attention, which still looks backward. “GPT+GRCE+XCTX” is pronounced “GPT with grace and extended context”.

The additions provide a few concrete benefits:

- The low-bandwidth GRCE path gives the model an explicit “what am I doing here?” signal that improves stability, debuggability, and interpretability. You can inspect the role/focus state to understand why a column behaves the way it does.
- The high-bandwidth XCTX path acts as a recurrent twin to multi-head attention. Attention is great when the future knows what it needs from the past; XCTX lets the past proactively tell the future “remember this” without guessing which head will retrieve it. We even train it with dropout-like noise so the stack learns to rely on that forward channel.

We also implement “looping” (https://arxiv.org/abs/2502.17416) along both axes: stacking along the layer/height axis *and* stacking along the time axis. With standard transformers the horizontal option is awkward, but the recurrent infrastructure makes it easy. Horizontal looping provides two major advantages: (1) each loop step can attend to the previous step’s KV cache, and (2) the GRCE/XCTX paths supply a short, cheap, high-bandwidth connection between steps so information doesn’t have to slog through the entire stack just to reach the next loop.
