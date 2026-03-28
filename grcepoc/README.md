# GPT with Gradient-limited Recurrent Context Encoding and Extended Context (GPT+GRCE+XCTX)

**This is a test-bed for experimenting with different tweaks to transformer architectures. Nothing here is stable.**

See [`notes.txt`](notes.txt) for the full specification and [`grce.py`](grce.py) for the (mostly AI-generated) implementation.

This repo extends a tiny picoGPT-style language model with two recurrent context channels: a lightweight low-bandwidth “role/focus” state alongside the usual token stream (GRCE) plus a high-bandwidth “short-term memory” channel for pushing information forward in time (XCTX) in parallel to multi-head attention, which still looks backward. “GPT+GRCE+XCTX” is pronounced “GPT with grace and extended context”.

The additions provide a few concrete benefits:

- The low-bandwidth GRCE path gives the model an explicit “what am I doing here?” signal that improves stability, debuggability, and interpretability. You can inspect the role/focus state to understand why a column behaves the way it does.
- The high-bandwidth XCTX path acts as a recurrent twin to multi-head attention. Attention is great when the future knows what it needs from the past; XCTX lets the past proactively tell the future “remember this” without guessing which head will retrieve it. We even train it with dropout-like noise so the stack learns to rely on that forward channel.

We also implement “looping” (https://arxiv.org/abs/2502.17416) along both axes: stacking along the vertical layer/height axis ("y-looping") *and* stacking along the horizontal time axis ("x-looping"). With standard transformers the horizontal option is awkward, but the recurrent infrastructure makes it easy. Horizontal looping provides two major advantages: (1) each loop step can easily attend to the previous step’s KV cache, and (2) the GRCE/XCTX paths supply a short, cheap, high-bandwidth connection between loop steps so information doesn’t have to slog through the entire stack just to reach the next loop.

SANE (Self-And-Next Encoder) spans extend this idea by creating multi-column decoder grids that reuse most of their activations from one pass to the next. Each pass performs its Y loop iterations and then, immediately before we inject the newly promoted column’s true token plus [|predict-next|], the shared state is run through a single RMSNorm. That keeps the intra-pass trajectories intact while still normalizing the data that enters the next pass.

### Getting Started

Create and activate venv:
```
python3 -m venv .venv
source activate
pip install -r requirements.txt
```

Setup the tiny "simplestwiki" corpus:
```
cd data
bash simplestwiki-setup.sh
cd ..
```

Train a tiny (8k) model for testing the flow:
```
python grce.py --tiny create
python grce.py --tiny corpus --add simplestwiki
python grce.py --tiny --cycles 10 --steps 100 --eval-interval 10 train
```

Plot test losses recorded during training:
```
python grce.py --tiny json
python plot.py --json model/default_model_v600_n10_w8_d3_h2_g4_x9.json --plot-steps test_loss
```

### Training a more serious model

Downloading, splitting, and tokenizing the wikipedia corpus:
```
cd data
bash wikipedia-setup.sh
cd ..
```

Training a base encode/decode transformer model (with optional "y-looping"):
```
python grce.py --pt model/wp_en_E_D4Y.pt create
python grce.py --pt model/wp_en_E_D4Y.pt corpus --add wikipedia-en-000{0,1,2,3,4,5,6,7}
python grce.py --pt model/wp_en_E_D4Y.pt --cycles 5000 --generate-with-decode \
  --lr-warmup-steps 200 --lr-cosine-steps 500 --block-size 128 --batch-size 32 \
  --layout '*[8E=8D4(Y|y)>head=*D4Y=1D4Yb>>b=1D4YB>>B=*D4Y=8D(1|2|3)(Y|y)>>D123Y=8D4Y>tail]' train --json
```

Plot various losses recorded during base encode/decode transformer training:
```
python plot.py --json model/wp_en_E_D4Y.json --median 10 --plot-time test_loss_head test_loss_tail test_loss
```

Additional training of the recurrent network (with optional "x-looping"):
```
cp model/wp_en_E_D4Y.pt model/wp_en_E_D4Y_f_t4x.pt
python grce.py --pt model/wp_en_E_D4Y_f_t4x.pt reset
python grce.py --pt model/wp_en_E_D4Y_f_t4x.pt --cycles 500 --lr-warmup-steps 200 --lr-cosine-steps 500 \
  --block-size 64 --batch-size 64 --layout '*[8E=8D(1|2|3|4)Y=*8f=*8t(2x|3x|4x)]' train --json
```

Plot stacked test losses recorded during training:
```
python plot.py --json model/wp_en_E_D4Y.json --json model/wp_en_E_D4Y_f_t4x.json --stack-sources --plot-time
```
