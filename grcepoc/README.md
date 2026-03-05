# GPT with Gradient-limited Recurrent Context Encoding and Extended Context (GPT+GRCE+XCTX)

See [`notes.txt`](notes.txt) for the full specification and [`grce.py`](grce.py) for the (mostly AI-generated) implementation.

This repo extends a tiny picoGPT-style language model with two recurrent context channels: a lightweight low-bandwidth “role/focus” state alongside the usual token stream (GRCE) plus a high-bandwidth “short-term memory” channel for pushing information forward in time (XCTX) in parallel to multi-head attention, which still looks backward. “GPT+GRCE+XCTX” is pronounced “GPT with grace and extended context”.

The additions provide a few concrete benefits:

- The low-bandwidth GRCE path gives the model an explicit “what am I doing here?” signal that improves stability, debuggability, and interpretability. You can inspect the role/focus state to understand why a column behaves the way it does.
- The high-bandwidth XCTX path acts as a recurrent twin to multi-head attention. Attention is great when the future knows what it needs from the past; XCTX lets the past proactively tell the future “remember this” without guessing which head will retrieve it. We even train it with dropout-like noise so the stack learns to rely on that forward channel.

We also implement “looping” (https://arxiv.org/abs/2502.17416) along both axes: stacking along the layer/height axis *and* stacking along the time axis. With standard transformers the horizontal option is awkward, but the recurrent infrastructure makes it easy. Horizontal looping provides two major advantages: (1) each loop step can attend to the previous step’s KV cache, and (2) the GRCE/XCTX paths supply a short, cheap, high-bandwidth connection between steps so information doesn’t have to slog through the entire stack just to reach the next loop.

### Getting Started

Setting up and activating venv:
```
python3 -m venv .venv
source activate
pip install -r requirements.txt
```

Setup the tiny "simplestwiki" corpus:
```
cd data
bash wikipedia-setup.sh
cd ..
```

Training a tiny model for testing the flow:
```
python grce.py --tiny create
python grce.py --tiny corpus --add simplestwiki
python grce.py --tiny --steps 10 train
```

### Training a more serious model

Downloading, splitting, and tokenizing wikipedia corpus:
```
cd data
bash wikipedia-setup.sh
cd ..
```

Training base transformer model (with optional "y"-looping):
```
python grce.py --pt model/wp_en_E_D4Y.pt create
python grce.py --pt model/wp_en_E_D4Y.pt corpus --add wikipedia-en-000{0,1,2,3,4,5,6,7}
python grce.py --pt model/wp_en_E_D4Y.pt --cycles 3600 --lr-warmup-steps 200 --lr-cosine-steps 500 \
  --block-size 100 --batch-size 50 --layout '*[8E=8D4Y>head=*D4Y=*D4Y=8D(1|2|3)Y>>D123Y=8D4Y>tail]' train --json
```

Training recurrent backbone network (with optional "x"-looping):
```
cp model/wp_en_E_D4Y.pt model/wp_en_E_D4Y_f_t4x.pt
python grce.py --pt model/wp_en_E_D4Y_f_t4x.pt reset
python grce.py --pt model/wp_en_E_D4Y_f_t4x.pt --cycles 500 --lr-warmup-steps 200 --lr-cosine-steps 500 \
  --block-size 64 --batch-size 64 --layout '*[8E=8D(1|2|3|4)Y=*8f=*8t(2x|3x|4x)]' train --json
```

Plot test losses recorded during training:
```
python plot.py --json model/wp_en_E_D4Y.json --json model/wp_en_E_D4Y_f_t4x.json --stack-sources --plot-time test_loss
```
