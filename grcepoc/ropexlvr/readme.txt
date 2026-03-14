## RoPE-XL

**RoPE-XL** (--use-rope-xl) is a modification of Rotary Positional Embeddings that preserves the standard RoPE behavior for short relative distances but smoothly compresses larger separations so that very long contexts remain distinguishable without phase wraparound.

Let Δ = p₂ − p₁ be the relative position between two tokens and let N be the original RoPE context length parameter. Define S = N/2. RoPE-XL leaves relative offsets unchanged for |Δ| < S, and smoothly compresses larger magnitudes according to

	Δ' = Δ                                     if |Δ| < S
	Δ' = sign(Δ) * (3S − 2S * sqrt(S / |Δ|))   otherwise

This mapping is continuous and differentiable at |Δ| = S. In particular,

	|Δ| = S      →  S
	|Δ| = N      →  ≈ 0.793 N
	|Δ| → ∞      →  1.5 N

Thus RoPE-XL keeps the exact RoPE geometry for short ranges (|Δ| < N/2) while smoothly compressing larger separations into the bounded interval [−1.5 N, 1.5 N]. The smooth transition avoids the slope discontinuity of earlier compression schemes and produces a gradual reduction in effective positional distance as |Δ| grows.


## RoPE-VR (RoPE with Value Rotation)

**RoPE-VR** (RoPE with Value Rotation, --use-rope-vr) extends standard RoPE by also applying the same positional rotation to the value vectors of **half of the K/V heads**. All heads use standard RoPE on queries and keys, preserving the usual relative-position attention behavior. The remaining K/V heads keep unrotated values and therefore behave exactly like standard RoPE attention.

In the heads using value rotation, the value vector carries a positional phase corresponding to its source position. When a token at position p attends to position j, the returned representation contains a component proportional to R(j)v_j. Because the query was rotated by R(p), the resulting representation encodes the relative offset (j − p) together with the content.

This allows **relative position information to be forwarded through the network**. If a later layer forms a query from such a representation at position p', the embedded phase interacts with the new query rotation so that it now represents (j − p'), the relative position of the same token j from the perspective of the new query position.

The non-rotated heads act as standard content channels, while the rotated-value heads transport phase-tagged representations that can carry relative-position pointers across layers. Splitting this behavior at the K/V-head level allows clean specialization and avoids forcing a single value projection to serve both rotated and unrotated roles.

----

## RoPE-XL and RoPE-VR experiments

python grce.py --pt model/wp_en_XS_ROPEXL_GMLP_Q4_ED4Y.pt --small --use-rope-xl --use-gmlp --n-query 4 create
python grce.py --pt model/wp_en_XS_ROPEXL_GMLP_Q4_ED4Y.pt corpus --add wikipedia-en-000{0,1,2,3,4,5,6,7}

python grce.py --pt model/wp_en_XS_ROPEXL_GMLP_Q4_ED4Y.pt --cycles 300 \
  --layout "(128[64D]|64[128D]|32[256D]|16[512D]|8[1024D]|32[64E=64D2Y=64D3Y=64D4Y])" \
  --lr-warmup-steps 100 --lr-cosine-steps 200 --generate-with-decode --align-articles train --json

#--

python grce.py --pt model/wp_en_XS_ROPEVR_GMLP_Q4_ED4Y.pt --small --use-rope-vr --use-gmlp --n-query 4 create
python grce.py --pt model/wp_en_XS_ROPEVR_GMLP_Q4_ED4Y.pt corpus --add wikipedia-en-000{0,1,2,3,4,5,6,7}

python grce.py --pt model/wp_en_XS_ROPEVR_GMLP_Q4_ED4Y.pt --cycles 300 \
  --layout "(128[64D]|64[128D]|32[256D]|16[512D]|8[1024D]|32[64E=64D2Y=64D3Y=64D4Y])" \
  --lr-warmup-steps 100 --lr-cosine-steps 200 --generate-with-decode --align-articles train --json

#--

python grce.py --pt model/wp_en_XS_ROPEVRALL_GMLP_Q4_ED4Y.pt --small --use-rope-vr-all --use-gmlp --n-query 4 create
python grce.py --pt model/wp_en_XS_ROPEVRALL_GMLP_Q4_ED4Y.pt corpus --add wikipedia-en-000{0,1,2,3,4,5,6,7}

python grce.py --pt model/wp_en_XS_ROPEVRALL_GMLP_Q4_ED4Y.pt --cycles 300 \
  --layout "(128[64D]|64[128D]|32[256D]|16[512D]|8[1024D]|32[64E=64D2Y=64D3Y=64D4Y])" \
  --lr-warmup-steps 100 --lr-cosine-steps 200 --generate-with-decode --align-articles train --json

#--

python grce.py --pt model/wp_en_XS_GMLP_Q4_ED4Y.pt --batch-size 32 --layout '1[700D=700D=600D]' eval --rand 10000 > model/eval_rope.txt
python grce.py --pt model/wp_en_XS_ROPEXL_GMLP_Q4_ED4Y.pt --batch-size 32 --layout '1[700D=700D=600D]' eval --rand 10000 > model/eval_ropexl.txt
python grce.py --pt model/wp_en_XS_ROPEVR_GMLP_Q4_ED4Y.pt --batch-size 32 --layout '1[700D=700D=600D]' eval --rand 10000 > model/eval_ropevr.txt
python grce.py --pt model/wp_en_XS_ROPEVRALL_GMLP_Q4_ED4Y.pt --batch-size 32 --layout '1[700D=700D=600D]' eval --rand 10000 > model/eval_ropevrall.txt
