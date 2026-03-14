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
