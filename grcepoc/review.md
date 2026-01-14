# Literature Review: Gradient-limited Recurrent Context Encoding (GRCE)
*(AI-generated placeholder; replace with curated research notes when ready.)*

## 1. Overview of GRCE
The Gradient-limited Recurrent Context Encoding (GRCE) channel augments a decoder-only Transformer with a narrow, additive recurrence. For every position, each block input is LayerNorm’ed and passed through a linear sampler `n_embd → n_grce`; the `n_layer` messages are summed, pushed through a shared bottleneck MLP (`n_grce → 4*n_grce → ReLU → LayerNorm → n_grce`), and optionally truncated according to `--context-span` / `--grce-dropout`. The resulting context vector (the only cross-time state) is then decoded back into per-layer biases via simple linear maps `n_grce → n_embd` that only touch the newest token row. Intuitively, GRCE is “attention rotated over depth”: each layer emits a fixed linear summary, the summaries are mixed once in time, and every layer at the next position injects the shared message as a bias. This yields a bottlenecked recurrent channel that is easy to probe and easy to disable (nogrce mode) while still giving the model a way to pass style/role cues forward.

## 2. Related Architectures
| Work | Mechanism | Key Similarities / Differences |
| --- | --- | --- |
| Transformer-XL (Dai et al., 2019) [1] | Recurrence by caching key/value memories from previous segments. | Shares the idea of forwarding limited information between segments, but GRCE forwards a compressed additive bias rather than raw attention states. Transformer-XL allows gradients through memories; GRCE can explicitly detach via spans. |
| Compressive Transformer (Rae et al., 2020) [2] | Extends memories with compressed summaries of older segments. | Like GRCE, uses learnable compression of past context, but CT compresses entire attention states, whereas GRCE only uses per-layer inputs and injects them as biases. |
| Recurrent Memory Transformer (Bullet, 2020) [3] | Adds dedicated memory tokens that persist across segments. | Both add a recurrent pathway separated from regular tokens. RMT stores state as tokens inside attention; GRCE stores it as an external vector applied as biases. |
| Gated Transformer-XL (Parisotto et al., 2020) [4] | Adds gating and memory to stabilize RL training. | GRCE’s context-span knob resembles GTXL’s gating in spirit (controlling gradient flow), but GTXL still uses KV memories, not additive biases. |
| RWKV (Peng et al., 2023) [5] | Hybrid RNN/Transformer with channel mixing and time-mixed states. | RWKV propagates state through recurrent convolutions; GRCE keeps a single learned vector that modulates per-layer inputs, closer to a learned global bias. |
| Retentive Networks (Sun et al., 2023) [6] | Replace attention with exponential moving averages. | RetNet uses recurrent state updates without attention. GRCE keeps full attention but adds a parallel recurrence for control signals. |
| Biased Transformer Variants (e.g., Adaptive Input Bias) [7] | Inject learned biases based on metadata or prompts. | GRCE can be interpreted as a dynamic bias computed on-the-fly from prior activations, rather than a static embedding. |

## 3. Reflection and Comparison
1. **Information bottleneck.** Unlike memory-based architectures that forward entire activation maps, GRCE enforces a strict `n_grce`-dimensional bottleneck per time step. This makes the recurrent channel interpretable and cheap but may limit long-horizon recall compared with Transformer-XL or RMT.
2. **Additive steering vs. attention reuse.** GRCE injects context as additive biases. This is similar to Feature-wise Linear Modulation (FiLM) but applied within a Transformer stack. In contrast, most recurrent Transformers reuse attention keys/values, effectively giving later tokens direct access to earlier activations. GRCE instead nudges the block input, so the effect is more akin to conditioning than to explicit memory lookup.
3. **Gradient control.** The `--context-span` knob (and the `--grce-dropout` ratio that randomly skips the channel during training) introduce deterministic and stochastic truncation, related to truncated BPTT or stop-gradient heuristics used in RL (e.g., GTXL). This is less common in standard Transformers, where gradients freely traverse cached memories. The ability to dial span/dropout may yield distinct long-term vs. short-term subspaces, aligning with hierarchical RNN intuitions while keeping the training stable when the channel is disabled.
4. **Computation/storage trade-offs.** GRCE adds two shallow MLPs per layer but keeps the forward pass O(L). Memory footprints grow only with `n_grce`, whereas architectures like Compressive Transformer add separate buffers. This makes GRCE attractive for lightweight deployments but may restrict capacity to store fine-grained history.
5. **Interpretability potential.** Because the recurrent vector is the only cross-time state, it may be easier to probe (e.g., PCA, clustering) than KV caches. This mirrors μ-Transformers or state-space models where a single latent can be visualized.

## 4. Opportunities for Future Study
- **Empirical comparison.** Benchmark GRCE against Transformer-XL or RWKV on long-context tasks to measure how much information the additive biases can transport.
- **Hybrid memory.** Combine GRCE with limited KV caching to capture both additive steering and explicit recall.
- **Context span scheduling.** Explore curriculum strategies where span grows during training, similar to curriculum BPTT.
- **Visualization.** Track context activations to confirm the hypothesized long-term vs. short-term slots, perhaps via probing classifiers.

## References
[1] Dai, Z. et al. *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*. ACL 2019.
[2] Rae, J. et al. *Compressive Transformers for Long-Range Sequence Modelling*. ICLR 2020.
[3] Bulatov, Y. et al. *Recurrent Memory Transformer*. 2020. arXiv:2006.11527.
[4] Parisotto, E. et al. *Stabilizing Transformers for Reinforcement Learning*. ICML 2020.
[5] Peng, B. et al. *RWKV: Reinventing RNNs for the Transformer Era*. 2023. arXiv:2305.13048.
[6] Sun, W. et al. *Retentive Network: A Successor to Transformer for Large Language Models*. 2023. arXiv:2307.08621.
[7] Pérez, E. et al. *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.
