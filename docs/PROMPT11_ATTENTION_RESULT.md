# Prompt 11/16 Result — Paged Attention Engine

Prompt 11 qualified `online-softmax` as the preferred CUDA prefill-attention tactic on the tested RTX 3080 Laptop / Qwen2.5-0.5B Q4_K_M workload, with prefill linear fixed at `batch-reuse8`.

Five randomized paired rounds showed:

- p1042 prefill: online-softmax improved the paired prefill speed ratio to approximately **1.142x**, 95% CI **[1.114, 1.170]**.
- p262 prefill: online-softmax improved the paired prefill speed ratio to approximately **1.049x**, 95% CI **[1.034, 1.065]**.
- p1042 total-speed ratio was approximately **1.137x**, 95% CI **[1.108, 1.166]**.
- p262 total-speed ratio was approximately **1.043x**, 95% CI **[1.030, 1.056]**.

Both attention tactics passed the frozen `atol=0.001` reference/CUDA differential gate. The online recurrence was numerically closer to the reference oracle in the tested trace than the baseline tactic. Device/KV high-water values were unchanged.

Nsight Systems verified the intended mechanism. On p1042, the baseline paged-attention kernel consumed roughly **1.36 s** of GPU time while online-softmax consumed roughly **1.05 s**, a reduction of about **23%**. Kernel-launch count was effectively unchanged, so the gain came from reducing attention compute rather than launch consolidation.

After the change, the long-context profile is approximately co-dominated by:

- quantized linear execution: about **51%** of GPU kernel time;
- online paged attention: about **47%**.

Prompt 11 therefore closes. Continuing attention optimization immediately would no longer follow a singular dominant bottleneck. The next independently demonstrated defect is concurrency: the scheduler can admit concurrent sequences, but CUDA decode has not yet converted them into physical multi-sequence execution.
