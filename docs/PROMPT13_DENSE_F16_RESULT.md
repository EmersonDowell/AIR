# Prompt 13 R2 Dense FP16 cuBLAS Result

`dense-f16-cublas` was evaluated as a calibration/Pareto candidate. It did not pass AIR's unchanged `atol=0.001` differential gate and therefore was never competitively benchmarked.

On the 64-token verification history:

- reuse8 max absolute errors were about `2.01e-5` and `1.65e-5`;
- dense FP16 max absolute errors were about `0.01606` and `0.00918`;
- both dense decisions remained finite and preserved top-1.

The result does not establish anything about candidate speed or Tensor Core value. It establishes only that the combined FP16 prepared-weight / FP16 activation / cuBLAS execution tactic does not satisfy the current numerical contract. The tactic is removed from the live vocabulary.
