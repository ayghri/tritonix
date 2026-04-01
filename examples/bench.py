from dense_block_sparse_mma import BsrMatrix  # (use the canvas file)
import torch

M, K, N = 1024, 2048, 1536
BK, BN = 32, 32
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

# make a 20% dense block mask
mask = torch.rand(K // BK, N // BN, device="cuda") < 0.20
B_bsr = BsrMatrix.from_dense(B, BK, BN, mask=mask)

# forward
C = B_bsr.mm(A)  # C = A @ B (block-sparse)
# backward works too
loss = C.float().pow(2).mean()
loss.backward()
