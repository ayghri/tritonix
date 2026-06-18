Ops
===

Matmul
------

.. autofunction:: tritonix.ops.matmul.triton_matmul

.. autodata:: tritonix.ops.matmul.matmul
   :annotation: = DynamicDispatcher({"triton": triton_matmul, "pytorch": torch.matmul})

.. autodata:: tritonix.ops.matmul.matmul_kernel
   :annotation: = TunableKernel wrapping matmul_kernel

.. autofunction:: tritonix.ops.matmul.matmul_splitk

Conv2D
------

.. autofunction:: tritonix.ops.conv2d.conv2d_forward

SwiGLU
------

.. automodule:: tritonix.ops.swiglu
   :members:
