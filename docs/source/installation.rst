Installation
============

Requirements
------------

* Python ≥ 3.11
* PyTorch ≥ 2.7
* Triton ≥ 3.4
* CUDA-capable GPU

From PyPI
---------

.. code-block:: bash

   pip install tritonix

From source
-----------

.. code-block:: bash

   git clone https://github.com/ayghri/tritonix.git
   cd tritonix
   pip install -e .

Optional: Bayesian tuning
--------------------------

The Bayesian optimization backend requires `ax-platform <https://ax.dev>`_:

.. code-block:: bash

   pip install tritonix[bayesian]
   # or
   pip install ax-platform
