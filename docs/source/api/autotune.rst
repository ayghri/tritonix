Autotuning
==========

Decorator
---------

.. autofunction:: tritonix.autotune.tunable

TunableKernel
-------------

.. autoclass:: tritonix.autotune.TunableKernel
   :members: tune, configs

Config Space
------------

.. autoclass:: tritonix.utils.spaces.PowerOfTwo

.. autoclass:: tritonix.utils.spaces.Range

.. autoclass:: tritonix.utils.spaces.Choice

.. autoclass:: tritonix.utils.spaces.ConfigSpace

Pruners
-------

.. autoclass:: tritonix.utils.pruners.MonotonicCascadeTrie
   :members: prune, is_pruned, generate_all_unpruned_midpoint

.. autoclass:: tritonix.utils.pruners.CoordinateMonotonicFunction
   :members: record, is_pruned
