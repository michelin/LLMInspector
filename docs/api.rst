.. _api:

API Documentation
=================

The package is layered: a test case flows from ``dataset`` through ``metrics``
into ``evaluate``, and ``generation`` produces the goldens that seed it.

Top level
---------

.. automodule:: llminspector
    :members:

Test case and dataset
---------------------

.. automodule:: llminspector.test_case.test_case
    :members:

.. automodule:: llminspector.dataset.dataset
    :members:

Models
------

.. automodule:: llminspector.models.base_model
    :members:

.. automodule:: llminspector.models.metered
   :members:

.. automodule:: llminspector.models.azure_openai
    :members:

Metrics
-------

.. automodule:: llminspector.metrics.base_metric
    :members:

.. automodule:: llminspector.metrics.rag
    :members:

.. automodule:: llminspector.metrics.safety
    :members:

.. automodule:: llminspector.metrics.quality
    :members:

.. automodule:: llminspector.metrics.nlp
    :members:

.. automodule:: llminspector.metrics.policy
    :members:

.. automodule:: llminspector.metrics.aggregate
    :members:

Evaluate
--------

.. automodule:: llminspector.evaluate.evaluate
    :members:

.. automodule:: llminspector.evaluate.result
    :members:

Generation
----------

.. automodule:: llminspector.generation.generator
    :members:

.. automodule:: llminspector.generation.source
    :members:

.. automodule:: llminspector.generation.stage
    :members:

.. automodule:: llminspector.generation.config
    :members:

.. automodule:: llminspector.generation.stages
   :members:

.. automodule:: llminspector.generation.context.chunking
   :members:

.. automodule:: llminspector.generation.context.index
   :members:

.. automodule:: llminspector.generation.context.loaders
   :members:

.. automodule:: llminspector.generation.context.selection
   :members:

.. automodule:: llminspector.generation.sources.documents
   :members:

.. automodule:: llminspector.generation.sources.scratch
   :members:

.. automodule:: llminspector.generation.sources.seed
   :members:

.. automodule:: llminspector.generation.adversarial
    :members:

.. automodule:: llminspector.generation.sources.contexts
    :members:

.. automodule:: llminspector.generation.perturbations
    :members:

Reporting and configuration
---------------------------

.. automodule:: llminspector.reporting.exporters
    :members:

.. automodule:: llminspector.config.settings
    :members:
