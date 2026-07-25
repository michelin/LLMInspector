.. _api:

API Documentation
=================

The package is layered: a test case flows from ``dataset`` through ``metrics``
into ``evaluate``, and ``synthesizer`` produces the goldens that seed it.

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

Synthesizer
-----------

.. automodule:: llminspector.synthesizer.base
    :members:

.. automodule:: llminspector.synthesizer.alignment
    :members:

.. automodule:: llminspector.synthesizer.adversarial
    :members:

.. automodule:: llminspector.synthesizer.rag
    :members:

.. automodule:: llminspector.synthesizer.perturbations
    :members:

.. automodule:: llminspector.synthesizer.engines.base
    :members:

Reporting and configuration
---------------------------

.. automodule:: llminspector.reporting.exporters
    :members:

.. automodule:: llminspector.config.settings
    :members:
