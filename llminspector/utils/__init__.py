"""Shared utilities that sit at the bottom of the layering.

Nothing here imports from another ``llminspector`` subpackage, so every layer
above may reach in freely:

* :mod:`~llminspector.utils.json_utils` — parsing an LLM's JSON reply
* :mod:`~llminspector.utils.optional` — turning a missing optional dependency
  into install instructions
* :mod:`~llminspector.utils.prompting` — f-string prompt rendering
* :mod:`~llminspector.utils.concurrency` — bounded, order-preserving async map

The modules are imported by path rather than re-exported here, so importing the
package stays free and every helper keeps exactly one import path.
"""
