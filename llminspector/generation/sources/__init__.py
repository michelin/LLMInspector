"""Golden sources — the front of the generation pipeline.

One module per way of seeding a run. Each implements
:class:`~llminspector.generation.source.GoldenSource`, so the stage chain that
follows is identical no matter where the goldens came from.
"""

from .contexts import ContextSource
from .curated_bank import CuratedBankSource

__all__ = ["ContextSource", "CuratedBankSource"]
