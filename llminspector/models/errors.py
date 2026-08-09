"""Provider-layer exceptions.

Small on purpose: the only failure the model layer raises on its own behalf is
a structured response it could not turn into the requested schema. Rate limits
are retried (``retry.py``) and everything else is the provider's own exception,
passed through untouched.
"""

from __future__ import annotations

__all__ = ["StructuredOutputError"]


class StructuredOutputError(ValueError):
    """A model's response could not be parsed into the requested schema.

    Raised only after the reask has also failed, so it means the model was asked
    twice and produced unusable output twice — not a transient blip.

    Subclasses ``ValueError`` because that is what the parse and validation
    failures underneath it raise (``json.JSONDecodeError`` is a ``ValueError``,
    and so is pydantic's ``ValidationError``). Callers already writing
    ``except ValueError`` around a parse keep working.

    Attributes
    ----------
    schema_name:
        Name of the pydantic model that was requested.
    response:
        The raw text of the final (post-reask) attempt, for debugging. The
        exception chains from the underlying parse error via ``__cause__``.
    """

    def __init__(self, schema_name: str, response: str, cause: BaseException) -> None:
        self.schema_name = schema_name
        self.response = response
        super().__init__(
            f"Could not parse a {schema_name} from the model's response after "
            f"one reask ({type(cause).__name__}: {cause}). "
            f"Final response was: {response!r}"
        )
