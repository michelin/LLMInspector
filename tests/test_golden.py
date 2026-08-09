"""Schema-seam tests for ``Golden`` — the shape Phase 1 introduced.

Phase 1 gave every golden a generated ``id``, moved identity/provenance fields
(``id``, ``context``, ``metadata``) onto a shared ``_BaseGolden``, and made
``to_test_case`` carry that identity onto ``LLMTestCase.golden_id``. Those three
things are a seam other code is meant to build on, so the tests below pin them
explicitly rather than exercising them incidentally: the exact field placement,
the exact serialized key sets, and the copy semantics of ``metadata``.
"""

import pytest
from pydantic import ValidationError

from llminspector.dataset.golden import Golden, _BaseGolden
from llminspector.test_case import LLMTestCase

# -- identity -----------------------------------------------------------------


def test_id_is_generated_automatically():
    g = Golden(input="q")
    assert isinstance(g.id, str)
    assert g.id != ""


def test_identical_goldens_get_different_ids():
    """Identity is per-object, never derived from content.

    Two goldens may legitimately carry the same input — different evolutions of
    one seed, or the same question exported twice — so hashing the fields would
    collapse rows that must stay distinguishable in a scored table.
    """
    a = Golden(input="q", expected_output="gt", context=["c"])
    b = Golden(input="q", expected_output="gt", context=["c"])
    assert a.id != b.id


def test_explicit_id_is_preserved():
    """A caller-supplied id wins: this is what makes the Excel round trip work."""
    assert Golden(input="q", id="fixed").id == "fixed"


def test_model_copy_preserves_id():
    """Copies are the *same* golden, so they keep its identity.

    The pipeline copies goldens between stages; if a copy minted a fresh id, the
    lineage link from a scored row back to its seed would break silently.
    """
    g = Golden(input="q", context=["c"], metadata={"k": "v"})
    assert g.model_copy().id == g.id
    assert g.model_copy(deep=True).id == g.id


def test_model_dump_key_set():
    """Pin the serialized shape so a new field is a decision, not an accident.

    ``model_dump`` output flows into export paths; silently gaining a key there
    changes what users see in a workbook.
    """
    dumped = Golden(input="q").model_dump()
    assert set(dumped) == {"id", "input", "expected_output", "context", "metadata"}


# -- the _BaseGolden / Golden split -------------------------------------------


def test_golden_subclasses_base_golden():
    assert issubclass(Golden, _BaseGolden)


def test_shared_fields_live_on_base_golden():
    """The shared fields must stay on the base class, not drift down to ``Golden``.

    ``_BaseGolden`` is the multi-turn extension point: a sibling
    ``ConversationalGolden`` is meant to substitute ``input`` /
    ``expected_output`` with ``scenario`` / ``expected_outcome`` while inheriting
    identity, grounding and provenance unchanged. If ``id`` / ``context`` /
    ``metadata`` were redeclared on ``Golden``, that sibling would silently lose
    them and every generic consumer written against ``_BaseGolden`` would stop
    working — with no test failing to say so. Hence this assertion on placement
    rather than on mere presence.
    """
    assert set(_BaseGolden.model_fields) == {"id", "context", "metadata"}
    assert set(Golden.model_fields) - set(_BaseGolden.model_fields) == {
        "input",
        "expected_output",
    }


def test_base_golden_is_constructible_and_turnless():
    """``_BaseGolden`` is a shared base, not an abstract class.

    It carries no turn shape at all, which is exactly why a sibling can supply a
    different one.
    """
    base = _BaseGolden()
    assert base.context is None
    assert base.metadata == {}
    assert isinstance(base.id, str) and base.id != ""
    assert not hasattr(base, "input")


# -- preserved pre-Phase-1 behaviour -------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None])
def test_input_rejects_blank_and_none(bad):
    with pytest.raises(ValidationError):
        Golden(input=bad)


def test_input_required():
    with pytest.raises(ValidationError):
        Golden()  # type: ignore[call-arg]


def test_context_coerces_single_string():
    assert Golden(input="q", context="just one").context == ["just one"]


def test_context_drops_blank_entries():
    g = Golden(input="q", context=["a", "", "   ", "b"])
    assert g.context == ["a", "b"]


def test_context_all_blank_becomes_none():
    assert Golden(input="q", context=["", "   "]).context is None


def test_unknown_kwarg_is_ignored():
    """``extra="ignore"`` — a stray spreadsheet column must not become an attribute.

    Producers route their extra columns through ``metadata`` instead; see
    ``EvaluationDataset.goldens_from_pandas``.
    """
    g = Golden(input="q", not_a_field="x")
    assert not hasattr(g, "not_a_field")
    assert "not_a_field" not in g.model_dump()


def test_metadata_default_is_per_instance():
    """Each golden gets its own dict, not one shared class-level default.

    A shared mutable default is the classic pydantic-adjacent bug: every
    generator writing lineage into ``metadata`` would pollute every other
    golden. ``default_factory`` prevents it — this test is what says we rely on
    that and would catch a switch to ``metadata: Dict = {}``.
    """
    a = Golden(input="q")
    b = Golden(input="q")
    a.metadata["source"] = "a"
    assert b.metadata == {}
    assert a.metadata is not b.metadata


# -- to_test_case --------------------------------------------------------------


def test_to_test_case_maps_core_fields():
    """``context`` is renamed to ``retrieval_context`` — same passages, each side
    of the pipeline naming them for what they are there."""
    g = Golden(input="q", expected_output="gt", context=["c1", "c2"])
    tc = g.to_test_case()
    assert isinstance(tc, LLMTestCase)
    assert tc.input == "q"
    assert tc.expected_output == "gt"
    assert tc.retrieval_context == ["c1", "c2"]


def test_to_test_case_carries_id_onto_golden_id():
    """The only link back from a scored row to the seed that produced it."""
    g = Golden(input="q", id="seed-1")
    assert g.to_test_case().golden_id == "seed-1"


def test_to_test_case_copies_metadata():
    """Promotion copies the lineage, so the test case cannot corrupt its golden.

    Equality plus non-identity is the cheap half of the check; the mutation is
    the guarantee that actually matters, since a shallow ``copy`` reference or a
    later refactor to ``metadata=self.metadata`` would pass identity-free
    assertions but fail here.
    """
    g = Golden(input="q", metadata={"lineage": "evolved", "score": 0.9})
    tc = g.to_test_case()
    assert tc.metadata == g.metadata
    assert tc.metadata is not g.metadata

    tc.metadata["score"] = 0.0
    tc.metadata["added_later"] = True
    assert g.metadata == {"lineage": "evolved", "score": 0.9}


def test_to_test_case_copies_nested_metadata():
    """The copy is deep, because metadata is not flat.

    The generation stages append one record per stage to a
    ``metadata["lineage"]`` list. Under a shallow ``dict(...)`` copy that list is
    the *same object* on the golden and on every test case promoted from it, so
    appending through one is visible through all of them — the exact corruption
    the copy exists to prevent, just one level down. This test fails on a
    shallow copy and passes on ``deepcopy``.
    """
    g = Golden(input="q", metadata={"lineage": [{"stage": "generate"}]})
    tc = g.to_test_case()

    tc.metadata["lineage"].append({"stage": "evolve"})
    tc.metadata["lineage"][0]["stage"] = "tampered"

    assert g.metadata == {"lineage": [{"stage": "generate"}]}


def test_to_test_case_actual_output():
    g = Golden(input="q")
    assert g.to_test_case().actual_output is None
    assert g.to_test_case(actual_output="a").actual_output == "a"


def test_to_test_case_policy():
    """Goldens have no policy field of their own.

    Policy text is supplied at promotion time because a policy belongs to the
    evaluation being run, not to the seed — the same golden can be scored
    against several policies.
    """
    g = Golden(input="q")
    assert "policy" not in Golden.model_fields
    assert g.to_test_case().policy is None
    assert g.to_test_case(policy="be nice").policy == "be nice"


def test_to_test_case_without_context():
    assert Golden(input="q").to_test_case().retrieval_context is None
