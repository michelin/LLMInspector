"""Phase 3 — adversarial generation from a curated attack bank.

Two things are pinned here, and they pull in opposite directions.

The first is the **melt fix**. ``CuratedBankSource.to_dataframe`` used to route
the filtered bank through ``df.melt(id_vars=["Capability", "Sub Capability",
"Prompt"])`` and rename the melted ``value`` column to ``Char Len``, on the
strength of a docstring claiming the bank carried per-attack *variant* columns.
It is now a direct ``.reindex(columns=[...])``. The melt was a no-op on the only
schema that ships, and wrong on both of the schemas it claimed to serve, so this
module tests all three shapes — see the three tests under "the melt fix".

The second is that **nothing else may move**. Adversarial output for the real
bank is the phase's exit criterion, so the legacy melt is reproduced verbatim
below as ``_legacy_to_dataframe`` and the new code is asserted equal to it over
every filter the real fixture supports. When those two ever disagree, the
question to ask is which one changed.

Everything here is pure pandas — no model, no network. ``show_progress=False``
throughout so ``a_map`` never reaches for tqdm.
"""

import asyncio

import pandas as pd
import pytest

from llminspector.dataset import goldens_to_dataframe
from llminspector.generation import (
    AdversarialGenerator,
    CuratedBankSource,
    GenerationConfig,
    GenerationResult,
    Generator,
)

SAMPLE_ADVERSARIAL = "tests/test_sample/test_adversarialdata.xlsx"

#: The bank's own schema. Named here rather than inlined because both the
#: legacy reproduction and the assertions below have to agree on it exactly.
_ID_VARS = ["Capability", "Sub Capability", "Prompt"]
_BANK_COLUMNS = [*_ID_VARS, "Char Len"]


#: A source needs a config to produce, and none of these tests want a progress
#: bar or a model. Built per call — GenerationConfig is mutable.
def _config() -> GenerationConfig:
    return GenerationConfig(show_progress=False)


def _produce(source: CuratedBankSource):
    """The source's goldens, driven through the *async* entry point.

    ``SyncGoldenSource`` exists to let a purely local source implement
    ``produce`` and inherit ``a_produce``. Calling the async one here means the
    inherited wrapper is exercised by every test in this module rather than by
    one token test that could rot unnoticed.
    """
    return asyncio.run(source.a_produce(_config()))


def _legacy_to_dataframe(filtered: pd.DataFrame) -> pd.DataFrame:
    """The pre-fix ``to_dataframe`` body, reproduced verbatim.

    Kept as executable code rather than prose so "the real bank's output did not
    change" is a comparison against the old implementation instead of against a
    number someone copied down once. This is the only place the melt still
    exists.
    """
    df_melt = filtered.melt(id_vars=_ID_VARS)
    transformed = df_melt.filter([*_ID_VARS, "value"])
    return transformed.rename(columns={"value": "Char Len"})


def _bank(**overrides) -> pd.DataFrame:
    """A three-row synthetic bank in the real schema."""
    columns = {
        "Capability": ["A", "A", "B"],
        "Sub Capability": ["x", "y", "z"],
        "Prompt": ["p1", "p2", "p3"],
        "Char Len": [10, 20, 30],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


@pytest.fixture(scope="module")
def real_bank() -> pd.DataFrame:
    """The shipped 3574-row attack bank, read once for the whole module."""
    return pd.read_excel(SAMPLE_ADVERSARIAL)


# --------------------------------------------------------------------------- #
# the melt fix — shape 1: the real bank, which must not move at all
# --------------------------------------------------------------------------- #


def test_real_bank_output_is_identical_to_the_legacy_melt(real_bank):
    """The exit criterion: byte-identical output for the shipped bank.

    The real schema has exactly one non-id column (``Char Len``), so melting it
    and re-selecting it produced the right answer for the wrong reason. Every
    filter the fixture supports — each capability and each sub-capability — is
    compared against the reproduced legacy implementation, values *and* dtypes.

    Row order is compared after ``reset_index`` because that is the one thing
    that legitimately differs: ``melt`` builds a new frame with a fresh
    RangeIndex, while ``reindex(columns=...)`` keeps the bank's original index
    labels. Nothing downstream reads the index — ``produce`` iterates rows and
    ``goldens_to_dataframe`` assigns fresh ids — so the values are the contract.
    """
    filters = [{"capability": cap} for cap in real_bank["Capability"].unique()]
    filters += [{"subcapability": sub} for sub in real_bank["Sub Capability"].unique()]

    for kwargs in filters:
        source = CuratedBankSource(real_bank, **kwargs)
        new = source.to_dataframe().reset_index(drop=True)
        legacy = _legacy_to_dataframe(source._filtered()).reset_index(drop=True)
        assert new.equals(legacy), f"output moved for {kwargs}"
        # .equals() is dtype-strict, but assert it separately so a failure says
        # "Char Len became object" rather than just "frames differ".
        assert list(new.dtypes) == list(legacy.dtypes)


def test_real_bank_row_count_and_columns(real_bank):
    """One row per matching bank row, in the bank's own four columns."""
    capability = real_bank["Capability"].iloc[0]
    source = CuratedBankSource(real_bank, capability=capability)
    matching = real_bank[real_bank["Capability"] == capability]

    frame = source.to_dataframe()
    assert list(frame.columns) == _BANK_COLUMNS
    assert len(frame) == len(matching)


def test_real_bank_golden_inputs_are_the_filtered_prompts_verbatim(real_bank):
    """The goldens are the bank's prompts, in order, unmodified.

    ``AdversarialGenerator`` has no stages, so this is the whole of adversarial
    generation: a curated attack must reach the model exactly as it was written
    down, or it is no longer the attack that was curated.
    """
    capability = real_bank["Capability"].iloc[0]
    matching = real_bank[real_bank["Capability"] == capability]
    expected = [str(p) for p in matching["Prompt"] if str(p).strip() != ""]

    goldens = _produce(CuratedBankSource(real_bank, capability=capability))
    assert [g.input for g in goldens] == expected


def test_the_real_bank_has_exactly_one_blank_prompt(real_bank):
    """Guards the fixture itself, which is what makes the skip test meaningful.

    3574 rows in, 3573 goldens out. If someone cleans the blank row out of the
    workbook, ``test_blank_prompts_are_skipped`` below silently stops covering
    the real data and this test is the one that says so.
    """
    assert len(real_bank) == 3574
    blank = real_bank["Prompt"].apply(lambda p: str(p).strip() == "")
    assert blank.sum() == 1

    source = CuratedBankSource(real_bank, sample_size=len(real_bank))
    assert len(_produce(source)) == 3573


# --------------------------------------------------------------------------- #
# the melt fix — shape 2: extra columns, which used to duplicate goldens
# --------------------------------------------------------------------------- #


def test_extra_columns_no_longer_fan_out_into_duplicate_goldens():
    """One golden per bank row, whatever else the bank carries.

    This is the schema the melt claimed to serve, and the one it served worst.
    With ``Char Len`` plus two extra columns, ``melt`` emitted **three** rows per
    bank row — three goldens sharing an identical ``Prompt`` — and stuffed each
    extra column's value into a field labelled ``Char Len`` regardless of what it
    actually held, so a golden could report ``Char Len="variant text"``.

    The extra columns are now ignored: they are not part of the bank schema, and
    a column nobody declared should not silently become a metadata field with
    somebody else's name on it.
    """
    # One capability across all three rows, and filter on it: the *unfiltered*
    # path goes through `DataFrame.sample`, which returns rows in random order,
    # so an ordered assertion needs the filtered branch.
    bank = _bank(
        Capability=["A", "A", "A"],
        Variant1=["v1a", "v1b", "v1c"],
        Variant2=["v2a", "v2b", "v2c"],
    )

    # What the old code did, kept in view: 3 rows x 3 melted columns.
    legacy = _legacy_to_dataframe(bank)
    assert len(legacy) == 9
    assert list(legacy["Prompt"]) == ["p1", "p2", "p3"] * 3
    assert "v1a" in list(legacy["Char Len"])  # a string, under a length column

    source = CuratedBankSource(bank, capability="A")
    frame = source.to_dataframe()
    assert list(frame.columns) == _BANK_COLUMNS
    assert len(frame) == 3

    goldens = _produce(source)
    assert [g.input for g in goldens] == ["p1", "p2", "p3"]
    assert [g.metadata["Char Len"] for g in goldens] == [10, 20, 30]
    # No metadata field named after an extra column, and none holding its value.
    assert set(goldens[0].metadata) == set(CuratedBankSource.metadata_keys)
    assert "v1a" not in goldens[0].metadata.values()


# --------------------------------------------------------------------------- #
# the melt fix — shape 3: no Char Len, which used to produce nothing at all
# --------------------------------------------------------------------------- #


def test_bank_without_char_len_still_produces_one_golden_per_row():
    """A bank with no ``Char Len`` yields goldens with ``Char Len`` unset.

    The worst of the three failure modes, because it was silent. With no
    non-id columns to melt, ``melt`` returned an **empty** frame, so a bank of
    perfectly good attack prompts generated *zero* goldens and raised nothing.

    ``Char Len`` is now reindexed rather than required, so a missing column comes
    back as ``NaN`` per row instead of a ``KeyError`` or an empty result. The
    prompts are what matter; the length annotation is a convenience.
    """
    # Filtered rather than sampled, for the row-order reason noted above.
    bank = _bank(Capability=["A", "A", "A"]).drop(columns=["Char Len"])

    # What the old code did: nothing, quietly.
    assert len(_legacy_to_dataframe(bank)) == 0

    source = CuratedBankSource(bank, capability="A")
    frame = source.to_dataframe()
    assert list(frame.columns) == _BANK_COLUMNS
    assert frame["Char Len"].isna().all()

    goldens = _produce(source)
    assert [g.input for g in goldens] == ["p1", "p2", "p3"]
    # Declared but unavailable: the key is present, the value is missing. The
    # declaration is about the output *shape*, which does not depend on the
    # input bank happening to carry the column.
    assert all(pd.isna(g.metadata["Char Len"]) for g in goldens)
    assert set(goldens[0].metadata) == set(CuratedBankSource.metadata_keys)


# --------------------------------------------------------------------------- #
# filtering — behaviour that predates the fix and must survive it
# --------------------------------------------------------------------------- #


def test_capability_filter():
    goldens = _produce(CuratedBankSource(_bank(), capability="A"))
    assert [g.input for g in goldens] == ["p1", "p2"]
    assert goldens[0].metadata["Capability"] == "A"
    assert goldens[0].metadata["Char Len"] == 10


def test_subcapability_filter():
    goldens = _produce(CuratedBankSource(_bank(), subcapability="y"))
    assert [g.input for g in goldens] == ["p2"]
    assert goldens[0].metadata["Sub Capability"] == "y"


def test_both_filters_are_combined_with_and():
    bank = _bank(Capability=["A", "A", "B"], **{"Sub Capability": ["x", "y", "x"]})
    source = CuratedBankSource(bank, capability="A", subcapability="x")
    assert [g.input for g in _produce(source)] == ["p1"]


@pytest.mark.parametrize(
    "capability, subcapability, expected",
    [
        ("a", None, ["p1", "p2"]),
        ("A", None, ["p1", "p2"]),
        (None, "Y", ["p2"]),
        ("A", "Y", ["p2"]),
    ],
)
def test_filters_are_case_insensitive(capability, subcapability, expected):
    """Both sides are lowercased, so bank casing and caller casing are free.

    Attack banks are hand-maintained spreadsheets; requiring a caller to match
    ``Sub Capability`` capitalisation exactly would make the filter a guessing
    game.
    """
    source = CuratedBankSource(
        _bank(), capability=capability, subcapability=subcapability
    )
    assert [g.input for g in _produce(source)] == expected


@pytest.mark.parametrize("spelling", ["all", "ALL", "All"])
def test_all_means_no_filter(spelling):
    """``"all"`` is the spreadsheet spelling of ``None``, in any case.

    Both attributes normalise to ``None`` at construction, so the filter branch
    never sees the string — which is what makes an unfiltered run take the
    sampling path rather than matching a literal capability named "all".
    """
    source = CuratedBankSource(
        _bank(), capability=spelling, subcapability=spelling, sample_size=3
    )
    assert source.capability is None
    assert source.subcapability is None
    assert sorted(g.input for g in _produce(source)) == ["p1", "p2", "p3"]


def test_sample_size_applies_only_when_unfiltered():
    """No capability and no sub-capability means "sample the bank"."""
    bank = _bank()
    goldens = _produce(CuratedBankSource(bank, sample_size=2))
    assert len(goldens) == 2
    # Sampling is uniform and unseeded, so the identity of the rows is not
    # assertable — only that they came from the bank.
    assert set(g.input for g in goldens) <= {"p1", "p2", "p3"}

    # A filter short-circuits the sample: 2 rows match, sample_size is ignored.
    filtered = _produce(CuratedBankSource(bank, capability="A", sample_size=1))
    assert len(filtered) == 2


def test_blank_prompts_are_skipped():
    """Empty, whitespace-only and missing prompts never become goldens.

    A blank row in a hand-maintained spreadsheet is an editing artefact, not an
    attack. Dropping it at the source keeps every downstream stage from having
    to decide what an empty attack means.
    """
    bank = pd.DataFrame(
        {
            "Capability": ["A"] * 4,
            "Sub Capability": ["x"] * 4,
            "Prompt": ["keep", "", "   ", None],
            "Char Len": [4, 0, 3, 0],
        }
    )
    goldens = _produce(CuratedBankSource(bank, capability="A"))
    assert [g.input for g in goldens] == ["keep"]


# --------------------------------------------------------------------------- #
# metadata_keys — the promise that the output shape is knowable up front
# --------------------------------------------------------------------------- #


def test_source_declares_its_metadata_keys():
    assert isinstance(CuratedBankSource.metadata_keys, tuple)
    assert CuratedBankSource.metadata_keys == (
        "Capability",
        "Sub Capability",
        "Char Len",
    )


def test_declared_keys_match_what_actually_lands_in_metadata(real_bank):
    """The declaration is only worth having if it stays true.

    Its whole purpose is to answer "what columns will this run emit?" without
    running the pipeline — which, for an LLM-backed source, means without paying
    for it. A declaration that has drifted from the emitted keys is worse than
    none, because callers build their schema from it.
    """
    generator = AdversarialGenerator.from_dataframe(
        real_bank,
        capability=real_bank["Capability"].iloc[0],
        config=_config(),
    )
    result = generator.generate()
    assert set(result.goldens[0].metadata) == set(generator.metadata_keys)


def test_output_columns_are_knowable_before_generating(real_bank):
    """The exported column set is derivable from ``metadata_keys`` alone."""
    generator = AdversarialGenerator.from_dataframe(
        real_bank, capability=real_bank["Capability"].iloc[0], config=_config()
    )
    expected = ["id", "input", "expected_output", "context", *generator.metadata_keys]

    generator.generate()
    assert list(generator.to_pandas().columns) == expected
    # goldens_to_dataframe moved to llminspector.dataset in this phase, and its
    # output now leads with `id`. Asserting through both entry points pins that
    # the generator's export is that function and not a second implementation.
    assert list(goldens_to_dataframe(generator._result.goldens).columns) == expected


# --------------------------------------------------------------------------- #
# AdversarialGenerator — a preset, and nothing more
# --------------------------------------------------------------------------- #


def test_generator_has_no_stages_so_goldens_pass_through_verbatim():
    """No filtering, no evolution, no restyling: the bank prompt *is* the golden."""
    generator = AdversarialGenerator.from_dataframe(_bank(), capability="A")
    assert generator.stages == []

    result = generator.generate()
    assert [g.input for g in result.goldens] == ["p1", "p2"]
    # Verbatim also means no lineage entry, because nothing happened to it.
    assert "lineage" not in result.goldens[0].metadata


def test_generate_returns_a_generation_result_not_a_dataset():
    """``synth.generate()`` returned an ``EvaluationDataset``; this returns a result.

    The difference is the point: an ``EvaluationDataset`` can only say what
    survived, whereas a ``GenerationResult`` also carries what was dropped and
    why. With no stages there is nothing to drop, so both lists are empty — the
    shape is what is being pinned.
    """
    result = AdversarialGenerator.from_dataframe(_bank(), capability="A").generate()

    assert isinstance(result, GenerationResult)
    assert len(result) == len(result.goldens) == 2
    assert result.errors == []
    assert result.rejected == []
    assert result.error_summary() == ""
    assert list(result.to_pandas()["input"]) == ["p1", "p2"]


def test_async_and_sync_entry_points_agree():
    """``generate()`` is an ``asyncio.run`` wrapper; there is no second pipeline."""
    generator = AdversarialGenerator.from_dataframe(_bank(), capability="A")
    from_async = asyncio.run(generator.a_generate())
    from_sync = AdversarialGenerator.from_dataframe(_bank(), capability="A").generate()
    assert [g.input for g in from_async.goldens] == [g.input for g in from_sync.goldens]


def test_to_pandas_raises_before_anything_has_been_generated():
    """Exporting reads a result; it never starts a run.

    ``BaseSynthesizer.to_pandas()`` called ``generate()`` when it had no result
    yet. On a curated bank that is merely surprising, but the method is shared
    with LLM-backed pipelines, where an innocuous-looking export line becomes a
    full paid generation run — and another one on the next call. The deliberate
    replacement is to raise and name the method to call.
    """
    generator = AdversarialGenerator.from_dataframe(_bank(), capability="A")

    with pytest.raises(RuntimeError, match="Nothing has been generated yet"):
        generator.to_pandas()

    generator.generate()
    assert len(generator.to_pandas()) == 2


def test_a_run_that_produced_nothing_exports_an_empty_frame():
    """ "Ran and produced nothing" is a different state from "never ran".

    Only the second one raises. A bank whose every prompt is blank is a real
    answer — zero goldens — and the caller has to be able to see it rather than
    be told to call ``generate()`` again.
    """
    bank = _bank(Prompt=["", "  ", None])
    generator = AdversarialGenerator.from_dataframe(bank, capability="A")

    result = generator.generate()
    assert result.goldens == []
    assert generator.to_pandas().empty


# --------------------------------------------------------------------------- #
# construction paths
# --------------------------------------------------------------------------- #


def test_from_excel_and_from_dataframe_build_equivalent_generators(real_bank):
    """Two doors, one generator. ``from_excel`` is ``read_excel`` + ``from_dataframe``."""
    capability = real_bank["Capability"].iloc[0]
    from_excel = AdversarialGenerator.from_excel(
        SAMPLE_ADVERSARIAL, capability=capability, config=_config()
    )
    from_df = AdversarialGenerator.from_dataframe(
        real_bank, capability=capability, config=_config()
    )

    for generator in (from_excel, from_df):
        assert isinstance(generator.source, CuratedBankSource)
        assert generator.source.capability == capability

    assert [g.input for g in from_excel.generate().goldens] == [
        g.input for g in from_df.generate().goldens
    ]


def test_the_constructor_requires_a_source():
    """A source is not optional — the old runtime ValueError is a TypeError now."""
    with pytest.raises(TypeError):
        AdversarialGenerator()


def test_the_preset_adds_no_behaviour_over_a_plain_generator(real_bank):
    """``AdversarialGenerator`` is ``Generator(CuratedBankSource(...), stages=())``.

    Asserted because the preset exists only to keep ``from_dataframe`` /
    ``from_excel`` at hand. The moment it produces something a plain
    ``Generator`` over the same source does not, it has grown machinery of its
    own and the layering claim in ``generation/CLAUDE.md`` is no longer true.
    """
    capability = real_bank["Capability"].iloc[0]

    def source():
        return CuratedBankSource(real_bank, capability=capability)

    preset = AdversarialGenerator(source(), config=_config()).generate()
    plain = Generator(source(), config=_config()).generate()

    assert [g.input for g in preset.goldens] == [g.input for g in plain.goldens]
    assert [g.metadata for g in preset.goldens] == [g.metadata for g in plain.goldens]
    assert len(preset.rejected) == len(plain.rejected) == 0


def test_seed_makes_the_unfiltered_sample_reproducible():
    """``GenerationConfig.seed`` reaches the one nondeterministic branch.

    The unfiltered path samples the bank; the capability filters are already
    deterministic. Without ``random_state`` threaded through, this was the only
    part of the pipeline that could not be reproduced, which would have made
    ``config.py``'s "threads into every random draw" a false promise.
    """
    bank = pd.DataFrame(
        {
            "Capability": ["c"] * 20,
            "Sub Capability": ["s"] * 20,
            "Prompt": [f"p{i}" for i in range(20)],
            "Char Len": list(range(20)),
        }
    )
    source = CuratedBankSource(bank, sample_size=5)
    seeded = GenerationConfig(seed=1234, show_progress=False)

    first = [g.input for g in source.produce(seeded)]
    second = [g.input for g in source.produce(seeded)]
    assert first == second

    other = [g.input for g in source.produce(GenerationConfig(seed=99))]
    assert other != first, "a different seed should draw a different sample"


def test_without_a_seed_the_sample_is_not_pinned():
    """seed=None keeps the old nondeterministic behaviour, deliberately."""
    bank = pd.DataFrame(
        {
            "Capability": ["c"] * 200,
            "Sub Capability": ["s"] * 200,
            "Prompt": [f"p{i}" for i in range(200)],
            "Char Len": list(range(200)),
        }
    )
    source = CuratedBankSource(bank, sample_size=20)
    config = GenerationConfig(show_progress=False)

    draws = {tuple(g.input for g in source.produce(config)) for _ in range(5)}
    assert len(draws) > 1
