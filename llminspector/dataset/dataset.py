"""The ``EvaluationDataset`` — a container of test cases and/or goldens with
pandas / Excel (de)serialization and configurable column-name mapping.

Column defaults reproduce the legacy ``helper.py`` ``*_col`` arguments
(``question / answer / ground_truth / contexts / policy``) so existing
spreadsheets load unchanged.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..test_case.test_case import LLMTestCase
from .golden import Golden


@dataclass
class ColumnMapping:
    """Maps :class:`LLMTestCase` attributes to spreadsheet column names.

    Defaults mirror the legacy ``helper.process_batch_metrics`` column args.
    """

    input_col: str = "question"
    actual_output_col: str = "answer"
    expected_output_col: str = "ground_truth"
    retrieval_context_col: str = "contexts"
    policy_col: str = "policy"


@dataclass
class GoldenColumnMapping:
    """Maps :class:`Golden` attributes to spreadsheet column names.

    ``id_col`` is the one addition to the legacy set. It is read back when
    present and written on export, so a golden keeps its identity across the
    round trip instead of being handed a fresh uuid on every reload — which
    would sever the ``golden_id`` link on any test case promoted from it.
    """

    input_col: str = "question"
    expected_output_col: str = "ground_truth"
    context_col: str = "contexts"
    id_col: str = "id"


def _cell(value: Any) -> Optional[str]:
    """Normalize a scalar spreadsheet cell to ``str`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _is_missing(value: Any) -> bool:
    """True for ``None`` and for pandas' NaN/NaT, without stringifying."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Lists, dicts and other containers make ``pd.isna`` return an array or
        # raise; a container is by definition present.
        return False


def _parse_context(value: Any) -> Optional[List[str]]:
    """Parse a context cell into a list of strings.

    Handles native lists/tuples, ``repr``-style ``"['a', 'b']"`` strings
    (as written back by :meth:`to_excel`), and plain single-context strings.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if str(v).strip() != ""]
        return items or None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "":
        return None
    if text[:1] in "[(":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                items = [str(v) for v in parsed if str(v).strip() != ""]
                return items or None
        except (ValueError, SyntaxError):
            pass
    return [text]


class EvaluationDataset:
    """Holds evaluation ``test_cases`` and synthesis ``goldens``.

    Construct directly, or via the ``from_pandas`` / ``from_excel`` (test
    cases) and ``goldens_from_pandas`` / ``goldens_from_excel`` classmethods.
    """

    def __init__(
        self,
        test_cases: Optional[List[LLMTestCase]] = None,
        goldens: Optional[List[Golden]] = None,
    ) -> None:
        self.test_cases: List[LLMTestCase] = list(test_cases or [])
        self.goldens: List[Golden] = list(goldens or [])

    def __len__(self) -> int:
        return len(self.test_cases)

    def __repr__(self) -> str:
        return (
            f"EvaluationDataset(test_cases={len(self.test_cases)}, "
            f"goldens={len(self.goldens)})"
        )

    # -- test cases -----------------------------------------------------------

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`LLMTestCase` from a DataFrame."""
        mapping = _resolve_mapping(ColumnMapping, mapping, col_overrides)
        test_cases: List[LLMTestCase] = []
        for _, row in df.iterrows():
            input_value = _cell(row.get(mapping.input_col))
            if input_value is None:
                continue
            test_cases.append(
                LLMTestCase(
                    input=input_value,
                    actual_output=_cell(row.get(mapping.actual_output_col)),
                    expected_output=_cell(row.get(mapping.expected_output_col)),
                    retrieval_context=_parse_context(
                        row.get(mapping.retrieval_context_col)
                    ),
                    policy=_cell(row.get(mapping.policy_col)),
                )
            )
        return cls(test_cases=test_cases)

    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet_name: Any = 0,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`LLMTestCase` from an ``.xlsx`` file."""
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls.from_pandas(df, mapping=mapping, **col_overrides)

    def to_pandas(
        self,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> pd.DataFrame:
        """Serialize ``test_cases`` back to a DataFrame using the mapping."""
        mapping = _resolve_mapping(ColumnMapping, mapping, col_overrides)
        rows = []
        for tc in self.test_cases:
            rows.append(
                {
                    mapping.input_col: tc.input,
                    mapping.actual_output_col: tc.actual_output,
                    mapping.expected_output_col: tc.expected_output,
                    mapping.retrieval_context_col: tc.retrieval_context,
                    mapping.policy_col: tc.policy,
                }
            )
        return pd.DataFrame(rows, columns=_test_case_columns(mapping))

    def to_excel(
        self,
        path: str,
        mapping: Optional[ColumnMapping] = None,
        **col_overrides: str,
    ) -> None:
        """Write ``test_cases`` to an ``.xlsx`` file."""
        self.to_pandas(mapping=mapping, **col_overrides).to_excel(path, index=False)

    # -- goldens --------------------------------------------------------------

    @classmethod
    def goldens_from_pandas(
        cls,
        df: pd.DataFrame,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`Golden` from a DataFrame.

        Every column that is not one of the four mapped core fields is collected
        into :attr:`Golden.metadata`. ``Golden`` sets ``extra="ignore"``, so
        without this the generation lineage, quality scores and source-file
        columns written by :meth:`goldens_to_pandas` vanished silently on
        reload — the export said one thing and the reload said another.
        """
        mapping = _resolve_mapping(GoldenColumnMapping, mapping, col_overrides)
        core = {
            mapping.input_col,
            mapping.expected_output_col,
            mapping.context_col,
            mapping.id_col,
        }
        extra_cols = [c for c in df.columns if c not in core]

        goldens: List[Golden] = []
        for _, row in df.iterrows():
            input_value = _cell(row.get(mapping.input_col))
            if input_value is None:
                continue
            # Missing cells are dropped rather than stored as None: exporting a
            # heterogeneous batch unions every golden's metadata keys and fills
            # the gaps with NaN, so keeping them would give every golden every
            # other golden's keys after one round trip.
            metadata = {
                col: row[col] for col in extra_cols if not _is_missing(row.get(col))
            }
            fields: dict = {
                "input": input_value,
                "expected_output": _cell(row.get(mapping.expected_output_col)),
                "context": _parse_context(row.get(mapping.context_col)),
                "metadata": metadata,
            }
            # Absent id column -> let the default factory mint one.
            golden_id = _cell(row.get(mapping.id_col))
            if golden_id is not None:
                fields["id"] = golden_id
            goldens.append(Golden(**fields))
        return cls(goldens=goldens)

    @classmethod
    def goldens_from_excel(
        cls,
        path: str,
        sheet_name: Any = 0,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> "EvaluationDataset":
        """Build a dataset of :class:`Golden` from an ``.xlsx`` file."""
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls.goldens_from_pandas(df, mapping=mapping, **col_overrides)

    def goldens_to_pandas(
        self,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> pd.DataFrame:
        """Serialize ``goldens`` back to a DataFrame using the mapping.

        Delegates to :func:`goldens_to_dataframe`, so the id and metadata
        columns appear here too. This used to hardcode the three core columns
        and drop ``metadata`` entirely, which made the export lossy in exactly
        the direction that matters — everything a generator adds lives in
        metadata.
        """
        mapping = _resolve_mapping(GoldenColumnMapping, mapping, col_overrides)
        return goldens_to_dataframe(self.goldens, mapping=mapping)

    def goldens_to_excel(
        self,
        path: str,
        mapping: Optional[GoldenColumnMapping] = None,
        **col_overrides: str,
    ) -> None:
        """Write ``goldens`` to an ``.xlsx`` file."""
        self.goldens_to_pandas(mapping=mapping, **col_overrides).to_excel(
            path, index=False
        )

    # -- promotion ------------------------------------------------------------

    def to_test_cases(
        self,
        answers: Optional[Sequence[Optional[str]]] = None,
        policies: Optional[Sequence[Optional[str]]] = None,
    ) -> List[LLMTestCase]:
        """Promote ``goldens`` to :class:`LLMTestCase` objects.

        The documented workflow is export goldens → run them through the system
        under test → re-import with answers. This is the in-memory equivalent,
        so a generate-then-evaluate script never has to touch a spreadsheet.

        Each case carries its golden's ``id`` on ``golden_id`` and a copy of its
        ``metadata``, so results stay traceable to the golden that produced them.

        Parameters
        ----------
        answers:
            System answers, positionally aligned with ``goldens``. Omitted
            entirely, every case gets ``actual_output=None``.
        policies:
            Policy text per golden, same alignment. Goldens carry no policy of
            their own — it belongs to the evaluation, not the seed.

        Raises
        ------
        ValueError
            When a supplied sequence's length does not match ``goldens``.
            Silently zipping to the shorter of the two would attach answers to
            the wrong questions, which no downstream check would catch.
        """
        for name, values in (("answers", answers), ("policies", policies)):
            if values is not None and len(values) != len(self.goldens):
                raise ValueError(
                    f"{name} has {len(values)} item(s) but there are "
                    f"{len(self.goldens)} golden(s); they must align positionally."
                )
        return [
            golden.to_test_case(
                actual_output=answers[i] if answers is not None else None,
                policy=policies[i] if policies is not None else None,
            )
            for i, golden in enumerate(self.goldens)
        ]


def goldens_to_dataframe(
    goldens: Sequence[Golden],
    mapping: Optional[GoldenColumnMapping] = None,
) -> pd.DataFrame:
    """Flatten goldens to a DataFrame: id, core fields, then metadata columns.

    Metadata columns are the union of every golden's keys, in first-seen order,
    so the column set is stable and a golden missing a key gets a blank cell
    rather than shifting the table.

    ``mapping`` selects the column *names*:

    * ``None`` (the default) uses the :class:`Golden` attribute names —
      ``id`` / ``input`` / ``expected_output`` / ``context``. This is what a
      generator's own export wants: a fresh artefact, named after the model.
    * a :class:`GoldenColumnMapping` uses the spreadsheet names, which is what
      :meth:`EvaluationDataset.goldens_to_pandas` passes so its output stays
      readable by :meth:`EvaluationDataset.goldens_from_pandas`. Those defaults
      are a frozen compatibility surface — see ``dataset/CLAUDE.md``.
    """
    if mapping is None:
        id_col, input_col, expected_col, context_col = (
            "id",
            "input",
            "expected_output",
            "context",
        )
    else:
        id_col = mapping.id_col
        input_col = mapping.input_col
        expected_col = mapping.expected_output_col
        context_col = mapping.context_col

    core = (id_col, input_col, expected_col, context_col)
    metadata_keys: List[str] = []
    seen = set(core)
    for golden in goldens:
        for key in golden.metadata:
            # A metadata key colliding with a core column would overwrite it in
            # the record dict; the core field wins and the collision is skipped.
            if key not in seen:
                seen.add(key)
                metadata_keys.append(key)

    records = []
    for golden in goldens:
        record: Dict[str, Any] = {
            id_col: golden.id,
            input_col: golden.input,
            expected_col: golden.expected_output,
            context_col: golden.context,
        }
        for key in metadata_keys:
            record[key] = golden.metadata.get(key)
        records.append(record)

    return pd.DataFrame(records, columns=list(core) + metadata_keys)


def _test_case_columns(mapping: ColumnMapping) -> List[str]:
    return [
        mapping.input_col,
        mapping.actual_output_col,
        mapping.expected_output_col,
        mapping.retrieval_context_col,
        mapping.policy_col,
    ]


def _resolve_mapping(mapping_cls, mapping, col_overrides):
    """Merge an explicit mapping with per-column keyword overrides."""
    base = mapping if mapping is not None else mapping_cls()
    if col_overrides:
        valid = set(base.__dataclass_fields__)
        unknown = set(col_overrides) - valid
        if unknown:
            raise TypeError(
                f"Unknown column argument(s): {sorted(unknown)}. "
                f"Valid: {sorted(valid)}"
            )
        base = mapping_cls(**{**base.__dict__, **col_overrides})
    return base
