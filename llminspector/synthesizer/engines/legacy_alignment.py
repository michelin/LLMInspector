"""Default alignment engine — legacy tag-augment -> HF-T5 paraphrase -> perturb.

Ports the three-stage ``Alignment`` pipeline verbatim, decoupled from
configparser: the tag/augmentation dictionaries and paraphrase count are now
explicit constructor args. The perturbation stage delegates to
:mod:`llminspector.synthesizer.perturbations` (which carries the two bug fixes).

``transformers`` is imported lazily so the module stays importable without it.
This whole class is the seam a future custom, ML-free alignment engine replaces.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...dataset.golden import Golden
from .. import perturbations
from ..alignment_tag import KeywordNotFoundException, tag_replace
from .base import AlignmentEngine

logger = logging.getLogger(__name__)

_T5_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"


class LegacyTagT5Engine(AlignmentEngine):
    """Tag-augment -> HF-T5 paraphrase -> perturb."""

    #: declared so to_pandas()'s column set is knowable without running it
    metadata_keys = (
        "capability",
        "subcapability",
        "input_prompt",
        "exploded_prompt",
        "paraphrased_prompt",
        "augmentation_type",
    )

    def __init__(
        self,
        alignment_df: pd.DataFrame,
        tag_keyword_dict: Dict[str, list],
        augmentation_dict: Dict[str, list],
        augmentations: Dict[str, Tuple[str, float]],
        paraphrase_count: int = 3,
    ) -> None:
        self.alignment_df1 = alignment_df.copy()
        self.tag_keyword_dict = tag_keyword_dict
        self.augmentation_dict = augmentation_dict
        self.augmentations = augmentations
        self.paraphrase_count = paraphrase_count

        self.result_pertubated_df = pd.DataFrame()
        self.exploded_prompt: List = []
        self.augmentation_type_list1: List = []
        self.input_prompt: List = []
        self.expected_response: List = []
        self.exploded_prompt1: List = []
        self.augmentation_type1: List = []
        self.input_prompt1: List = []
        self.expected_response1: List = []

    # -- stage 1: tag augmentation -------------------------------------------

    def checklist_tagaugmentation(self) -> None:
        keyword_dict = self.tag_keyword_dict
        tag_key_list = list(keyword_dict)
        augmentation_dict = self.augmentation_dict

        align_replace_tag = tag_replace()
        self.alignment_df1["Tags"] = self.alignment_df1.UserInput.str.extract(
            r"{(.+?)}", expand=True
        )

        input_tags = self.alignment_df1["Tags"].replace("", np.nan).dropna()
        input_tags = [item.lower() for item in input_tags]

        tag_list_infile = []
        tag_list = [item.strip("{}").lower() for item in tag_key_list]
        for tag in input_tags:
            if tag in tag_list:
                tag_list_infile.append(tag)

        unique_tag_list_infile = list(set(tag_list_infile))
        logger.info("Tag keys found in the file: %s", unique_tag_list_infile)

        for tag_key in unique_tag_list_infile:
            for aug_key, aug_value in augmentation_dict.items():
                if tag_key.lower() in [value.lower() for value in aug_value]:
                    tag_value = "{" + tag_key + "}"

                    selected_input_prompt = self.alignment_df1.loc[
                        self.alignment_df1["UserInput"].str.contains(
                            tag_value, case=False
                        )
                    ]
                    selected_input_prompt_list = selected_input_prompt[
                        "UserInput"
                    ].to_list()

                    input_prompt_list = []
                    for i in self.exploded_prompt:
                        if i not in selected_input_prompt_list and re.search(
                            tag_value, i, re.IGNORECASE
                        ):
                            input_prompt_list.append(i)

                    try:
                        modified_input_prompt = align_replace_tag.replace_tag(
                            selected_input_prompt_list, tag_value, keyword_dict
                        )
                        a = modified_input_prompt
                        if modified_input_prompt is not None:
                            for t in range(len(a)):
                                for n in range(len(a[t])):
                                    self.input_prompt.append(a[t][0])
                                    self.expected_response.extend(
                                        selected_input_prompt["Expected_Result"]
                                    )
                                    self.augmentation_type_list1.append(
                                        "Different " + str(aug_key)
                                    )
                                    self.exploded_prompt.append(a[t][n])
                        else:
                            raise KeywordNotFoundException(tag_value)
                    except KeywordNotFoundException as e:
                        logger.warning("Tag replacement failed: %s", e)

    def alignment_data(self) -> pd.DataFrame:
        self.checklist_tagaugmentation()

        for t in range(len(self.alignment_df1["UserInput"])):
            if "{" not in self.alignment_df1["UserInput"][t]:
                self.exploded_prompt1.append(self.alignment_df1["UserInput"][t])
                self.input_prompt1.append(self.alignment_df1["UserInput"][t])
                self.augmentation_type1.append("None")
                self.expected_response1.append(self.alignment_df1["Expected_Result"][t])

        for i in range(len(self.exploded_prompt)):
            if (
                "{" not in self.exploded_prompt[i]
                and self.exploded_prompt[i] not in self.exploded_prompt1
            ):
                self.exploded_prompt1.append(self.exploded_prompt[i])
                self.input_prompt1.append(self.input_prompt[i])
                self.augmentation_type1.append(self.augmentation_type_list1[i])
                self.expected_response1.append(self.expected_response[i])

        zipped = list(
            zip(
                self.input_prompt1,
                self.exploded_prompt1,
                self.expected_response1,
                self.augmentation_type1,
            )
        )
        return pd.DataFrame(
            zipped,
            columns=[
                "input_prompt",
                "exploded_prompt",
                "Expected_Result",
                "augmentation_type",
            ],
        )

    # -- stage 2: HF-T5 paraphrase -------------------------------------------

    def paraphrase_prompts(
        self,
        input_df,
        num_beams=4,
        num_beam_groups=4,
        repetition_penalty=1.5,
        diversity_penalty=3.1,
        no_repeat_ngram_size=2,
        max_length=128,
    ) -> pd.DataFrame:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_T5_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(_T5_MODEL)
        num_return_sequences = self.paraphrase_count

        paraphrased_prompt_list = []
        input_prompt_list = []
        exploded_prompt_list = []
        expected_result_list = []
        augmentation_type_list = []

        tag_list = input_df["exploded_prompt"].to_list()
        df_out = pd.DataFrame()

        for i in range(len(tag_list)):
            input_ids = tokenizer(
                f"paraphrase: {tag_list[i]}",
                return_tensors="pt",
                padding="do_not_pad",
                max_length=max_length,
                truncation=True,
            ).input_ids

            input_prompt_list.append(
                [input_df["input_prompt"][i]] * num_return_sequences
            )
            exploded_prompt_list.append(
                [input_df["exploded_prompt"][i]] * num_return_sequences
            )
            expected_result_list.append(
                [input_df["Expected_Result"][i]] * num_return_sequences
            )
            augmentation_type_list.append(
                [input_df["augmentation_type"][i]] * num_return_sequences
            )

            outputs = model.generate(
                input_ids,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=no_repeat_ngram_size,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                max_length=max_length,
                diversity_penalty=diversity_penalty,
            )
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            paraphrased_prompt_list.append(res)

            flat_paraphrased = [x for sub in paraphrased_prompt_list for x in sub]
            flat_input = [x for sub in input_prompt_list for x in sub]
            flat_exploded = [x for sub in exploded_prompt_list for x in sub]
            flat_expected = [x for sub in expected_result_list for x in sub]
            flat_augmentation = [x for sub in augmentation_type_list for x in sub]

            zipped = list(
                zip(
                    flat_input,
                    flat_exploded,
                    flat_paraphrased,
                    flat_expected,
                    flat_augmentation,
                )
            )
            df_out = pd.DataFrame(
                zipped,
                columns=[
                    "input_prompt",
                    "exploded_prompt",
                    "paraphrased_prompt",
                    "Expected_Result",
                    "augmentation_type",
                ],
            )

        return df_out

    # -- stage 3: perturbation -----------------------------------------------

    _PERTURBATIONS = {
        "uppercase": perturbations.uppercase_transform,
        "lowercase": perturbations.lowercase_transform,
        "titlecase": perturbations.titlecase_transform,
        "add_punctuation": perturbations.add_punctuation,
        "strip_punctuation": perturbations.strip_punctuation,
        "typo": perturbations.add_typo,
        "context": perturbations.add_context,
        "contract": perturbations.add_contraction,
        "ocr_typo": perturbations.add_ocr_typo,
        "abbreviate": perturbations.add_abbreviation,
    }

    def transform_df(
        self, paraphrased_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        alignment_df = (
            paraphrased_df
            if paraphrased_df is not None
            else self.paraphrase_prompts(self.alignment_data())
        )

        if "capability" not in alignment_df.columns:
            alignment_df["capability"] = ""
        if "subcapability" not in alignment_df.columns:
            alignment_df["subcapability"] = ""

        for subcapability, (capability, probability) in self.augmentations.items():
            mask = np.random.rand(len(alignment_df)) < probability
            filter_df = alignment_df[mask].copy()
            logger.debug("Augmentation key: %s", subcapability)
            samples = filter_df["paraphrased_prompt"]

            transform = self._PERTURBATIONS.get(subcapability)
            perturbated_prompts = transform(samples) if transform else []

            if len(perturbated_prompts) > 0:
                filter_df = filter_df.copy()
                filter_df.loc[:, "perturbated_prompt"] = perturbated_prompts
                filter_df.loc[:, "subcapability"] = subcapability
                filter_df.loc[:, "capability"] = capability
                filter_df["augmentation_type"].fillna("None")
                column_order = [
                    "capability",
                    "subcapability",
                    "input_prompt",
                    "exploded_prompt",
                    "paraphrased_prompt",
                    "perturbated_prompt",
                    "augmentation_type",
                ]
                filter_df = filter_df[column_order]
                self.result_pertubated_df = pd.concat(
                    [self.result_pertubated_df, filter_df], ignore_index=True
                )
            else:
                logger.warning("Unknown augmentation type: %s", subcapability)

        return self.result_pertubated_df

    # -- output --------------------------------------------------------------

    def generate(self) -> List[Golden]:
        df = self.transform_df()
        return _dataframe_to_goldens(df)


def _dataframe_to_goldens(df: pd.DataFrame) -> List[Golden]:
    """Map the perturbed alignment DataFrame to goldens.

    ``input`` is the final perturbated prompt; the remaining stage columns are
    preserved in ``metadata`` so the documented output columns survive.
    """
    goldens: List[Golden] = []
    for _, row in df.iterrows():
        prompt = row.get("perturbated_prompt")
        if prompt is None or str(prompt).strip() == "":
            continue
        expected = row.get("Expected_Result")
        goldens.append(
            Golden(
                input=str(prompt),
                expected_output=None if expected is None else str(expected),
                metadata={
                    "capability": row.get("capability"),
                    "subcapability": row.get("subcapability"),
                    "input_prompt": row.get("input_prompt"),
                    "exploded_prompt": row.get("exploded_prompt"),
                    "paraphrased_prompt": row.get("paraphrased_prompt"),
                    "augmentation_type": row.get("augmentation_type"),
                },
            )
        )
    return goldens
