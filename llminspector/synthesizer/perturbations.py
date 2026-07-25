"""Text perturbation transforms used by the alignment synthesizer.

Extracted verbatim from the legacy ``Alignment`` static methods, with the two
silent no-op bugs fixed (see below). These are engine-agnostic string helpers:
a future custom alignment engine can reuse them unchanged, so they live outside
the swappable engine.

Lookup tables come from :mod:`llminspector.data`. Two legacy bugs are fixed:

* ``add_contraction`` and ``add_abbreviation`` both built ``perturbed_samples``
  but ``return sample_list`` (the unmodified input) — they now return the
  perturbed output. (REFACTOR_PHASES.md Phase 5 ⚠️.)
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import List, Optional

from ..data import CONTRACTION_MAP, abbreviation_dict, ocr_typo_dict

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _inverted_ocr_typo_dict():
    inverted = defaultdict(list)
    for k, v in ocr_typo_dict.items():
        inverted[v].append(k)
    return inverted


def uppercase_transform(sample_list, prob: float = 1.0) -> List[str]:
    transformed_samples = []
    for sample in sample_list:
        words = sample.split()
        num_transform_words = int(prob * len(words))
        transformed_indices = random.sample(range(len(words)), num_transform_words)
        transformed_words = [
            words[i].upper() if i in transformed_indices else words[i]
            for i in range(len(words))
        ]
        transformed_samples.append(" ".join(transformed_words))
    return transformed_samples


def lowercase_transform(sample_list, prob: float = 1.0) -> List[str]:
    transformed_samples = []
    for sample in sample_list:
        words = sample.split()
        num_transform_words = int(prob * len(words))
        transformed_indices = random.sample(range(len(words)), num_transform_words)
        transformed_words = [
            words[i].lower() if i in transformed_indices else words[i]
            for i in range(len(words))
        ]
        transformed_samples.append(" ".join(transformed_words))
    return transformed_samples


def titlecase_transform(sample_list, prob: float = 1.0) -> List[str]:
    perturbed_samples = []
    for sample in sample_list:
        if isinstance(sample, str):
            words = sample.split()
            num_transform_words = int(prob * len(words))
            transformed_indices = random.sample(range(len(words)), num_transform_words)
            transformed_words = [
                words[i].title() if i in transformed_indices else words[i]
                for i in range(len(words))
            ]
            perturbed_samples.append(" ".join(transformed_words))
    return perturbed_samples


def add_punctuation(
    sample_list, prob: float = 1.0, whitelist: Optional[list] = None, count: int = 1
) -> List[str]:
    if whitelist is None:
        whitelist = ["!", "?", ",", ".", "-", ":", ";"]

    def check_whitelist(text, whitelist):
        for ij in whitelist:
            text = text.replace(ij, "")
        chosen_punc = random.choice(whitelist)
        return text + chosen_punc

    perturbed_samples = []
    for s in sample_list:
        sample = deepcopy(s)
        for _ in range(count):
            if random.random() < prob:
                perturbed_samples.append(check_whitelist(sample, whitelist))
    return perturbed_samples


def strip_punctuation(
    sample_list,
    prob: float = 1.0,
    whitelist: Optional[list] = None,
    # kept for signature parity with add_punctuation
    count: int = 1,  # pylint: disable=unused-argument
) -> List[str]:
    if whitelist is None:
        whitelist = ["!", "?", ",", ".", "-", ":", ";"]

    def check_whitelist(text, whitelist):
        for i in whitelist:
            text = text.replace(i, "")
        return text

    perturbed_samples = []
    for s in sample_list:
        sample = deepcopy(s)
        if random.random() < prob:
            perturbed_samples.append(check_whitelist(sample, whitelist))
    return perturbed_samples


def add_typo(sample_list, error_rate: float = 0.5) -> List[str]:
    perturbed_samples = []
    for sentence in sample_list:
        words = sentence.split()
        typoed_words = []
        for word in words:
            if random.random() < error_rate and len(word) > 1:
                typo_type = random.choice(["insert", "delete", "substitute"])
                if typo_type == "insert":
                    pos = random.randint(0, len(word) - 1)
                    char_to_insert = random.choice("abcdefghijklmnopqrstuvwxyz")
                    word = word[:pos] + char_to_insert + word[pos:]
                elif typo_type == "delete":
                    pos = random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos + 1 :]
                elif typo_type == "substitute":
                    pos = random.randint(0, len(word) - 1)
                    char_to_substitute = random.choice("abcdefghijklmnopqrstuvwxyz")
                    word = word[:pos] + char_to_substitute + word[pos + 1 :]
            typoed_words.append(word)
        perturbed_samples.append(" ".join(typoed_words))
    return perturbed_samples


def add_context(
    sample_list,
    prob: float = 1.0,
    starting_context=None,
    ending_context=None,
    strategy=None,
    count: int = 1,
) -> List[str]:
    if starting_context is None or ending_context is None:
        from ..data import ending_context as _ec
        from ..data import starting_context as _sc

        if starting_context is None:
            starting_context = _sc
        if ending_context is None:
            ending_context = _ec

    def context(text, strategy):
        possible_methods = ["start", "end", "combined"]
        if strategy is None:
            strategy = random.choice(possible_methods)
        elif strategy not in possible_methods:
            logger.warning("Strategy %r is not a known perturbation.", strategy)

        if strategy in ("start", "combined") and random.random() < prob:
            add_tokens = random.choice(starting_context)
            add_string = (
                " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
            )
            if text != "-":
                text = add_string + " " + text

        if strategy in ("end", "combined") and random.random() < prob:
            add_tokens = random.choice(ending_context)
            add_string = (
                " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
            )
            if text != "-":
                text = text + " " + add_string

        return text

    perturbed_samples = []
    for s in sample_list:
        for _ in range(count):
            sample = deepcopy(s)
            sample = context(sample, strategy)
            perturbed_samples.append(sample)
    return perturbed_samples


def add_contraction(sample_list, prob: float = 1.0) -> List[str]:
    def custom_replace(match):
        token = match.group(0)
        contracted_token = CONTRACTION_MAP.get(
            token, CONTRACTION_MAP.get(token.lower())
        )
        is_upper_case = token[0]
        expanded_contraction = is_upper_case + contracted_token[1:]
        return expanded_contraction

    def search_contraction(text):
        replaced_string = text
        for contraction in CONTRACTION_MAP:
            search = re.search(contraction, text, flags=re.IGNORECASE | re.DOTALL)
            if search and (random.random() < prob):
                replaced_string = re.sub(
                    contraction,
                    custom_replace,
                    replaced_string,
                    flags=re.IGNORECASE | re.DOTALL,
                )
        return replaced_string

    perturbed_samples = []
    for sample in sample_list:
        perturbed_samples.append(search_contraction(sample))
    return perturbed_samples  # FIX: legacy returned the unmodified sample_list


def add_ocr_typo(sample_list, prob: float = 1.0, count: int = 1) -> List[str]:
    inverted_ocr_typo_dict = _inverted_ocr_typo_dict()

    def ocr_typo(regex, text):
        perturbed_text = text
        for word, typo_word in inverted_ocr_typo_dict.items():
            typo_word = random.choice(typo_word)
            matches = re.finditer(regex, perturbed_text)
            for match in matches:
                start = match.start()
                end = match.end()
                token = perturbed_text[start:end]
                if token.lower() == word and (random.random() < prob):
                    if token.isupper():
                        typo_word = typo_word.upper()
                    perturbed_text = (
                        perturbed_text[:start] + typo_word + perturbed_text[end:]
                    )
        return perturbed_text

    perturbed_samples = []
    for s in sample_list:
        for _ in range(count):
            sample = deepcopy(s)
            sample = ocr_typo(r"[^,\s.!?]+", sample)
            perturbed_samples.append(sample)
    return perturbed_samples


def add_abbreviation(sample_list, prob: float = 1.0) -> List[str]:
    def insert_abbreviation(text):
        perturbed_text = text
        for abbreviation, expansions in abbreviation_dict.items():
            for expansion in expansions:
                pattern = r"(?i)\b" + re.escape(expansion) + r"\b"
                corrected_token = abbreviation
                matches = re.finditer(pattern, perturbed_text)
                for match in matches:
                    start = match.start()
                    end = match.end()
                    token = perturbed_text[start:end]
                    if corrected_token != token and (random.random() < prob):
                        perturbed_text = (
                            perturbed_text[:start]
                            + corrected_token
                            + perturbed_text[end:]
                        )
        return perturbed_text

    perturbed_samples = []
    for sample in sample_list:
        perturbed_samples.append(insert_abbreviation(sample))
    return perturbed_samples  # FIX: legacy returned the unmodified sample_list
