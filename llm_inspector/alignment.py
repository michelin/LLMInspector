import llm_inspector.alignment_replace_function as align_replace
import pandas as pd
import re
import datetime
from collections import defaultdict
import ast
from copy import deepcopy
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from llm_inspector.constants import (
    TYPO_FREQUENCY,
    CONTRACTION_MAP,
    ending_context,
    starting_context,
    ocr_typo_dict,
    abbreviation_dict,
    starting_context,
    ending_context,
)

inverted_ocr_typo_dict = defaultdict(list)
for k, v in ocr_typo_dict.items():
    inverted_ocr_typo_dict[v].append(k)


class KeywordNotFoundException(Exception):
    def __init__(self, tag_value, message="Keyword not found in the config File"):
        self.tag_value = tag_value
        self.message = message
        super().__init__(f"{message}: {tag_value}")


class Alignment:

    def __init__(
        self,
        config,
        inputfile=None,
        paraphrase_count=None,
        outfilepath=None,
        capability=None,
        subcapability=None,
    ):
        self.tag_file_path = config["Alignment_File"]
        self.tag_augmentation = config["Tag_Augmentation_Key"]
        self.augmentation_type = config["Augmentation_Type"]
        self.paraphrase_count = paraphrase_count
        dt_time = datetime.datetime.now()
        augmentations = ast.literal_eval(self.tag_file_path["augmentations"])

        self.infile = (
            self.tag_file_path["Alignment_input_FilePath"]
            + self.tag_file_path["Alignment_GoldenDataset_FileName"]
        )
        self.inputfile = inputfile if inputfile is not None else self.infile

        alignment_df = pd.read_excel(self.inputfile)
        self.alignment_df1 = alignment_df

        self.output_path = (
            self.tag_file_path["Alignment_output_path"]
            + self.tag_file_path["Alignment_Output_fileName"]
            + str(dt_time.year)
            + str(dt_time.month)
            + str(dt_time.day)
            + "__"
            + str(dt_time.hour)
            + str(dt_time.minute)
            + ".xlsx"
        )
        self.align_output_path = (
            outfilepath if outfilepath is not None else self.output_path
        )

        if capability is not None:
            self.capability = capability
        else:
            self.capability = []

        if subcapability is not None:
            self.subcapability = subcapability
        else:
            self.subcapability = []

        #self.editor = Editor()  # Initializing Checklist

        self.augmentations = augmentations
        self.result_pertubated_df = pd.DataFrame()

        self.exploded_prompt = []
        self.augmentation_type_list1 = []
        self.input_prompt = []
        self.expected_response = []
        self.exploded_prompt1 = []
        self.augmentation_type1 = []
        self.input_prompt1 = []
        self.expected_response1 = []

    def checklist_tagaugmentation(self):
        keyword_dict = ast.literal_eval(self.tag_augmentation["tag_keyword_dict"])
        tag_key_list = list(keyword_dict)

        augmentation_dict = ast.literal_eval(
            self.augmentation_type["augmentation_dict"]
        )
        aug_value = augmentation_dict.values()

        align_replace_tag = align_replace.tag_replace()
        # getting tags in User input
        self.alignment_df1["Tags"] = self.alignment_df1.UserInput.str.extract(
            r"{(.+?)}", expand=True
        )
        #self.editor = Editor()

        input_tags = self.alignment_df1["Tags"].replace("", np.nan).dropna()
        input_tags = [item.lower() for item in input_tags]

        tag_list_infile = []

        tag_list = [item.strip("{}").lower() for item in tag_key_list]
        # lexicon_key_list = [
        #     "country",
        #     "city",
        #     "nationality",
        #     "female",
        #     "religion",
        #     "sexual_adj",
        # ]

        for tag in input_tags:
            if tag in tag_list:
                tag_list_infile.append(tag)

        # for tag in input_tags:
        #     if tag in lexicon_key_list:
        #         lexicon_key_infile.append(tag)

        tag_key_list = set(tag_list_infile)
        unique_tag_list_infile = list(tag_key_list)
        print("Tag key found in the file : ", unique_tag_list_infile)

        #lexicon_key_list = set(lexicon_key_infile)
        #unique_lexicon_key_inFile = list(lexicon_key_list)
        #print("Lexicon key found in the file : ", unique_lexicon_key_inFile)

        # For Loop for Pertubating data with given Tags as part of Tag List
        for tag_key in unique_tag_list_infile:
            for aug_key, aug_value in augmentation_dict.items():
                if tag_key.lower() in [value.lower() for value in aug_value]:
                    tag_value = "{" + tag_key + "}"

                    input_prompt_list = []  # d1
                    selected_input_prompt_list = []  # d2

                    selected_input_prompt = self.alignment_df1.loc[
                        self.alignment_df1["UserInput"].str.contains(
                            tag_value, case=False
                        )
                    ]
                    selected_input_prompt_list = selected_input_prompt[
                        "UserInput"
                    ].to_list()

                    for i in self.exploded_prompt:
                        if i not in selected_input_prompt_list and re.search(tag_value, i, re.IGNORECASE):
                            input_prompt_list.append(i)

                    modified_input_prompt = []
                    
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
                        print("Error:", e)

        # for lexicon_key in unique_lexicon_key_inFile:
        #     key_value = "{" + lexicon_key + "}"

        #     input_prompt_list = []
        #     selected_input_prompt = self.alignment_df1.loc[
        #         self.alignment_df1["UserInput"].str.contains(key_value, case=False)
        #     ]
        #     selected_input_prompt_list = selected_input_prompt["UserInput"].to_list()

        #     for i in self.exploded_prompt:
        #         if i not in selected_input_prompt_list and re.search(key_value, i, re.IGNORECASE):
        #             input_prompt_list.append(i)

        #     replaced_prompt_list = []
        #     replaced_prompt_list = selected_input_prompt_list + input_prompt_list

        #     modified_input_prompt = []

        #     for j in range(len(replaced_prompt_list)):
        #         ret = self.editor.template(replaced_prompt_list[j])
        #         modified_input_prompt = ret.data[0:5]
        #         for n in range(len(modified_input_prompt)):
        #             self.input_prompt.append(replaced_prompt_list[j])
        #             self.expected_response.extend(
        #                 selected_input_prompt["Expected_Result"]
        #             )
        #             self.augmentation_type_list1.append(
        #                 "Checklist Lexicon Keys_" + str(lexicon_key)
        #             )
        #             # self.self.augmentation_type.append(self.self.augmentation_type[y])
        #             self.exploded_prompt.append(modified_input_prompt[n])

    def alignment_data(self):
        self.checklist_tagaugmentation()

        for t in range(len(self.alignment_df1["UserInput"])):
            if "{" not in self.alignment_df1["UserInput"][t]:
                # if alignment_df1['UserInput'][t] not in exploded_prompt:
                self.exploded_prompt1.append(self.alignment_df1["UserInput"][t])
                self.input_prompt1.append(self.alignment_df1["UserInput"][t])
                self.augmentation_type1.append("None")
                self.expected_response1.append(self.alignment_df1["Expected_Result"][t])

        for i in range(len(self.exploded_prompt)):
            if "{" not in self.exploded_prompt[i] and self.exploded_prompt[i] not in self.exploded_prompt1:
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
        replaced_prompt = pd.DataFrame(
            zipped,
            columns=[
                "input_prompt",
                "exploded_prompt",
                "Expected_Result",
                "augmentation_type",
            ],
        )

        return replaced_prompt

    def paraphrase_prompts(
        self,
        input_df,
        num_beams=4,
        num_beam_groups=4,
        repetition_penalty=1.5,
        diversity_penalty=3.1,
        no_repeat_ngram_size=2,
        max_length=128,
    ):

        tokenizer = AutoTokenizer.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        )
        num_return_sequences = self.paraphrase_count if self.paraphrase_count is not None else int(self.tag_file_path["paraphrase_count"])
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

            flat_list_paraphrased = [
                item for sub_list in paraphrased_prompt_list for item in sub_list
            ]
            flat_list_input_prompt = [
                item for sub_list_1 in input_prompt_list for item in sub_list_1
            ]
            flat_list_exploded_prompt = [
                item for sub_list_2 in exploded_prompt_list for item in sub_list_2
            ]
            flat_list_expected_result = [
                item for sub_list_2 in expected_result_list for item in sub_list_2
            ]
            flat_list_augmentation_type = [
                item for sub_list_2 in augmentation_type_list for item in sub_list_2
            ]

            zipped = list(
                zip(
                    flat_list_input_prompt,
                    flat_list_exploded_prompt,
                    flat_list_paraphrased,
                    flat_list_expected_result,
                    flat_list_augmentation_type,
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

    def transform_df(self, prob=1.0):
        alignment_df = self.alignment_data()

        alignment_df = self.paraphrase_prompts(alignment_df)

        if "capability" not in alignment_df.columns:
            alignment_df["capability"] = ""
        if "subcapability" not in alignment_df.columns:
            alignment_df["subcapability"] = ""

        for subcapability, (capability, probability) in self.augmentations.items():
            mask = np.random.rand(len(alignment_df)) < probability
            filter_df = alignment_df[mask].copy()
            print(f"augmentation key: {subcapability}")
            perturbated_prompts = []
            samples = filter_df["paraphrased_prompt"]

            if subcapability == "uppercase":
                perturbated_prompts = self.uppercase_transform(samples)
            elif subcapability == "lowercase":
                perturbated_prompts = self.lowercase_transform(samples)
            elif subcapability == "titlecase":
                perturbated_prompts = self.titlecase_transform(samples)
            elif subcapability == "add_punctuation":
                perturbated_prompts = self.add_punctuation(samples)
            elif subcapability == "strip_punctuation":
                perturbated_prompts = self.strip_punctuation(samples)
            elif subcapability == "typo":
                perturbated_prompts = self.add_typo(samples)
            elif subcapability == "context":
                perturbated_prompts = self.add_context(
                    samples,
                    starting_context=starting_context,
                    ending_context=ending_context,
                )
            elif subcapability == "contract":
                perturbated_prompts = self.add_contraction(samples)
            elif subcapability == "ocr_typo":
                perturbated_prompts = self.add_ocr_typo(samples)
            elif subcapability == "abbreviate":
                perturbated_prompts = self.add_abbreviation(samples)

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
                # print(filter_df.head())
                self.result_pertubated_df = pd.concat(
                    [self.result_pertubated_df, filter_df], ignore_index=True
                )
            else:
                print(f"The augmentation type {subcapability} doest not exist")

        return self.result_pertubated_df

    @staticmethod
    def uppercase_transform(sample_list, prob=1.0):
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

    @staticmethod
    def lowercase_transform(sample_list, prob=1.0):
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

    @staticmethod
    def add_punctuation(sample_list, prob=1.0, whitelist=None, count=1):
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

    @staticmethod
    def strip_punctuation(sample_list, prob=1.0, whitelist=None, count=1):
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

    @staticmethod
    def add_typo(sample_list, error_rate=0.5):
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
                        word = word[:pos] + word[pos + 1:]
                    elif typo_type == "substitute":
                        pos = random.randint(0, len(word) - 1)
                        char_to_substitute = random.choice("abcdefghijklmnopqrstuvwxyz")
                        word = word[:pos] + char_to_substitute + word[pos + 1:]
                typoed_words.append(word)
            perturbed_samples.append(" ".join(typoed_words))
        return perturbed_samples

    @staticmethod
    def add_context(
        sample_list,
        prob=1.0,
        starting_context=None,
        ending_context=None,
        strategy=None,
        count=1,
    ):
        def context(text, strategy):
            possible_methods = ["start", "end", "combined"]
            if strategy is None:
                strategy = random.choice(possible_methods)
            elif strategy not in possible_methods:
                print("strategy not in possible methods.")

            if (strategy == "start" or strategy == "combined") and random.random() < prob:
                add_tokens = random.choice(starting_context)
                add_string = (
                    " ".join(add_tokens)
                    if isinstance(add_tokens, list)
                    else add_tokens
                )
                if text != "-":
                    text = add_string + " " + text

            if (strategy == "end" or strategy == "combined") and random.random() < prob:
                add_tokens = random.choice(ending_context)
                add_string = (
                    " ".join(add_tokens)
                    if isinstance(add_tokens, list)
                    else add_tokens
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

    @staticmethod
    def add_contraction(sample_list, prob=1.0):
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
        return sample_list

    @staticmethod
    def add_ocr_typo(sample_list, prob=1.0, count: int = 1):
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

    @staticmethod
    def add_abbreviation(sample_list, prob=1.0):
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

        return sample_list

    @staticmethod
    def titlecase_transform(sample_list, prob=1.0):
        perturbed_samples = []

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                words = sample.split()
                num_transform_words = int(prob * len(words))
                transformed_indices = random.sample(
                    range(len(words)), num_transform_words
                )
                transformed_words = [
                    words[i].title() if i in transformed_indices else words[i]
                    for i in range(len(words))
                ]
                perturbed_samples.append(" ".join(transformed_words))
        return perturbed_samples

    def export_alignment_data(self):
        self.transform_df()

        if self.result_pertubated_df is not None:
            self.result_pertubated_df.to_excel(self.align_output_path, index=False)
        else:
            raise ValueError(
                "DataFrame is not initialized. Please provide the correct path."
            )
        return self.align_output_path
