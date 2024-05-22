from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from textstat import textstat
import evaluate
from deep_translator import GoogleTranslator
from transformers import pipeline
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
from presidio_analyzer import AnalyzerEngine
from langdetect import detect
from datasets import Dataset
from ragas.metrics import answer_correctness, answer_similarity
from ragas import evaluate
import datetime
from transformers import logging
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import os
from pathlib import Path
from dotenv import load_dotenv

logging.set_verbosity(logging.ERROR)


class EvalMetrics:
    def __init__(self, df, config, env_path, out_dir=None):
        dotenv_path = Path(env_path)
        load_dotenv(dotenv_path=dotenv_path)
        self.api_version = os.getenv("api_version")
        self.azure_endpoint = os.getenv("azure_endpoint")
        self.api_key = os.getenv("api_key")
        # config = ConfigParser()
        # config.read(config_path)
        self.initialize_models()
        Insight_File = config["Insights_File"]
        dt_time = datetime.datetime.now()
        self.question_col = Insight_File["question_col"]
        self.answer_col = Insight_File["answer_col"]
        self.ground_truth_col = Insight_File["ground_truth_col"]
        self.threshold = float(Insight_File["threshold"])
        self.df = df
        self.result_df = df
        self.output_lang = Insight_File["output_lang"]
        self.output_path = (
            Insight_File["Insights_output_path"]
            + Insight_File["Insights_Output_fileName"]
            + "_"
            + str(dt_time.year)
            + str(dt_time.month)
            + str(dt_time.day)
            + "_"
            + str(dt_time.hour)
            + str(dt_time.minute)
            + ".xlsx"
        )
        self.Ins_output = out_dir if out_dir is not None else self.output_path
        self.metrics_list = Insight_File["Metrics"]
        self.metrics_list = eval(self.metrics_list)
        if len(self.metrics_list) == 0:
            self.metrics_list = [
                "rouge_score",
                "bert_score",
                "answer_similarity",
                "answer_correctness",
                "question_toxicity",
                "answer_toxicity",
                "pii_detection",
                "readability",
                "translate",
                "question_language",
                "answer_language",
                "question_sentiment",
                "answer_sentiment",
                "question_emotion",
                "answer_emotion",
            ]
        print("eval_metrics initiated")

    def initialize_models(self):
        """
        Initializes the required models.
        """
        self.azure_model = AzureChatOpenAI(
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment="gpt-35-turbo-16k",
            model="gpt-35-turbo-16k",
            api_key=self.api_key,
            validate_base_url=False,
            timeout=180,
            temperature=1.0,
        )

        self.azure_embeddings = AzureOpenAIEmbeddings(
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            api_key=self.api_key,
            timeout=180,
        )

    def calculate_metrics(self):
        if "rouge_score" in self.metrics_list:
            rouge_score = self.accuracy_rouge(
                self.df[self.ground_truth_col], self.df[self.answer_col]
            )
            self.result_df["rouge-l"] = rouge_score
            print("rouge done")
        if "bert_score" in self.metrics_list:
            bertscore = self.bertscore(
                self.df[self.ground_truth_col], self.df[self.answer_col]
            )
            self.result_df["bertscore"] = bertscore
            print("bertscore done")
        if "answer_similarity" in self.metrics_list:
            answer_similarity = self.answer_similarity_eval(
                self.df[self.question_col],
                self.df[self.answer_col],
                self.df[self.ground_truth_col],
            )
            self.result_df["answer similarity"] = answer_similarity
            print("answer similarity done")
        if "answer_correctness" in self.metrics_list:
            answer_correctness = self.answer_correctness_eval(
                self.df[self.question_col],
                self.df[self.answer_col],
                self.df[self.ground_truth_col],
            )
            self.result_df["answer correctness"] = answer_correctness
            print("answer correctness done")
        if "question_toxicity" in self.metrics_list:
            question_toxicity = self.toxicity_detection(
                self.df[self.question_col].to_list()
            )
            self.result_df["question toxicity"] = question_toxicity
            print("qn toxicity")
        if "answer_toxicity" in self.metrics_list:
            answer_toxicity = self.toxicity_detection(
                self.df[self.answer_col].to_list()
            )
            self.result_df["answer toxicity"] = answer_toxicity
            print("answer toxicity done")
        if "pii_detection" in self.metrics_list:
            pii_detection = self.PII_detection(self.df[self.answer_col].to_list())
            self.result_df["PII"] = pii_detection
            print("pii detection done")
        if "readability" in self.metrics_list:
            (
                flesch_kincaid_grades,
                flesch_reading_eases,
                automated_readability_indexes,
            ) = self.text_quality(self.df[self.answer_col].to_list())
            self.result_df["flesch_kincaid_grades"] = flesch_kincaid_grades
            self.result_df["flesch_reading_eases"] = flesch_reading_eases
            self.result_df["automated_readability_indexes"] = (
                automated_readability_indexes
            )
            print("readability index done")
        if "translate" in self.metrics_list:
            traslated_text = self.translate(
                self.df[self.answer_col].to_list(), output_lang=self.output_lang
            )
            self.result_df["traslated_text"] = traslated_text
            print("Translation done")
        if "question_language" in self.metrics_list:
            question_language = self.lang_detect(self.df[self.question_col].to_list())
            self.result_df["question_language"] = question_language
            print("question lang detect done")
        if "answer_language" in self.metrics_list:
            answer_language = self.lang_detect(self.df[self.answer_col].to_list())
            self.result_df["answer_language"] = answer_language
            print("answer lang detect done")
        if "question_sentiment" in self.metrics_list:
            question_sentiment = self.sentiment_analysis(
                self.df[self.question_col].to_list()
            )
            self.result_df["question_sentiment"] = question_sentiment
            print("Question sentiment done")
        if "answer_sentiment" in self.metrics_list:
            answer_sentiment = self.sentiment_analysis(
                self.df[self.answer_col].to_list()
            )
            self.result_df["answer_sentiment"] = answer_sentiment
            print("answer sentiment done")
        if "question_emotion" in self.metrics_list:
            question_emotion = self.emotion_analysis(
                self.df[self.question_col].to_list()
            )
            self.result_df["question_emotion"] = question_emotion
            print("Question emotion done")
        if "answer_emotion" in self.metrics_list:
            answer_emotion = self.emotion_analysis(self.df[self.answer_col].to_list())
            self.result_df["answer_emotion"] = answer_emotion
            print("answer emotion done")
        if "bert_score" in self.metrics_list:
            bertscore = self.df["bertscore"].to_list()
            self.df["Result"] = self.df["bertscore"].apply(
                lambda x: "Pass" if x > self.threshold else "Fail"
            )

        return self.result_df

    def accuracy_rouge(self, expected_responses, actual_responses):
        rouge = Rouge()
        rouge_scores = []
        for i in range(len(actual_responses)):
            score = rouge.get_scores(actual_responses[i], expected_responses[i])
            rouge_scores.append(score[0]["rouge-l"]["f"])
        return rouge_scores

    def accuracy_bleu(self, expected_responses, actual_responses):
        bleu_scores = []
        for i in range(len(expected_responses)):
            reference = [expected_responses[i].split()]
            candidate = actual_responses[i].split()
            bleu_scores.append(sentence_bleu(reference, candidate))
        return bleu_scores

    def bertscore(self, references, predictions):
        bertscore = load("bertscore")
        results = bertscore.compute(
            predictions=predictions, references=references, lang="en"
        )["f1"]
        return results

    def toxicity_detection(self, texts):
        model_path = "martin-ha/toxic-comment-model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        results = pipeline(texts)
        results = [i["label"] for i in results]
        return results

    def PII_detection(self, texts):
        analyzer = AnalyzerEngine()
        entities = [
            "CREDIT_CARD",
            "CRYPTO",
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "IP_ADDRESS",
            "NRP",
            "LOCATION",
            "PERSON",
            "PHONE_NUMBER",
            "MEDICAL_LICENSE",
            "URL",
            "IN_PAN",
            "IN_AADHAAR",
            "IN_VEHICLE_REGISTRATION",
        ]
        results = [
            analyzer.analyze(text=text, entities=entities, language="en")
            for text in texts
        ]
        return results

    def text_quality(self, actual_responses):
        flesch_kincaid_grades = []
        flesch_reading_eases = []
        automated_readability_indexes = []
        for i in actual_responses:
            flesch_kincaid_grades.append(textstat.flesch_kincaid_grade(i))
            flesch_reading_eases.append(textstat.flesch_reading_ease(i))
            automated_readability_indexes.append(
                textstat.automated_readability_index(i)
            )
        return (
            flesch_kincaid_grades,
            flesch_reading_eases,
            automated_readability_indexes,
        )

    def translate(self, texts, output_lang):
        translator = GoogleTranslator(source="auto", target=output_lang)
        batch_output = translator.translate_batch(texts)
        return batch_output

    def lang_detect(self, texts):
        det_langs = [detect(i) for i in texts]
        return det_langs

    def sentiment_analysis(self, texts):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_task = pipeline(
            "sentiment-analysis", model=model_path, tokenizer=model_path
        )
        sentiments = sentiment_task(texts)
        sentiments = [i["label"] for i in sentiments]
        return sentiments

    def emotion_analysis(self, texts):
        classifier = pipeline(
            "text-classification", model="SamLowe/roberta-base-go_emotions"
        )
        emotions = classifier(texts)
        emotions = [i["label"] for i in emotions]
        return emotions

    def answer_similarity_eval(self, questions, answer, ground_truth):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_similarity],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["answer_similarity"].to_list()

    def answer_correctness_eval(self, questions, answer, ground_truth):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_correctness],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["answer_correctness"].to_list()

    def export_insights(self):
        print("Insights can be obtained on the provided path:", self.Ins_output)
        self.result_df.to_excel(self.Ins_output)
        return self.result_df
