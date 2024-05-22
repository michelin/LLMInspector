import pytest
import pandas as pd
import os
from llm_inspector.eval_metrics import EvalMetrics
from configparser import ConfigParser
from rouge import Rouge

config = ConfigParser()
config.read(".//test_sample//test_config.ini")
data = {
    'question': [
        "Can you explain the theory of relativity proposed by Albert Einstein?",
        "What are the causes and effects of global warming?",
        "Describe the process of photosynthesis in plants.",
        "What is the impact of artificial intelligence on society?",
        "How does the human digestive system work?",
        "Discuss the rise and fall of the Roman Empire.",
        "Explain the concept of supply and demand in economics.",
        "What are the benefits and drawbacks of renewable energy sources?",
        "Describe the events leading up to the American Civil War.",
        "What are the key features of democratic governance?"
    ],
    'answer': [
        "The concept of relativity proposed by Albert Einstein describes gravity as a geometric property of spacetime. It introduces new concepts of time and space, including time dilation and length contraction. The theory has been confirmed by various experiments and observations, and it forms the basis of modern physics.",
        "Global warming is primarily caused by the increase in greenhouse gases, such as carbon dioxide and methane, in the Earth's atmosphere. This leads to the trapping of heat, resulting in higher temperatures and climate change. The effects of global warming include rising sea levels, extreme weather events, and impacts on ecosystems and biodiversity.",
        "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy. It involves the absorption of light by chlorophyll, the conversion of carbon dioxide and water into glucose and oxygen, and the release of oxygen as a byproduct. Photosynthesis is crucial for the production of oxygen and the sustenance of life on Earth.",
        "The effect of artificial intelligence (AI) on society can be profound. AI has the potential to revolutionize various aspects of society, including healthcare, transportation, and communication. It can automate tasks, analyze large amounts of data, and make predictions. However, there are concerns about the ethical implications of AI, including job displacement, privacy issues, and bias in decision-making algorithms.",
        "The human digestive system is responsible for breaking down food into nutrients that can be absorbed by the body. It consists of several organs, including the mouth, esophagus, stomach, small intestine, large intestine, liver, gallbladder, and pancreas. Each organ plays a specific role in the digestion process.",
        "The rise and fall of the Roman Empire is one of the most significant events in history. The empire emerged in the 8th century BCE and expanded through conquest and colonization. However, internal strife, economic decline, and external invasions led to its eventual collapse in the 5th century CE.",
        "The concept of supply and demand in economics is fundamental to understanding how markets work. Supply refers to the amount of a product available for sale, while demand represents the desire of consumers to purchase it. The interaction between supply and demand determines the price and quantity of goods and services in a market.",
        "Renewable energy sources have numerous benefits, including reducing greenhouse gas emissions, improving energy security, and creating jobs. However, they also have drawbacks, such as intermittency, land use conflicts, and environmental impacts.",
        "The events leading to the American Civil War were complex and multifaceted. Disagreements over slavery, states' rights, and economic differences were some of the key factors that contributed to the conflict. The war resulted in significant social, political, and economic changes in the United States.",
        "Democratic governance encompasses a range of principles and practices aimed at ensuring accountability, transparency, and participation in government. Key features include free and fair elections, protection of individual rights, separation of powers, and government accountability to the public."
    ],
    'ground_truth': [
        "The theory of relativity, proposed by Albert Einstein, describes gravity as a geometric property of spacetime. It introduces new concepts of time and space, including time dilation and length contraction. The theory has been confirmed by various experiments and observations, and it forms the basis of modern physics.",
        "Global warming is primarily caused by the increase in greenhouse gases, such as carbon dioxide and methane, in the Earth's atmosphere. This leads to the trapping of heat, resulting in higher temperatures and climate change. The effects of global warming include rising sea levels, extreme weather events, and impacts on ecosystems and biodiversity.",
        "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy. It involves the absorption of light by chlorophyll, the conversion of carbon dioxide and water into glucose and oxygen, and the release of oxygen as a byproduct. Photosynthesis is crucial for the production of oxygen and the sustenance of life on Earth.",
        "Artificial intelligence (AI) has the potential to revolutionize various aspects of society, including healthcare, transportation, and communication. It can automate tasks, analyze large amounts of data, and make predictions. However, there are concerns about the ethical implications of AI, including job displacement, privacy issues, and bias in decision-making algorithms.",
        "The human digestive system is a complex series of organs and glands that process food and extract nutrients. It includes the mouth, esophagus, stomach, small intestine, large intestine, liver, gallbladder, and pancreas. Digestion involves mechanical and chemical processes that break down food into smaller molecules, which can be absorbed by the body.",
        "The Roman Empire was a powerful civilization that emerged in the 8th century BCE and lasted for over a millennium. It expanded through conquest and colonization, reaching its peak during the 2nd century CE. However, internal strife, economic decline, and external invasions led to its eventual collapse in the 5th century CE.",
        "Supply and demand is a fundamental concept in economics that determines the price and quantity of goods and services in a market. It states that the price of a product is determined by the balance between supply (the amount of a product available) and demand (the desire of consumers to purchase it). When supply exceeds demand, prices tend to fall, and vice versa.",
        "Renewable energy sources, such as solar, wind, and hydroelectric power, offer numerous benefits, including reducing greenhouse gas emissions, improving energy security, and creating jobs. However, they also have drawbacks, such as intermittency, land use conflicts, and environmental impacts.",
        "The American Civil War was a conflict fought between the northern states (Union) and the southern states (Confederacy) from 1861 to 1865. It was primarily caused by disagreements over slavery, states' rights, and economic differences. The war resulted in the abolition of slavery, the preservation of the Union, and significant social and political changes.",
        "Democratic governance is a system of government in which power is vested in the people, who exercise it directly or through elected representatives. Key features include free and fair elections, protection of individual rights and liberties, separation of powers, and accountability of government officials to the public."
    ]
}

df = pd.DataFrame(data)

def test_init():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")

    assert isinstance(evalmetric, EvalMetrics)


def test_accuracy_rouge():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    rouge_instance = Rouge()
    actual_responses = df["answer"]
    expected_responses = df["ground_truth"]
    expected_scores = []
    for i in range(len(actual_responses)):
        expected_score = rouge_instance.get_scores(actual_responses[i], expected_responses[i])[0]["rouge-l"]["f"]
        expected_scores.append(expected_score)

    # Call the method being tested
    actual_scores = evalmetric.accuracy_rouge(expected_responses, actual_responses)

    # Compare the actual scores with the expected scores
    assert actual_scores == expected_scores


def test_bertscore():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    actual_responses = df["answer"]
    expected_responses = df["ground_truth"]

    # Call the method being tested
    actual_scores = evalmetric.bertscore(expected_responses, actual_responses)

    # Ensure that the actual scores are not empty
    assert actual_scores

def test_toxicity_detection():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    # Define some sample texts
    texts = [
        "This is a normal comment.",
        "I hope you have a nice day!",
        "You're a mental.",
        "I will kill you."
    ]

    # Call the method being tested
    results = evalmetric.toxicity_detection(texts)

    # Ensure that the results have the same length as the input texts
    assert len(results) == len(texts)

    # Ensure that the results contain the expected labels
    # You may need to adjust these expected labels based on the model's output format
    expected_labels = ['non-toxic', 'non-toxic', 'toxic', 'toxic']
    assert results == expected_labels


def test_text_quality():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    actual_responses = [
        "This is a sample response.",
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "123 Main Street, Anytown, USA.",
    ]

    # Call the method being tested
    flesch_kincaid_grades, flesch_reading_eases, automated_readability_indexes = evalmetric.text_quality(actual_responses)

    # Ensure that the results have the same length as the input actual responses
    assert len(flesch_kincaid_grades) == len(actual_responses)
    assert len(flesch_reading_eases) == len(actual_responses)
    assert len(automated_readability_indexes) == len(actual_responses)


def test_lang_detect():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    texts = ["This is in English", "Este es en Español", "Ceci est en Français"]
    expected = ["en", "es", "fr"]
    assert evalmetric.lang_detect(texts) == expected

def test_sentiment_analysis():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    texts = ["I love this product!", "This is terrible.", "The movie was okay."]
    expected = ["positive", "negative", "positive"]
    assert evalmetric.sentiment_analysis(texts) == expected

def test_emotion_analysis():
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    texts = ["I am so happy!", "I am feeling sad.", "I am feeling angry."]
    expected = ["joy", "sadness", "anger"]
    assert evalmetric.emotion_analysis(texts) == expected


@pytest.mark.parametrize("texts, output_lang, expected_output", [
    (["Hello", "World"], "es", ["Hola", "Mundo"]),
])
def test_translate(texts, output_lang, expected_output):
    evalmetric = EvalMetrics(df=df, config=config, env_path=".//test_sample//.env")
    output = evalmetric.translate(texts, output_lang)
    assert output == expected_output