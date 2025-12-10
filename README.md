**Logical Fallacy Detection in Climate Change Misinformation**

This project builds a multiclass NLP classifier that detects 11 types of logical fallacies found in online climate-change misinformation. 


**Overview: Climate Logical Fallacy Detector**

A machine learning system that classifies logical fallacies in climate-related statements and generates human-readable explanations. 

This system was built using:

Classical ML (TF-IDF + Linear SVM) baseline model for benchmarking.

Transformer-based fine-tuning (DistilRoBERTa) for 11 fallacy classification.

FLAN-T5-small (lightweight LLM) for generating explanations (streamlit app).

Optional: Qwen2.5-1.5B-Instruct for deeper offline explanations.

A fully interactive Streamlit web application.

The project is structured for reproducibility, modularity, and extensibility.


**Features:**

Uses a fine-tuned DistilRoBERTa Transformer to categorize text into 11 fallacy classes:

CHERRY_PICKING

EVADING_THE_BURDEN_OF_PROOF

FALSE_ANALOGY

FALSE_AUTHORITY

FALSE_CAUSE

HASTY_GENERALISATION

NO_FALLACY

POST_HOC

RED_HERRINGS

STRAWMAN

VAGUENESS


**Human-Readable  Explanation**

After classification, the FLAN-T5-small model generates natural language explanations describing why the text may contain the detected fallacy.


**Streamlit App**

A clean UI for:

Entering input text

Running classifier

Viewing probabilities

Generating explanations

Switching models (future feature)


**Models Used**

1. TF-IDF + Linear SVM Baseline

Provides a traditional ML comparison benchmark.

2. DistilRoBERTa Fine-Tuned Classifier

Fine-tuned on climate fallacy dataset (train/dev/test splits)

Includes class-balancing through oversampling

Supports evaluation with weighted loss options

3. FLAN-T5-small Explainer

Lightweight and fast

Generates 2–3 sentence explanations

Fully integrated into Streamlit UI

4. Qwen2.5-1.5B-Instruct (Optional for local offline use)

Very strong reasoning ability

Can generate long, deep explanations

Disabled in Streamlit due to memory constraints


**Acknowledgements**

Special thanks to:

Tariq60 for the open-source fallacy-detection dataset

https://github.com/Tariq60/fallacy-detection

HuggingFace Transformers team

Qwen LLM developers (Alibaba Cloud & Qwen Team)

Google FLAN-T5 team

Microsoft for the PHI models (tested offline)

This project is built on top of outstanding open-source efforts.

All credits to the respective authors.


**Future Improvement**

Streamlit UI upgrade (tabs, themes, fallacy examples)

Add option to choose explainer model (Flan / Qwen / Phi)

Deploy to HuggingFace Space

Improve classifier performance with:

1. Better data augmentation

2. Contrastive learning

3. Prompt-based fine-tuning


**Contact**

Project Maintainer:

Faithful Kyeremeh (SteadyHands01)

GitHub: https://github.com/SteadyHands01

Email:faithfulkyeremeh@gmail.com


