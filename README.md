🧠 Logical Fallacy Detection in Climate Change Misinformation

This project builds a multiclass NLP classifier that detects 11 types of logical fallacies found in online climate-change misinformation. It combines:

Classical ML (TF-IDF + Linear SVM)

Transformer-based fine-tuning (DistilRoBERTa)

Dataset balancing via random oversampling

Explainability using Qwen2.5-1.5B-Instruct

The system not only predicts the fallacy but also generates human-readable explanations using a small instruction-tuned LLM.

📌 Features
✔ Fallacy dataset preprocessing & cleaning
✔ Baseline classification using LinearSVC
✔ Transformer finetuning with weighted loss
✔ Balanced vs. unbalanced comparisons
✔ Qwen2.5-1.5B-Instruct explainer
✔ Fully modular codebase (src/)
✔ Ready for Streamlit deployment

🗂 Dataset

The dataset contains the following text components:

fact_checked_segment — argument/snippet

comment_by_fact_checker — reasoning

article — file reference

label_str — normalized fallacy label

label_id — numeric mapping using labels_climate.py

Fallacy classes include:
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

A mapping script (labels_climate.py) ensures consistent ID ↔ label conversions.

🧹 Preprocessing Pipeline

1️⃣ Clean the raw text

Includes:

lowercasing

removing special chars

whitespace normalization

Located in:

src/data/clean_text.py

2️⃣ Encode labels

Using:

from src.labels_climate import LABEL2ID, ID2LABEL

3️⃣ Drop missing entries

Ensures no NaN text reaches TF-IDF or tokenizers.

Saved in Processed Folder as :
Saved processed datasets to:
data/processed/
   climate_train.csv
   climate_dev.csv
   climate_test.csv
  
4️⃣ Dataset balancing

Oversampling minority classes to handle heavy imbalance.

Stored in:

Notebooks/Fallacies_NLP.ipynb

Produces:

data/combined_csv/climate_train_balanced.csv

🔧 Models Trained
⭐ Baseline: TF-IDF + LinearSVM

File:

src/models/baseline.py

This establishes a classical ML benchmark.

🤖 Transformer: DistilRoBERTa Fine-Tuning

File:

src/models/slm_finetune.py


Includes:

Stratified train/val split

HuggingFace datasets

Weighted loss for class imbalance

Custom WeightedTrainer

Tokenization with truncation & padding

Saving full model + tokenizer

Supports easy loading:

model = AutoModelForSequenceClassification.from_pretrained("outputs/slm_climate_multiclass")

💬 Explainability with Qwen2.5-1.5B-Instruct

File:

src/explainers/explainer_qwen.py


Produces short, human-friendly fallacy explanations.

Example usage:

explainer = QwenExplainer()
explanation = explainer.explain(text, predicted_label)

📊 Results Summary
Baseline (Unbalanced)

Accuracy: low due to heavy class imbalance
Model predicts mostly NO_FALLACY

Baseline (Balanced)

Accuracy: improved
Better recall across fallacies
Still limited by TF-IDF representation

Transformer Finetuned (Balanced)

Macro-F1 improved significantly
Performance spread across classes
Still challenging due to dataset complexity

Full tables are in the  notebook & project docs.

📁 Repository Structure
Steadx01_NLP/
│
├── src/
│   ├── data/
│   │   ├── clean_text.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── slm_finetune.py
│   │   └── __init__.py
|   |
|   |
│   │├── explainer_qwen.py
│   │└── __init__.py
│   │
│   ├── labels_climate.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── combined_csv/
│
├── notebooks/
│   ├── Fallacies_NLP.ipynb
│   
│
├── outputs/
│   └── slm_climate_multiclass/   (ignored by .gitignore)
│
├── README.md
├── .gitignore
└── requirements.txt

🚀 Installation
1️⃣ Clone the repo
git clone https://github.com/SteadyHands01/Steadx01_NLP.git
cd Steadx01_NLP

2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run Training

Baseline:

python src/models/baseline.py

Transformer:

python src/models/slm_finetune.py

💬 Generate Explanation (Qwen)

Example:

from src.explainers.explainer_qwen import QwenExplainer
explainer = QwenExplainer()
explain = explainer.explain("sample text here", "CHERRY_PICKING")
print(explain)

📦 Future Work

Add Streamlit inference app

Replace DistilRoBERTa with ModernBERT / DeBERTa-v3

Add knowledge distillation

Build a real-time misinformation monitor dashboard

🙌 Acknowledgements

HuggingFace Transformers

Qwen team

Scikit-learn

Climate change fallacy dataset authors 
Tariq60: https://github.com/Tariq60/fallacy-detection/tree/master/data

Microsoft (Phi series inspiration)


