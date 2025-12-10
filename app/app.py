# Streamlit

import sys
from pathlib import Path


import streamlit as st
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.labels_climate import ID2LABEL, LABEL2ID
from src.explainer_flan import FlanExplainer, FlanExplainerConfig


# Configuration
# Use PROJECT_ROOT so the path works regardless of where Streamlit is launched from
MODEL_DIR = "SteadyHands/climate-fallacy-roberta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Configure Streamlit Page 
st.set_page_config(
     page_title= "Climate Fallacy Detector",
     page_icon="🤡",
     layout="wide",
)

# Cached Loaders
# Load Fallacy Classifier

@st.cache_resource
def load_classifier():
    """Load fine-tuned DistilRoBERTa classifier and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# Load Flan Explainer Model

@st.cache_resource
def load_flan_explainer():
    """This loads Flan-based Explainer which is light-weight. Qwen crashed so we are using Flan for Explanations"""
    cfg = FlanExplainerConfig()
    explainer = FlanExplainer(cfg)
    return explainer

classifier_tokenizer, classifier_model = load_classifier()


# Classification Function for Text

def classify_text(text: str):
    inputs = classifier_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = classifier_model(**inputs)

    logits = outputs.logits[0].detach().cpu().numpy()
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    pred_id = int(np.argmax(logits))
    pred_label = ID2LABEL[pred_id]
    return pred_label, pred_id, probs


# Main Streamlit App UI
# Layout: Side Bar
st.sidebar.title(" 🌟 About App ")
st.sidebar.markdown(
    """
    🐹 This App showcases:

- A **fine-tuned DistilRoBERTa** model for classifying  
  logical fallacies in climate-related text  
- A **FLAN-T5-small** explainer that generates  
  short, human-readable justifications.

- Offline experiments used a larger SLM (Qwen2.5-1.5B-Instruct).
- For this live demo, a lighter model (FLAN-T5-small) is used due to hardware limits.

**Workflow**

1. Enter or paste a climate-related statement  
2. Click **“Classify Fallacy”**  
3. Click **“Generate Explanation”** to ask Flan-T5-small
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Device:** `" + DEVICE + "`")
st.sidebar.markdown("**Model dir:** `" + str(MODEL_DIR) + "`")

# Now the Main Layout --> Ladies and Gentlemen

st.title("🤡 Climate Logical Fallacy Detector")
st.caption(
    "Detects common logical fallacies in climate-change related text and "
    "generates natural-language explanations."
)

# Pre-fill with an example to guide users
default_text = (
    "Climate has always changed in the past, so current global warming "
    "cannot be caused by human activity."
)

user_text = st.text_area(
    "Enter a climate-related argument or social media post:",
    value=default_text,
    height=200,
)

col_left, col_right = st.columns([1, 1])

with col_left:
    classify_btn = st.button("🔍 Classify Fallacy", type="primary")

with col_right:
    #  Make the explanation button clickable; logic below decides if we can actually explain
    explain_btn = st.button("👽 Generate Explanation (FLAN)")

# Helper to display prediction + confidence + probability chart."""

def show_classification(pred_label, pred_id, probs):
    st.success(f"**Predicted fallacy:** {pred_label}")

    # Confidence information on input
    top_conf = float(probs[pred_id])
    st.write(f"**Model confidence:** {top_conf:.2%}")

    # Probability bar chart
    st.subheader("📊 Class probabilities")
    prob_df = pd.DataFrame(
        {
            "Fallacy": [ID2LABEL[i] for i in range(len(ID2LABEL))],
            "Probability": probs,
        }
    ).sort_values("Probability", ascending=False)

    st.bar_chart(
        prob_df.set_index("Fallacy")["Probability"],
        height=250,
    )

#  Now for goodness sake, lets store some of user predictions in session_state so explanation can use it 
if "last_pred_label" not in st.session_state:
    st.session_state.last_pred_label = None
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None
if "last_text" not in st.session_state:
    st.session_state.last_text = None

# Now lets classify shall we?

if classify_btn:
    if not user_text.strip():
        st.warning("Please enter some text before classifying.")
    else:
        with st.spinner("Running classifier..."):
            pred_label, pred_id, probs = classify_text(user_text)

        # store in session_state
        st.session_state.last_pred_label = pred_label
        st.session_state.last_probs = probs
        st.session_state.last_text = user_text

        show_classification(pred_label, pred_id, probs)
        

# RoyalMajesty Qwen Explanations ; take her serious
# Unfortunately she crashed due to hardware limitations "hehehehehe"
# Let me introduce you to FLAN-T5-small , our sunshine 🌟

if explain_btn:
    if not user_text.strip():
        st.warning("Please enter some text before requesting an explanation.")
    else:
       # Always (re)classify for the current text to be 100% safe
        with st.spinner("Running classifier before explanation..."):
            pred_label, pred_id, probs = classify_text(user_text)

        # Store new prediction
        st.session_state.last_pred_label = pred_label
        st.session_state.last_probs = probs
        st.session_state.last_text = user_text

        # Show classification being explained
        st.subheader("🔍 Classification Result (used for explanation)")
        show_classification(pred_label, pred_id, probs)

         # Now lets get explanation from FLAN 
        st.subheader("👑 Explanation (FLAN-T5-small)")
        with st.spinner("Asking FLAN for an explanation..."):
            try:
                explainer = load_flan_explainer()
                explanation = explainer.explain(
                    user_text,
                    pred_label,
                )
                st.write(explanation)
            except Exception as e:
                st.error(
                    "There was an error while generating the explanation. "
                    "Please check logs or try again."
                )
                st.exception(e)
