# app.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2025
# A basic app for Streamlit

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st

from xai import XAI


@st.cache_resource
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# You can switch label sets—these are example mappings for the classifier
# labels = {0: 'A1', 1: 'A4', 2: 'A7', 3: 'A8', 4: 'A9', 5: 'A11', 6: 'A12', 7: 'A14', 8: 'A16', 9: 'A17'}
labels = {0: 'argue', 1: 'fiction', 2: 'instruct', 3: 'newswire', 4: 'regulatory', 5: 'personal', 6: 'promote', 7: 'research', 8: 'reference', 9: 'review'}

st.set_page_config(layout="wide")
st.title("Genre classifier")

# Model name 
MODEL_NAME = "ssharoff/genres"

col1, col2 = st.columns([1, 2])
with col1:
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
    if input_method == "Paste Text":
        text = st.text_area("✍️ Enter text:", "", height=150)
    else:
        file = st.file_uploader("📂 Upload a text file", type=["txt"])
        text = ""
        if file is not None:
            text = file.read().decode("utf-8")

    explain_xai = st.checkbox("🔎 Explain Predictions (Captum XAI)")
    top_k_attributions = st.number_input("Top_k_attributions: ", min_value=0, value=10)

    predict_clicked = st.button("🔍 Analyse")

with col2:
    if predict_clicked:
        # Prepare text
        if input_method == "Paste Text" and not text.strip():
            st.warning("⚠️ Please enter some text to classify.")
        elif input_method == "Upload File" and not text.strip():
            st.warning("⚠️ The uploaded file seems empty.")
        else:
            with st.spinner("🔄 Loading the classifier..."):
                tokenizer, model = load_model(MODEL_NAME)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

            with st.spinner("🔄 Running the classifier..."):
                enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                # move inputs to the same device as model
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    outputs = model(**enc)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    predicted_class = int(logits.argmax(dim=-1).item())

                label_name = labels.get(predicted_class, f"class_{predicted_class}")
            st.markdown(f"### Predicted genre: **{label_name}**")

            if explain_xai:
                st.subheader("🔍 Integrated Gradients Explainability")
                with st.spinner("Computing attributions..."):
                    # XAI will re-tokenize because it needs special baseline
                    xai = XAI(text, label_name, tokenizer, model, device)
                    html_output, top_attributions = xai.generate_html(label_names=labels, top_k_attributions=top_k_attributions)

                # Render explanation HTML
                st.components.v1.html(html_output, height=500, scrolling=True)

                # Show top attributed words
                st.write("📋 **Top Attributed Words**")
                st.dataframe(top_attributions.sort_values(by="Attribution", key=abs, ascending=False))

                # Download CSV
                csv = top_attributions.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Attributions",
                    data=csv,
                    file_name="word_attributions.csv",
                    mime="text/csv",
                )
