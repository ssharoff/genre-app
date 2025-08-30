#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2025  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
# A basic app for streamlit

import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from IPython.display import display, HTML
from xai import XAI


@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

labels = {0: 'A1', 1: 'A4', 2: 'A7', 3: 'A8', 4: 'A9', 5: 'A11', 6: 'A12', 7: 'A14', 8: 'A16', 9: 'A17'}


st.set_page_config(layout="wide")
st.title("Genre classifier")
# ---- Layout: Input on the Left, Results on the Right ----

model_name="ssharoff/genres"

col1, col2 = st.columns([1, 2])
with col1:
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
    if input_method == "Paste Text":
        text = st.text_area("✍️ Enter text:", "", height=150)
    else:
        file = st.file_uploader("📂 Upload a text file", type=["txt"]).

    explain_xai = st.checkbox("🔎 Explain Predictions (Captum XAI)")
    predict_clicked = st.button("🔍 Analyse")

with col2:
    sentences = []
    if input_method == "Paste Text" and text.strip():
        sentences = [text]
    elif input_method == "Upload File" and file is not None:
        text = file.read().decode("utf-8")
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if len(sentences)>0:
        with st.spinner("🔄 Loading the classifier..."):
            tokenizer, model = load_model(model_name)
        with st.spinner("🔄 Running the classifier..."):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                st.write(dir(outputs))
                predicted_class = outputs.logits.argmax().item()
                label=labels.get(predicted_class, "Prediction error")
                st.write(f'Predicted genre: {label}')

                if explain_xai:
                    st.subheader("🔍 Integrated Gradients Explainability")
                    with st.spinner("Computing attributions..."):
                        xai = XAI(text, label, tokenizer, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                        html_output, top_attributions = xai.generate_html()
                    st.write(HTML(html_output), unsafe_allow_html=True)
                    st.write("📋 **Top Attributed Words**")
                    st.dataframe(top_attributions.sort_values(by='Attributions', key=abs))

                    # Provide download link
                    csv = top_attributions.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Attributions", data=csv, file_name="classified_sentences.csv", mime="text/csv")

    else:
        st.warning("⚠️ Please enter some text to classify.")
        sentences = []

