#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Copyright (C) 2025  Serge Sharoff
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
# A basic app for streamlit

import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xai import XAI


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

labels = {0: 'A1', 1: 'A4', 2: 'A7', 3: 'A8', 4: 'A9', 5: 'A11', 6: 'A12', 7: 'A14', 8: 'A16', 9: 'A17'}


model_name="ssharoff/genres"
fname=sys.argv[1]
with sys.open(fname) as f:
    text = f.read().decode("utf-8")
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences)>0:
        tokenizer, model = load_model(model_name)
        if len(sentences)>0:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                sys.stderr.write(dir(outputs))
                predicted_class = outputs.logits.argmax().item()
                label=labels.get(predicted_class, "Prediction error")
                #st.write(f'Predicted genre: {label}')
                if explain_xai:
                    if predicted_class:
                        xai = XAI(text, label, tokenizer, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                        html_output, top_attributions = xai.generate_html()
                    sys.stderr.write(html_output)
                    print(top_attributions.sort_values(by='Attributions', key=abs))

                    # Provide download link
                    csv = top_attributions.to_csv(index=False).encode('utf-8')
                    #st.download_button(label="📥 Download Attributions", data=csv, file_name="classified_sentences.csv", mime="text/csv")

    else:
        sys.stderr.write("⚠️ Please enter some text to classify.")
        sentences = []

