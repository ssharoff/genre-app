import torch
from captum.attr import LayerIntegratedGradients
import pandas as pd
from collections import defaultdict
import string
from utils import detect_language, STOPWORDS_DICT, PUNCTUATION
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Captum XAI Class
class XAI:
    def __init__(self, text, label, tokenizer, model, device):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.input_ids = None
        self.ref_input_ids = None
        self.predicted_label = None

    def construct_input_ref(self):
        text_ids = self.tokenizer.encode(self.text, add_special_tokens=False)
        input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]
        ref_input_ids = [self.tokenizer.cls_token_id] + [self.tokenizer.pad_token_id] * len(text_ids) + [self.tokenizer.sep_token_id]
        self.input_ids = torch.tensor([input_ids], device=self.device)
        self.ref_input_ids = torch.tensor([ref_input_ids], device=self.device)
        return self.input_ids, self.ref_input_ids

    def filter_stopwords_punctuation(self, words, attributions, text):
        detected_lang = detect_language(text)
        stopwords_set = STOPWORDS_DICT.get(detected_lang, STOPWORDS_DICT["english"])
        filtered_words, filtered_attributions = [], []

        for word, attribution in zip(words, attributions):
            if word.lower() not in stopwords_set and word not in PUNCTUATION:
                filtered_words.append(word)
                filtered_attributions.append(attribution)

        return filtered_words, filtered_attributions

    def aggregate_token_attributions(self, attributions, tokens):
        word_attributions = defaultdict(float)
        word_list = []
        current_word = ""

        for i, token in enumerate(tokens):
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    word_list.append(current_word)
                current_word = token

            word_attributions[current_word] += attributions[i].item()

        if current_word:
            word_list.append(current_word)

        word_attributions_list = [word_attributions[word] for word in word_list]
        return word_list, word_attributions_list

    def compute_attributions(self):
        self.input_ids, self.ref_input_ids = self.construct_input_ref()
        self.tokens = [t for t in self.tokenizer.convert_ids_to_tokens(self.input_ids[0]) if t not in ["[CLS]", "[SEP]"]]

        with torch.no_grad():
            logits = self.model(self.input_ids)[0]
            probs = torch.softmax(logits, dim=1)
            self.predicted_label = torch.argmax(probs, dim=1).item()

        def forward_func(inputs):
            logits = self.model(inputs)[0]
            probs = torch.softmax(logits, dim=1)
            return probs[:, self.predicted_label]

        lig = LayerIntegratedGradients(forward_func, self.model.bert.embeddings)

        attributions, _ = lig.attribute(
            inputs=self.input_ids,
            baselines=self.ref_input_ids,
            n_steps=500,
            internal_batch_size=3,
            return_convergence_delta=True
        )

        token_attributions = attributions.sum(dim=-1).squeeze()
        words, word_attributions = self.aggregate_token_attributions(token_attributions, self.tokens)

        norm = torch.norm(torch.tensor(word_attributions)) + 1e-8
        normalized = [float(attr) / norm for attr in word_attributions]

        return self.filter_stopwords_punctuation(words, normalized, self.text)

    def predict_probabilities(self):
        logits = self.model(self.input_ids)[0]
        probs = torch.softmax(logits, dim=1).squeeze()
        return probs.tolist()

    def generate_html(self):
        words, word_attributions = self.compute_attributions()
        probabilities = self.predict_probabilities()

        label_names = ['Simple', 'Complex']
        predicted_label_name = label_names[self.predicted_label]

        # Highlighted tokens
        max_abs_attr = max(abs(a) for a in word_attributions) or 1.0
        token_html = ""
        for word, score in zip(words, word_attributions):
            alpha = abs(score / max_abs_attr)
            color = f"rgba(255, 0, 0, {alpha:.2f})" if score > 0 else f"rgba(0, 128, 0, {alpha:.2f})"
            token_html += f"<span style='background-color: {color}; padding: 2px;'>{word} </span>"

        # Attribution table
        top_attributions = (
            pd.DataFrame({'Word': words, 'Attribution': word_attributions})
              .assign(Abs=lambda df: df['Attribution'].abs())
              .sort_values(by='Abs', ascending=False)
              .drop(columns='Abs')
              .head(10)
        )

        # HTML output
        html_content = f"""
        <div style="margin-bottom: 20px;">
            <h4>Prediction Probabilities</h4>
            <div>
                <div>Simple</div>
                <div style="width: 100%; height: 20px; background-color: #eee; border-radius: 5px; margin: 5px 0;">
                    <div style="width: {probabilities[0] * 100:.2f}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
                </div>
                <p>Probability: {probabilities[0]:.2f}</p>
                <div>Complex</div>
                <div style="width: 100%; height: 20px; background-color: #eee; border-radius: 5px; margin: 5px 0;">
                    <div style="width: {probabilities[1] * 100:.2f}%; height: 100%; background-color: #F44336; border-radius: 5px;"></div>
                </div>
                <p>Probability: {probabilities[1]:.2f}</p>
            </div>
            <h4>Text with Highlighted Words</h4>
            <p>{token_html}</p>
            <div style="margin-top: 10px; font-size: 14px;">
                <strong>Legend for <em>{predicted_label_name}</em> prediction:</strong>
                <span style="background-color: rgba(255, 0, 0, 0.5); padding: 2px; margin-left: 5px;">Red</span> = supports prediction,
                <span style="background-color: rgba(0, 128, 0, 0.5); padding: 2px; margin-left: 5px;">Green</span> = opposes prediction
            </div>
        </div>
        """
        return html_content, top_attributions
