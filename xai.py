# xai.py
import torch
from captum.attr import LayerIntegratedGradients
import pandas as pd
from collections import defaultdict
from utils import detect_language, STOPWORDS_DICT, PUNCTUATION


class XAI:
    def __init__(self, text, label, tokenizer, model, device):
        """
        text: str
        label: string label name (not used for logits; kept for display/compat)
        tokenizer: HF tokenizer
        model: HF AutoModelForSequenceClassification
        device: torch.device
        """
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.model = model.to(device)  # ensure same device
        self.device = device

        self.input_ids = None
        self.ref_input_ids = None
        self.predicted_label = None
        self.tokens = None

        self.model.eval()

    # ---------- helpers ----------

    def construct_input_ref(self):
        # Manually add special tokens so we can create a matching "reference" (baseline)
        text_ids = self.tokenizer.encode(self.text, add_special_tokens=False)
        input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]
        ref_input_ids = (
            [self.tokenizer.cls_token_id]
            + [self.tokenizer.pad_token_id] * len(text_ids)
            + [self.tokenizer.sep_token_id]
        )
        self.input_ids = torch.tensor([input_ids], device=self.device)
        self.ref_input_ids = torch.tensor([ref_input_ids], device=self.device)
        return self.input_ids, self.ref_input_ids

    def _logits(self, inputs):
        """
        Get logits whether the model returns a tuple or a ModelOutput.
        """
        out = self.model(inputs)
        return out.logits if hasattr(out, "logits") else out[0]

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
        """
        Merge subword token attributions into word-level attributions.
        Supports:
          - WordPiece (BERT/DeBERTa): continuation marked by "##"
          - SentencePiece (RoBERTa/XLM-R): word starts marked by "▁"
        """
        word_attributions = defaultdict(float)
        word_list = []
        current_word = ""

        for i, tok in enumerate(tokens):
            if tok.startswith("##"):  # WordPiece continuation
                current_word += tok[2:]
            elif tok.startswith("▁"):  # SentencePiece word start
                if current_word:
                    word_list.append(current_word)
                current_word = tok[1:]  # strip leading marker
            else:
                # New piece without explicit marker: treat as new word
                if current_word:
                    word_list.append(current_word)
                current_word = tok

            val = attributions[i]
            if hasattr(val, "item"):
                val = val.item()
            word_attributions[current_word] += float(val)

        if current_word:
            word_list.append(current_word)

        word_attributions_list = [word_attributions[w] for w in word_list]
        return word_list, word_attributions_list

    # ----------  XAI ----------

    def compute_attributions(self):
        self.input_ids, self.ref_input_ids = self.construct_input_ref()

        # Convert ids back to tokens for visualization/aggregation
        all_tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[0])
        # drop specials in the pretty view
        self.tokens = [t for t in all_tokens if t not in ["[CLS]", "[SEP]", "<s>", "</s>"]]

        with torch.no_grad():
            logits = self._logits(self.input_ids)
            probs = torch.softmax(logits, dim=1)
            self.predicted_label = int(torch.argmax(probs, dim=1).item())

        def forward_func(inputs):
            logits = self._logits(inputs)
            probs = torch.softmax(logits, dim=1)
            return probs[:, self.predicted_label]

        # Hook embeddings in a model-agnostic way
        embeddings_layer = self.model.get_input_embeddings()
        lig = LayerIntegratedGradients(forward_func, embeddings_layer)

        attributions, _ = lig.attribute(
            inputs=self.input_ids,
            baselines=self.ref_input_ids,
            n_steps=200,                # 200 is often enough; tweak if needed
            internal_batch_size=4,
            return_convergence_delta=True,
        )

        # Sum over embedding dim -> [seq_len]
        token_attributions = attributions.sum(dim=-1).squeeze(0)

        # Drop the special tokens to align with self.tokens. For BERT it's [CLS] ... [SEP],
        # for RoBERTa/XLM-R it's <s> ... </s> — both patterns are single special at start and end.
        if token_attributions.shape[0] >= 2:
            token_attributions = token_attributions[1:-1]

        # Aggregate subwords into words
        words, word_attributions = self.aggregate_token_attributions(token_attributions, self.tokens)

        # Normalize for stable coloring (L2)
        denom = (torch.norm(torch.tensor(word_attributions)) + 1e-8).item()
        normalized = [float(a) / denom for a in word_attributions]

        return self.filter_stopwords_punctuation(words, normalized, self.text)

    def predict_probabilities(self):
        logits = self._logits(self.input_ids)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        return probs.tolist()

    def generate_html(self, label_names=None, top_k_probs=5):
        """
        Build HTML with highlighted tokens and a small probability bar chart.
        label_names: optional list or dict mapping id->name for display.
        top_k_probs: show top-k probabilities.
        """
        words, word_attributions = self.compute_attributions()
        probs = self.predict_probabilities()

        # Probabilities display (top-k)
        prob_pairs = list(enumerate(probs))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = prob_pairs[:top_k_probs]

        def label_of(i):
            if isinstance(label_names, dict):
                return label_names.get(i, str(i))
            if isinstance(label_names, (list, tuple)) and i < len(label_names):
                return label_names[i]
            return str(i)

        prob_bars = ""
        for idx, p in top_pairs:
            name = label_of(idx)
            prob_bars += f"""
                <div style="margin-top: 6px;">{name}</div>
                <div style="width: 100%; height: 18px; background-color: #eee; border-radius: 5px;">
                    <div style="width: {p * 100:.2f}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
                </div>
                <div style="font-size: 12px;">{p:.3f}</div>
            """

        # Highlighted text
        max_abs_attr = max(abs(a) for a in word_attributions) if word_attributions else 1.0
        if max_abs_attr == 0:
            max_abs_attr = 1.0
        token_html = ""
        for w, s in zip(words, word_attributions):
            alpha = min(1.0, abs(s) / max_abs_attr)
            # Red = supports predicted class, Green = opposes
            color = f"rgba(255, 0, 0, {alpha:.2f})" if s > 0 else f"rgba(0, 128, 0, {alpha:.2f})"
            token_html += f"<span style='background-color:{color}; padding:1px 2px; margin:1px; border-radius:3px;'>{w}</span> "

        predicted_name = label_of(self.predicted_label)

        html_content = f"""
        <div style="margin-bottom: 16px;">
            <h4 style="margin:0 0 8px 0;">Prediction: <code>{predicted_name}</code></h4>
            <div>{prob_bars}</div>
            <h4 style="margin:16px 0 8px 0;">Text with Highlighted Words</h4>
            <p style="line-height:1.9;">{token_html}</p>
            <div style="margin-top: 6px; font-size: 12px; opacity: 0.8;">
                <strong>Legend:</strong>
                <span style="background-color: rgba(255, 0, 0, 0.5); padding: 0 4px; border-radius:3px;">Red</span> supports,
                <span style="background-color: rgba(0, 128, 0, 0.5); padding: 0 4px; border-radius:3px;">Green</span> opposes
            </div>
        </div>
        """
        # Table of top attributions
        top_attributions = (
            pd.DataFrame({"Word": words, "Attribution": word_attributions})
            .assign(Abs=lambda df: df["Attribution"].abs())
            .sort_values(by="Abs", ascending=False)
            .drop(columns="Abs")
            .head(10)
        )

        return html_content, top_attributions
