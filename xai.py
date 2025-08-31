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
          - WordPiece (##)
          - SentencePiece (▁)
          - Byte-level BPE (Ġ)
        """
        from collections import defaultdict
        word_attributions = defaultdict(float)
        word_list = []
        current_word = ""
    
        # Detect scheme(s) present in this tokenization
        uses_wp = any(t.startswith("##") for t in tokens)           # WordPiece continuation
        uses_spm = any(t.startswith("▁") for t in tokens)           # SentencePiece word-start
        uses_bpe_space = any(t.startswith("Ġ") for t in tokens)     # RoBERTa/GPT-2 space marker

        def add_attr(val, key):
            if hasattr(val, "item"):
                val = val.item()
            word_attributions[key] += float(val)
    
        for i, tok in enumerate(tokens):
            # --- WordPiece path ---
            if uses_wp and tok.startswith("##"):
                piece = tok[2:]
                current_word += piece
                add_attr(attributions[i], current_word)
                continue
    
            # --- SentencePiece path (XLM-R/XLNet style) ---
            if uses_spm:
                if tok.startswith("▁"):
                    # start a new word
                    if current_word:
                        word_list.append(current_word)
                    current_word = tok[1:]  # drop leading marker
                else:
                    # continuation of the current word
                    if current_word == "":
                        current_word = tok
                    else:
                        current_word += tok
                add_attr(attributions[i], current_word)
                continue
    
            # --- Byte-level BPE path (RoBERTa/GPT-2 style) ---
            if uses_bpe_space:
                if tok.startswith("Ġ"):
                    if current_word:
                        word_list.append(current_word)
                    current_word = tok[1:]  # drop the space marker
                else:
                    if current_word == "":
                        current_word = tok
                    else:
                        current_word += tok
                add_attr(attributions[i], current_word)
                continue
    
            # --- Fallback: treat non-marked tokens as new words ---
            if current_word:
                word_list.append(current_word)
            current_word = tok
            add_attr(attributions[i], current_word)
    
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

    def generate_html(self, label_names=None, top_k_probs=5, top_k_attributions=10):
        """
        Build HTML with highlighted tokens and a compact probability bar list.
        - Predicted class bar is GREEN.
        - Other bars are RED.
        - Green chips: supports predicted class, Red chips: opposes predicted class.
        """
        words, word_attributions = self.compute_attributions()
        probs = self.predict_probabilities()
    
        # Sort by prob (desc) and take top-k
        prob_pairs = list(enumerate(probs))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = prob_pairs[:top_k_probs]
    
        def label_of(i):
            if isinstance(label_names, dict):
                return label_names.get(i, str(i))
            if isinstance(label_names, (list, tuple)) and i < len(label_names):
                return label_names[i]
            return str(i)
    
        # Colors
        green = "#43A047"   # predicted class bar + positive token attribution
        red = "#E53935"     # non-predicted bars + negative token attribution
        track = "#ECEFF1"   # bar track
        text_muted = "#607D8B"
    
        # Build bar list
        bars_html = ""
        for idx, p in top_pairs:
            name = label_of(idx)
            width_pct = max(0.4, p * 100)  # ensure tiny probs are still visible
            fill = green if idx == self.predicted_label else red
            value = f"{p:.3f}"
    
            bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label">{name}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{width_pct:.2f}%; background:{fill};"></div>
                        <div class="bar-value">{value}</div>
                    </div>
                </div>
            """
    
        # Highlighted text chips with tooltips
        max_abs_attr = max(abs(a) for a in word_attributions) if word_attributions else 1.0
        if max_abs_attr == 0:
            max_abs_attr = 1.0
    
        chips_html = ""
        for w, s in zip(words, word_attributions):
            alpha = min(1.0, abs(s) / max_abs_attr)
            # Positive supports predicted class -> green; negative opposes -> red
            chip_color = f"rgba(67, 160, 71, {alpha:.2f})" if s > 0 else f"rgba(229, 57, 53, {alpha:.2f})"
            chips_html += (
                f"<span class='chip' title='attribution: {s:+.4f}' "
                f"style='background:{chip_color};'>{w}</span>"
            )
    
        predicted_name = label_of(self.predicted_label)
    
        html_content = f"""
        <style>
          .section h4 {{ margin: 0 0 10px 0; }}
          .bar-row {{
            display: grid;
            grid-template-columns: 140px 1fr;
            align-items: center;
            gap: 10px;
            margin: 6px 0;
          }}
          .bar-label {{ font-weight: 500; }}
          .bar-track {{
            position: relative;
            height: 20px;
            background: {track};
            border-radius: 6px;
            overflow: hidden;
          }}
          .bar-fill {{
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
          }}
          .bar-value {{
            position: absolute;
            right: 8px;
            top: 0;
            height: 100%;
            display: flex;
            align-items: center;
            font-size: 12px;
            color: {text_muted};
            font-variant-numeric: tabular-nums;
          }}
          .chips {{
            line-height: 2.1;
          }}
          .chip {{
            display: inline-block;
            padding: 2px 6px;
            margin: 2px 4px 2px 0;
            border-radius: 6px;
            color: #111;
            background: #ddd;
            cursor: default;
            text-decoration: none;
          }}
          .legend small {{ color: {text_muted}; }}
          .legend .swatch {{
            display:inline-block; width: 30px; height: 12px; border-radius: 3px; margin: 0 6px -2px 6px;
          }}
        </style>
    
        <div class="section">
          <h4>Prediction: <code>{predicted_name}</code></h4>
          <div>{bars_html}</div>
    
          <h4 style="margin-top:16px;">Text with Highlighted Words</h4>
          <div class="chips">{chips_html}</div>
    
          <div class="legend" style="margin-top:8px; font-size:12px;">
            <strong>Legend:</strong>
            <span class="swatch" style="background:{green};"></span><small>supports predicted class</small>
            <span class="swatch" style="background:{red};"></span><small>opposes predicted class</small>
          </div>
        </div>
        """
    
        # Table of top attributions
        top_attributions = (
            pd.DataFrame({"Word": words, "Attribution": word_attributions})
            .assign(Abs=lambda df: df["Attribution"].abs())
            .sort_values(by="Abs", ascending=False)
            .drop(columns="Abs")
            .head(top_k_attributions)
        )
    
        return html_content, top_attributions
