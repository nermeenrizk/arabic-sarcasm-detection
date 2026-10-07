"""
Arabic sarcasm detection on ArSarcasm-v2: feature injection into MARBERT.

Three configurations are compared under an identical training protocol:

1. Baseline            MARBERT [CLS] representation -> classification head.
2. Hybrid-Surface      MARBERT + general surface features (length, counts of
                       punctuation and digits) + one-hot sentiment and dialect,
                       fused by simple concatenation.
3. Hybrid-Incongruity  MARBERT + features motivated by the emotional-attitude
                       component of the implicit display theory of irony
                       (emphasis markers, ellipses, lexicon polarity, and a
                       lexical-polarity vs. annotated-sentiment divergence flag)
                       + one-hot sentiment and dialect, fused through a gate.

Shared protocol: stratified 10% validation split, AdamW (lr 2e-5), batch size 16,
at most 6 epochs with early stopping on validation macro-F1 (patience 2), and a
class-weighted cross-entropy loss. Each configuration is trained with three seeds;
the test set is evaluated once per run, using the best validation checkpoint.

Note: the sentiment and dialect columns are gold annotations provided with
ArSarcasm-v2, so the hybrid configurations assume they are available at test time.

Results are written after every run, so an interrupted session can be resumed.
"""

import os
import random
import re
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# Working directory (Kaggle if available, otherwise the current directory)
# ----------------------------------------------------------------------------
BASE_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.chdir(BASE_DIR)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEEDS = [42, 123, 2024]
EPOCHS = 6
PATIENCE = 2
BATCH_SIZE = 16
MAX_LEN = 128
LR = 2e-5
VAL_FRACTION = 0.1
MODEL_NAME = "UBC-NLP/MARBERT"
GATED_PROJ_DIM = 128  # dimension of the projected feature vector in gated fusion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
DATA_DIR = "ArSarcasm-v2"
if not os.path.exists(DATA_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/iabufarha/ArSarcasm-v2.git"],
        check=True,
    )

train_df_full = pd.read_csv("ArSarcasm-v2/ArSarcasm-v2/training_data.csv")
test_df = pd.read_csv("ArSarcasm-v2/ArSarcasm-v2/testing_data.csv")

train_df, val_df = train_test_split(
    train_df_full,
    test_size=VAL_FRACTION,
    random_state=42,
    stratify=train_df_full["sarcasm"],
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ----------------------------------------------------------------------------
# Class weights, computed once from the training split and shared by all models
# ----------------------------------------------------------------------------
class_counts = train_df["sarcasm"].value_counts().sort_index()
n_samples = len(train_df)
n_classes = len(class_counts)
class_weights = torch.tensor(
    [n_samples / (n_classes * class_counts[i]) for i in range(n_classes)],
    dtype=torch.float,
).to(DEVICE)
print(f"Class weights (0 = non-sarcastic, 1 = sarcastic): {class_weights.tolist()}")

# ----------------------------------------------------------------------------
# Small Arabic polarity lexicon (MSA and common dialectal forms).
# Deliberately minimal; its coverage is a limitation of this study.
# ----------------------------------------------------------------------------
POSITIVE_WORDS = {
    "حلو", "جميل", "رهيب", "تحفة", "عظيم", "ممتاز", "رائع", "جامد", "حبيبي",
    "مبروك", "الحمدلله", "الحمدلله", "احسن", "أحسن", "كويس", "نايس", "حبيب",
    "سعيد", "سعيدة", "فرحان", "فرحانة", "بجنن", "خرافي", "زي الفل", "مية مية",
    "ذكي", "شاطر", "شاطرة", "بطل", "بطلة", "نجاح", "ناجح", "فخور", "متحمس",
}
NEGATIVE_WORDS = {
    "وحش", "زفت", "فاشل", "فاشلة", "تعبان", "تعبانة", "سيء", "سيئة", "غبي",
    "غبية", "كارثة", "بايظ", "بايظة", "زبالة", "قرف", "مقرف", "حزين", "حزينة",
    "فشل", "خايب", "خايبة", "معفن", "تافه", "تافهة", "مزعج", "مزعجة", "غلط",
    "ظلم", "مصيبة", "نكد", "زهقان", "زهقانة",
}
SENTIMENT_LABEL_SIGN = {"positive": 1, "neutral": 0, "negative": -1}


def lexicon_polarity_score(text):
    words = re.findall(r"[\u0600-\u06FF]+", str(text))
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return pos - neg


def sentiment_incongruity_flag(text, sentiment_label):
    """1 if lexical polarity and annotated sentiment point in opposite directions, else 0."""
    score = lexicon_polarity_score(text)
    label_sign = SENTIMENT_LABEL_SIGN.get(str(sentiment_label).lower(), 0)
    if score > 0 and label_sign < 0:
        return 1
    if score < 0 and label_sign > 0:
        return 1
    return 0


# Emoji incongruity is only used if the training data actually contains emoji.
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
emoji_hits = train_df["tweet"].astype(str).apply(lambda t: len(EMOJI_PATTERN.findall(t))).sum()
HAS_EMOJI = emoji_hits > 0
print(f"Emoji found in training data: {emoji_hits} (emoji feature {'enabled' if HAS_EMOJI else 'disabled'})")

POSITIVE_EMOJI = set("😀😃😄😁😊🙂😍❤️👍🔥😂🤣✨")
NEGATIVE_EMOJI = set("😡😠😢😭👎💔😞😔")


def emoji_incongruity_flag(text, sentiment_label):
    if not HAS_EMOJI:
        return 0
    chars = list(str(text))
    pos = sum(1 for c in chars if c in POSITIVE_EMOJI)
    neg = sum(1 for c in chars if c in NEGATIVE_EMOJI)
    label_sign = SENTIMENT_LABEL_SIGN.get(str(sentiment_label).lower(), 0)
    if pos > neg and label_sign < 0:
        return 1
    if neg > pos and label_sign > 0:
        return 1
    return 0


# ----------------------------------------------------------------------------
# Feature sets. Scalers and encoders are fitted on the training split only.
# ----------------------------------------------------------------------------
def raw_surface_numeric(df):
    """General surface features used by Hybrid-Surface."""
    tweets = df["tweet"].astype(str)
    feats = pd.DataFrame({
        "length": tweets.str.len(),
        "word_count": tweets.str.split().str.len(),
        "exclam_count": tweets.str.count("!"),
        "question_count": tweets.str.count(r"\?"),
        "punct_count": tweets.apply(lambda t: sum(1 for c in t if c in "،,.-!؟?")),
        "digit_count": tweets.apply(lambda t: sum(1 for c in t if c.isdigit())),
    })
    return feats.values.astype(float)


def raw_incongruity_numeric(df):
    """Emphasis and polarity features used by Hybrid-Incongruity."""
    tweets = df["tweet"].astype(str)
    feats = pd.DataFrame({
        "exclam_count": tweets.str.count("!"),
        "question_count": tweets.str.count(r"\?"),
        "punct_count": tweets.apply(lambda t: sum(1 for c in t if c in "،,.-!؟?")),
        "ellipsis_count": tweets.apply(lambda t: len(re.findall(r"\.{2,}|…", t))),
        "lexicon_polarity": tweets.apply(lexicon_polarity_score),
    })
    return feats.values.astype(float)


surface_scaler = StandardScaler().fit(raw_surface_numeric(train_df))
incong_scaler = StandardScaler().fit(raw_incongruity_numeric(train_df))
cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
    train_df[["sentiment", "dialect"]].astype(str)
)


def build_surface_features(df):
    numeric = surface_scaler.transform(raw_surface_numeric(df))
    categorical = cat_encoder.transform(df[["sentiment", "dialect"]].astype(str))
    return np.concatenate([numeric, categorical], axis=1).astype(np.float32)


def build_incongruity_features(df):
    numeric = incong_scaler.transform(raw_incongruity_numeric(df))
    categorical = cat_encoder.transform(df[["sentiment", "dialect"]].astype(str))
    sent_incong = df.apply(
        lambda r: sentiment_incongruity_flag(r["tweet"], r["sentiment"]), axis=1
    ).values.reshape(-1, 1).astype(np.float32)
    emoji_incong = df.apply(
        lambda r: emoji_incongruity_flag(r["tweet"], r["sentiment"]), axis=1
    ).values.reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [numeric, categorical, sent_incong, emoji_incong], axis=1
    ).astype(np.float32)


surface_features = {
    "train": build_surface_features(train_df),
    "val": build_surface_features(val_df),
    "test": build_surface_features(test_df),
}
incong_features = {
    "train": build_incongruity_features(train_df),
    "val": build_incongruity_features(val_df),
    "test": build_incongruity_features(test_df),
}
NUM_SURFACE = surface_features["train"].shape[1]
NUM_INCONG = incong_features["train"].shape[1]
print(f"Surface features: {NUM_SURFACE} dims | Incongruity features: {NUM_INCONG} dims")

# ----------------------------------------------------------------------------
# Datasets and loaders (the baseline receives a dummy feature column it ignores)
# ----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SarcasmDataset(TorchDataset):
    def __init__(self, texts, features, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.features = features
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        enc = self.tokenizer(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "features": torch.tensor(self.features[item], dtype=torch.float),
            "labels": torch.tensor(int(self.labels[item]), dtype=torch.long),
        }


def make_loader(df, features, shuffle):
    ds = SarcasmDataset(df["tweet"].values, features, df["sarcasm"].values, tokenizer)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


zeros_train = np.zeros((len(train_df), 1), dtype=np.float32)
zeros_val = np.zeros((len(val_df), 1), dtype=np.float32)
zeros_test = np.zeros((len(test_df), 1), dtype=np.float32)

LOADERS = {
    "baseline": {
        "train": make_loader(train_df, zeros_train, True),
        "val": make_loader(val_df, zeros_val, False),
        "test": make_loader(test_df, zeros_test, False),
    },
    "hybrid_surface": {
        "train": make_loader(train_df, surface_features["train"], True),
        "val": make_loader(val_df, surface_features["val"], False),
        "test": make_loader(test_df, surface_features["test"], False),
    },
    "hybrid_incongruity": {
        "train": make_loader(train_df, incong_features["train"], True),
        "val": make_loader(val_df, incong_features["val"], False),
        "test": make_loader(test_df, incong_features["test"], False),
    },
}
MODEL_CONFIGS = {
    "baseline": {"mode": "baseline", "num_features": 0},
    "hybrid_surface": {"mode": "concat", "num_features": NUM_SURFACE},
    "hybrid_incongruity": {"mode": "gated", "num_features": NUM_INCONG},
}


# ----------------------------------------------------------------------------
# Models: baseline, concatenation, gated fusion
# ----------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Projects the feature vector and learns a per-example gate that controls
    how much of it is passed to the classifier alongside the text representation."""

    def __init__(self, text_dim, feat_dim, proj_dim=GATED_PROJ_DIM):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, proj_dim)
        self.gate = nn.Linear(text_dim + proj_dim, proj_dim)
        self.out_dim = text_dim + proj_dim

    def forward(self, text_repr, feat):
        feat_p = torch.tanh(self.feat_proj(feat))
        gate_input = torch.cat([text_repr, feat_p], dim=1)
        g = torch.sigmoid(self.gate(gate_input))
        gated_feat = g * feat_p
        return torch.cat([text_repr, gated_feat], dim=1)


class SarcasmClassifier(nn.Module):
    def __init__(self, model_name, mode="baseline", num_features=0, num_classes=2):
        super().__init__()
        self.mode = mode
        self.marbert = AutoModel.from_pretrained(model_name)
        hidden_size = self.marbert.config.hidden_size

        if mode == "baseline":
            classifier_input = hidden_size
        elif mode == "concat":
            classifier_input = hidden_size + num_features
        elif mode == "gated":
            self.fusion = GatedFusion(hidden_size, num_features)
            classifier_input = self.fusion.out_dim
        else:
            raise ValueError(f"unknown mode {mode}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, features=None):
        outputs = self.marbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        if self.mode == "baseline":
            rep = cls_output
        elif self.mode == "concat":
            rep = torch.cat([cls_output, features], dim=1)
        elif self.mode == "gated":
            rep = self.fusion(cls_output, features)
        return self.classifier(rep)


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            features = batch["features"].to(DEVICE)
            outputs = model(input_ids, attention_mask, features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "sarcasm_f1": f1_score(all_labels, all_preds, pos_label=1, average="binary"),
    }


def train_one_run(model_key, seed):
    cfg = MODEL_CONFIGS[model_key]
    loaders = LOADERS[model_key]
    set_seed(seed)
    model = SarcasmClassifier(
        MODEL_NAME, mode=cfg["mode"], num_features=cfg["num_features"]
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(loaders["train"], desc=f"[{model_key} seed={seed}] Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            features = batch["features"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, loaders["val"])
        print(f"  epoch {epoch+1}: val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"val_sarcasm_f1={val_metrics['sarcasm_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"])
    test_metrics["best_val_macro_f1"] = best_val_f1
    return test_metrics


# ----------------------------------------------------------------------------
# Full experiment: 3 configurations x 3 seeds, saved after every run
# ----------------------------------------------------------------------------
RESULTS_PATH = "results_raw.csv"

if os.path.exists(RESULTS_PATH):
    results = pd.read_csv(RESULTS_PATH).to_dict("records")
    done_runs = {(r["model"], r["seed"]) for r in results}
    print(f"Resuming: {len(results)} completed runs found")
else:
    results = []
    done_runs = set()

for model_key in ["baseline", "hybrid_surface", "hybrid_incongruity"]:
    for seed in SEEDS:
        if (model_key, seed) in done_runs:
            print(f"== {model_key} | seed={seed} | already done, skipping ==")
            continue

        metrics = train_one_run(model_key, seed)
        metrics["model"] = model_key
        metrics["seed"] = seed
        results.append(metrics)
        print(f"== {model_key} | seed={seed} | TEST: {metrics} ==\n")

        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

results_df = pd.DataFrame(results)
summary = (
    results_df.groupby("model")[["accuracy", "macro_f1", "sarcasm_f1"]]
    .agg(["mean", "std"])
)
summary.to_csv("results_summary.csv")

print("\n=== Test-set results, mean and std over 3 seeds ===")
print(summary)
