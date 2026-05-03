# -*- coding: utf-8 -*-
"""
Compute ER / IP / EX empathy scores for a JSONL dataset with:
  - data["text"]      -> seeker (client) full text (NOT chunked)
  - data["generated"] -> response (assistant) text (chunked to <=64 tokens)

New Strategy:
  1) seeker: keep full (model will truncate internally to 64 tokens)
  2) response: sentence split + token-aware chunking (<= 64 tokens)
  3) score each response chunk with 3 HF models (ER/IP/EX)
  4) SUM over chunks -> per-sample score (NOT average)
  5) dataset mean is computed over per-sample SUM scores

Dependencies:
  pip install torch transformers pandas tqdm
"""

import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig


# -----------------------------
# Sentence split + token chunk
# -----------------------------
_SENT_SPLIT_REGEX = re.compile(r"(?<=[\.\!\?\。\！\？])\s+|[\r\n]+")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using common punctuation."""
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    sents = [s.strip() for s in _SENT_SPLIT_REGEX.split(text) if s.strip()]
    return sents


def chunk_sentences_by_tokens(
    sentences: List[str],
    tokenizer,
    max_tokens: int = 64,
) -> List[str]:
    """
    Merge consecutive sentences into chunks so that token_len(chunk) <= max_tokens.
    If a single sentence is too long: truncate by tokens.
    """
    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    chunks: List[str] = []
    cur: List[str] = []

    for sent in sentences:
        if not sent:
            continue

        # If a single sentence is too long, truncate
        if token_len(sent) > max_tokens:
            ids = tokenizer.encode(sent, add_special_tokens=False)[:max_tokens]
            sent = tokenizer.decode(ids, skip_special_tokens=True).strip()

        if not cur:
            cur = [sent]
        else:
            candidate = " ".join(cur + [sent])
            if token_len(candidate) <= max_tokens:
                cur.append(sent)
            else:
                chunks.append(" ".join(cur))
                cur = [sent]

    if cur:
        chunks.append(" ".join(cur))

    if not chunks:
        chunks = [""]

    return chunks


# -----------------------------
# HF Empathy models wrapper
# -----------------------------
@dataclass
class EmpathyModels:
    er_name: str = "RyanDDD/empathy-mental-health-reddit-ER"
    ip_name: str = "RyanDDD/empathy-mental-health-reddit-IP"
    ex_name: str = "RyanDDD/empathy-mental-health-reddit-EX"
    max_tokens: int = 64


class EmpathyScorer:
    """
    Load ER/IP/EX models once, then score texts efficiently.
    """

    def __init__(self, model_cfg: EmpathyModels, device: Optional[str] = None):
        self.cfg = model_cfg
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self._load_all_models()

    def _load_one(self, name: str):
        tokenizer = AutoTokenizer.from_pretrained(name)
        _ = AutoConfig.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True).to(self.device)
        model.eval()
        return model, tokenizer

    def _load_all_models(self):
        print(f"[Info] Loading empathy models on {self.device} ...")

        er, er_tok = self._load_one(self.cfg.er_name)
        ip, ip_tok = self._load_one(self.cfg.ip_name)
        ex, ex_tok = self._load_one(self.cfg.ex_name)

        self.models["ER"], self.tokenizers["ER"] = er, er_tok
        self.models["IP"], self.tokenizers["IP"] = ip, ip_tok
        self.models["EX"], self.tokenizers["EX"] = ex, ex_tok

        print("[Info] Models loaded: ER / IP / EX")

    @torch.no_grad()
    def _predict_level(self, dim: str, seeker_post: str, response_post: str) -> int:
        """
        Robust inference:
          - If model has predict(): use it
          - Else: use forward() with tokenized inputs (most stable)
        """
        model = self.models[dim]
        tokenizer = self.tokenizers[dim]

        # 1) try predict()
        if hasattr(model, "predict"):
            pred, _ = model.predict(
                seeker_post=seeker_post,
                response_post=response_post,
                tokenizer=tokenizer,
                device=self.device,
            )
            return int(pred)

        # 2) fallback to forward()
        encoded_sp = tokenizer(
            seeker_post,
            max_length=self.cfg.max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_rp = tokenizer(
            response_post,
            max_length=self.cfg.max_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoded_sp = {k: v.to(self.device) for k, v in encoded_sp.items()}
        encoded_rp = {k: v.to(self.device) for k, v in encoded_rp.items()}

        outputs = model(
            input_ids_SP=encoded_sp["input_ids"],
            input_ids_RP=encoded_rp["input_ids"],
            attention_mask_SP=encoded_sp["attention_mask"],
            attention_mask_RP=encoded_rp["attention_mask"],
        )

        logits = outputs[0]  # (batch, 3)
        pred = torch.argmax(logits, dim=1).item()
        return int(pred)

    def score_pair_response_chunk_sum(self, seeker_text: str, response_text: str) -> Dict[str, float]:
        """
        NEW metric aggregation (your request):
          - seeker_text: FULL text, no chunking
          - response_text: chunked (<=64 tokens)
          - per-sample score = SUM of chunk-level predictions (NOT average)
        """
        seeker_text = "" if seeker_text is None else str(seeker_text)
        response_text = "" if response_text is None else str(response_text)

        chunk_tok = self.tokenizers["IP"]
        resp_sents = split_into_sentences(response_text)
        resp_chunks = chunk_sentences_by_tokens(resp_sents, chunk_tok, self.cfg.max_tokens)

        scores: Dict[str, float] = {}
        for dim in ["ER", "IP", "EX"]:
            per_chunk = [self._predict_level(dim, seeker_text, rc) for rc in resp_chunks]
            scores[dim] = float(sum(per_chunk))  # ✅ SUM (no division)

        scores["TOTAL"] = scores["ER"] + scores["IP"] + scores["EX"]
        scores["NUM_CHUNKS"] = float(len(resp_chunks))  # optional diagnostic
        return scores


# -----------------------------
# Dataset runner
# -----------------------------
def read_jsonl(path: str, limit: int = -1) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="empathy_scores_sum.csv")
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--seeker_key", type=str, default="text", help="Key for seeker/client text")
    parser.add_argument("--generated_key", type=str, default="generated", help="Key for generated response")

    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    cfg = EmpathyModels(max_tokens=args.max_tokens)
    scorer = EmpathyScorer(cfg)

    data = read_jsonl(args.input_path, limit=args.limit)
    print(f"[Info] Loaded {len(data)} samples from {args.input_path}")

    rows = []

    for idx, item in tqdm(list(enumerate(data)), total=len(data)):
        seeker_text = str(item.get(args.seeker_key, ""))
        gen_text = str(item.get(args.generated_key, ""))

        scores = scorer.score_pair_response_chunk_sum(
            seeker_text=seeker_text,
            response_text=gen_text
        )

        rows.append({
            "idx": idx,
            "ER_sum": scores["ER"],
            "IP_sum": scores["IP"],
            "EX_sum": scores["EX"],
            "TOTAL_sum": scores["TOTAL"],
            "num_chunks": scores["NUM_CHUNKS"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] per-sample sum scores -> {args.output_csv}")

    print("\n======= Dataset-level Mean of SUM Scores =======")
    print(f"ER_sum_mean        = {df['ER_sum'].mean():.4f}")
    print(f"IP_sum_mean        = {df['IP_sum'].mean():.4f}")
    print(f"EX_sum_mean        = {df['EX_sum'].mean():.4f}")
    print(f"TOTAL_sum_mean     = {df['TOTAL_sum'].mean():.4f}")
    print(f"avg_num_chunks     = {df['num_chunks'].mean():.4f}")


if __name__ == "__main__":
    main()
