import json
from collections import defaultdict
import numpy as np
import sacrebleu
from sacrebleu.metrics import CHRF  # ✅ chrF / chrF++
from comet import download_model, load_from_checkpoint


def group_docs(jsonl_path: str):
    src_map = defaultdict(list)
    mt_map = defaultdict(list)
    ref_map = defaultdict(list)

    used_doc_key = None
    has_doc_id = False

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            d = json.loads(line)

            src = d.get("source", None) or d.get("src", None)
            mt = d.get("generated", None)
            ref = d.get("reference", None)

            if src is None or mt is None or ref is None:
                raise ValueError(f"Line {i} missing one of [source/src, generated, reference].")

            doc_id = None
            for k in ["doc_id", "document_id", "docid", "document"]:
                if k in d:
                    has_doc_id = True
                    used_doc_key = k
                    doc_id = str(d[k])
                    break

            if doc_id is None:
                doc_id = str(i)  # each line treated as one document

            src_map[doc_id].append(src)
            mt_map[doc_id].append(mt)
            ref_map[doc_id].append(ref)

    doc_ids = sorted(src_map.keys())
    src_docs = [" ".join(src_map[x]) for x in doc_ids]
    mt_docs  = [" ".join(mt_map[x])  for x in doc_ids]
    ref_docs = [" ".join(ref_map[x]) for x in doc_ids]

    meta = {"has_doc_id": has_doc_id, "used_doc_key": used_doc_key, "num_docs": len(doc_ids)}
    return src_docs, mt_docs, ref_docs, meta


def compute_d_bleu(mt_docs, ref_docs):
    bleu = sacrebleu.corpus_bleu(mt_docs, [ref_docs])
    return bleu.score, bleu


def compute_d_chrf(mt_docs, ref_docs, word_order=2):
    """
    word_order=0 -> chrF
    word_order=2 -> chrF++ (recommended)
    """
    chrf = CHRF(word_order=word_order)
    score_obj = chrf.corpus_score(mt_docs, [ref_docs])
    return score_obj.score, score_obj

# document level
from multiprocessing import Pool
from functools import partial
import sacrebleu

def compute_single_ter(args):
    mt, ref = args
    ter_obj = sacrebleu.corpus_ter([mt], [[ref]])
    return ter_obj.score

def compute_d_ter(mt_docs, ref_docs, n_workers=8):
    from tqdm import tqdm
    
    print(f"[TER] Computing with {n_workers} workers...")
    
    with Pool(n_workers) as pool:
        ter_scores = list(tqdm(
            pool.imap(compute_single_ter, zip(mt_docs, ref_docs)),
            total=len(mt_docs)
        ))
    
    avg_ter = np.mean(ter_scores)
    print(f"[TER] Average: {avg_ter:.4f}")
    return avg_ter, ter_scores

def compute_d_comet(src_docs, mt_docs, ref_docs, model_name="Unbabel/wmt22-comet-da", batch_size=8):
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    comet_data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src_docs, mt_docs, ref_docs)]
    out = model.predict(
        comet_data,
        batch_size=batch_size,
        gpus=1
    )
    scores = out.scores
    return float(np.mean(scores)), scores


if __name__ == "__main__":
    input_file = "./wmt23_translation_deepseek.jsonl"
    print(input_file)
    src_docs, mt_docs, ref_docs, meta = group_docs(input_file)

    print("==== Data Grouping ====")
    if meta["has_doc_id"]:
        print(f"Grouped by {meta['used_doc_key']}, #docs={meta['num_docs']}")
    else:
        print(f"No doc_id -> each line is a document, #docs={meta['num_docs']}")

    d_bleu, bleu_obj = compute_d_bleu(mt_docs, ref_docs)
    print("\n==== WMT-style d-BLEU (doc-level sacreBLEU) ====")
    print(f"d-BLEU: {d_bleu:.4f}")
    print("Full:", bleu_obj)

    d_chrf, chrf_obj = compute_d_chrf(mt_docs, ref_docs, word_order=2) 
    print("\n==== WMT-style d-chrF++ (doc-level sacreBLEU) ====")
    print(f"d-chrF++: {d_chrf:.4f}")
    print("Full:", chrf_obj)

    d_ter, ter_obj = compute_d_ter(mt_docs, ref_docs)
    print("\n==== WMT-style d-TER (doc-level sacreBLEU) ====")
    print(f"d-TER: {d_ter:.4f}  (lower is better)")
    print("Full:", ter_obj)

    d_comet, comet_scores = compute_d_comet(src_docs, mt_docs, ref_docs)
    print("\n==== WMT-style d-COMET (doc-level COMET) ====")
    print(f"d-COMET: {d_comet:.6f}")