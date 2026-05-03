import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import re
import numpy as np
import types
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from rouge_score import rouge_scorer
from unidecode import unidecode
from bert_score import score

def process_string(text):
    text = text.replace("\n", " ")
    text = text.strip()
    text = unidecode(text)
    return text

def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()

def cal_caption_score_from_dict(result_dict, use_bert_score=False):
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer_old = Rouge()
    rouge_scorer_new = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
    rouge_scores = []
    rouge_lsum_scores = []
    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)

    eval_line = 'EVAL'
    meteor_scorer.lock.acquire()
    count = 0
    meteor_scores = []

    candidates = []
    references = []

    for sample in tqdm(result_dict, desc="Processing"):
        caption = re.sub(r'[^\w\s]', '', sample["reference"])
        generation = re.sub(r'[^\w\s]', '', sample["generate"])

        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer_old.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        
        rouge_score_new = rouge_scorer_new.score(sample["reference"], sample["generate"])
        rouge_lsum_scores.append(rouge_score_new['rougeLsum'].fmeasure)
        
        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += ' ||| {}'.format(stat)
        count += 1
        
        candidates.append(sample["generate"])
        references.append(sample["reference"])

    meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
    meteor_scorer.meteor_p.stdin.flush()
    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
    meteor_scorer.lock.release()

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    rouge_lsum_score = np.mean(np.array(rouge_lsum_scores))
    cider_score, _ = cider_scorer.compute_score()
    
    if use_bert_score:
        P, R, F1 = score(candidates, references, lang="en", verbose=False)
        bert_score = F1.mean().item()
    else:
        bert_score = 0.0

    return {
        'bleu': blue_score,
        'rouge': rouge_score,
        'rouge_lsum': rouge_lsum_score,
        'cider': cider_score,
        'meteor': meteor_score,
        'bert_score': bert_score
    }

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                item = json.loads(line)
                data.append(item)
    return data

def get_data_file(file_path):

    data = load_jsonl_file(file_path)
    original_count = len(data)


    for item in data:
        if "</s>" in item["generate"]:
            item["generate"] = process_string(item["generate"].split("</s>")[0])
        else:
            item["generate"] = process_string(item["generate"])
        
        item["reference"] = process_string(item["reference"].replace("</s>", ""))
    
    return data

def print_results(scores, file_name):
    print("\n" + "="*120)
    print("="*120 + "\n")
    print("-"*120)
    print(f"  BLEU-1: {scores['bleu'][0]:.4f}")
    print(f"  BLEU-2: {scores['bleu'][1]:.4f}")
    print(f"  BLEU-3: {scores['bleu'][2]:.4f}")
    print(f"  BLEU-4: {scores['bleu'][3]:.4f}")
    print(f"  ROUGE:  {scores['rouge']:.4f}")
    print(f"  ROUGE-LSum: {scores['rouge_lsum']:.4f}")
    print(f"  CIDEr:  {scores['cider']:.4f}")
    print(f"  METEOR: {scores['meteor']:.4f}")
    print(f"  BERTScore: {scores['bert_score']:.4f}")
    print()
    print("="*120 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str, required=True)
    parser.add_argument('--use_bert_score', '-b', action='store_true')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"{args.input_file}")
        return
    
    if not os.path.isfile(args.input_file):
        print(f"{args.input_file}")
        return
    
    if not args.input_file.endswith('.jsonl'):
        print(f"{args.input_file}")
    

    data = get_data_file(args.input_file)
    scores = cal_caption_score_from_dict(data, use_bert_score=args.use_bert_score)
    file_name = os.path.basename(args.input_file)
    print_results(scores, file_name)

if __name__ == "__main__":
    main()