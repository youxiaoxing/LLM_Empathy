import spacy
from typing import Dict
import nltk
from nltk.corpus import wordnet as wn

# First run requires downloading WordNet
try:
    wn.ensure_loaded()
except:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class LinguisticDistancingAnalyzer:
    """
    Linguistic Distancing Analyzer - Simplified Version
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found, downloading...")
            import os
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        
        # Nominalization suffixes
        self.nominal_suffixes = ("tion", "sion", "ment", "ance", "ence", "ity", "ness", "al")
        
        # Only keep basic personal pronouns (for quick judgment)
        self.human_pronouns = {
            "i", "you", "he", "she", "we", "they",
            "me", "him", "her", "us", "them",
            "who", "whom"
        }
        
        # WordNet cache
        self._wordnet_cache = {}
    
    def is_human_wordnet(self, word: str) -> bool:
        """
        Use WordNet to directly determine if a word is human-related
        
        Core logic: Check if the word's semantic hierarchy contains person (human)
        """
        # Check cache
        if word in self._wordnet_cache:
            return self._wordnet_cache[word]
        
        # Get all noun synsets for this word
        synsets = wn.synsets(word, pos=wn.NOUN)
        
        if not synsets:
            self._wordnet_cache[word] = False
            return False
        
        # Check each synset
        for synset in synsets:
            # Get all paths from current word to root node
            for path in synset.hypernym_paths():
                # Check each concept in the path
                for hypernym in path:
                    # person.n.01 is the root concept for "person" in WordNet
                    if hypernym.name() == 'person.n.01':
                        self._wordnet_cache[word] = True
                        return True
        
        self._wordnet_cache[word] = False
        return False
    
    def is_agentless_passive(self, sent) -> bool:
        """Determine if sentence contains agentless passive voice"""
        for token in sent:
            if token.dep_ == "nsubjpass":
                verb = token.head
                has_agent = any(child.dep_ == "agent" for child in verb.children)
                if not has_agent:
                    return True
        return False
    
    def count_nominalizations(self, doc) -> int:
        """Count number of nominalized nouns"""
        nominal_count = 0
        for token in doc:
            if token.pos_ == "NOUN" and token.text.lower().endswith(self.nominal_suffixes):
                if token.ent_type_ not in ("PERSON", "ORG", "GPE"):
                    nominal_count += 1
        return nominal_count
    
    def is_impersonal_subject(self, token) -> bool:
        """
        Determine if subject is impersonal (simplified version)
        
        Logic:
        1. Personal pronoun → is human
        2. PERSON entity → is human
        3. WordNet check → is human-related word
        """
        # Must be a subject
        if token.dep_ not in ("nsubj", "nsubjpass"):
            return False
        
        # 1. Personal pronouns
        if token.pos_ == "PRON" and token.lower_ in self.human_pronouns:
            return False  # is human
        
        # 2. Named entity (person name)
        if token.ent_type_ == "PERSON":
            return False  # is human
        
        # 3. Use WordNet check (only for nouns)
        if token.pos_ in ("NOUN", "PROPN"):
            if self.is_human_wordnet(token.lemma_.lower()):
                return False  # is human
        
        # None satisfied → impersonal subject
        return True
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Perform complete linguistic distancing analysis on text"""
        doc = self.nlp(text)
        
        sentences = list(doc.sents)
        n_sentences = len(sentences)
        if n_sentences == 0:
            return {
                "agentless_passive_density": 0.0,
                "nominalization_density": 0.0,
                "impersonal_subject_ratio": 0.0
            }
        
        # 1. Agentless passive voice density
        agentless_count = sum(1 for sent in sentences if self.is_agentless_passive(sent))
        agentless_density = agentless_count / n_sentences
        
        # 2. Nominalization density
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        nominal_count = self.count_nominalizations(doc)
        nominalization_density = nominal_count / total_words if total_words > 0 else 0.0
        
        # 3. Impersonal subject ratio
        subjects = [tok for tok in doc if tok.dep_ in ("nsubj", "nsubjpass")]
        if len(subjects) > 0:
            impersonal_count = sum(1 for tok in subjects if self.is_impersonal_subject(tok))
            impersonal_ratio = impersonal_count / len(subjects)
        else:
            impersonal_ratio = 0.0
        
        return {
            "agentless_passive_density": agentless_density,
            "nominalization_density": nominalization_density,
            "impersonal_subject_ratio": impersonal_ratio,
            "n_sentences": n_sentences,
            "n_agentless_passive": agentless_count,
            "n_nominalizations": nominal_count,
            "n_subjects": len(subjects),
            "n_impersonal_subjects": sum(1 for tok in subjects if self.is_impersonal_subject(tok))
        }
    
    def analyze_subjects_detail(self, text: str):
        """Analyze all subjects in detail (for debugging)"""
        doc = self.nlp(text)
        subjects = [tok for tok in doc if tok.dep_ in ("nsubj", "nsubjpass")]
        
        results = []
        for token in subjects:
            is_human_wn = self.is_human_wordnet(token.lemma_.lower()) if token.pos_ in ("NOUN", "PROPN") else False
            
            info = {
                "text": token.text,
                "lemma": token.lemma_,
                "is_pronoun": token.pos_ == "PRON" and token.lower_ in self.human_pronouns,
                "is_person_ner": token.ent_type_ == "PERSON",
                "is_human_wordnet": is_human_wn,
                "is_impersonal": self.is_impersonal_subject(token),
                "sentence": token.sent.text.strip()
            }
            results.append(info)
        
        return results
    
    def compare_texts(self, original_text: str, generated_text: str) -> Dict:
        """Compare linguistic distancing differences between original and generated text"""
        original_scores = self.analyze(original_text)
        generated_scores = self.analyze(generated_text)
        
        differences = {
            "agentless_passive_diff": generated_scores["agentless_passive_density"] - original_scores["agentless_passive_density"],
            "nominalization_diff": generated_scores["nominalization_density"] - original_scores["nominalization_density"],
            "impersonal_subject_diff": generated_scores["impersonal_subject_ratio"] - original_scores["impersonal_subject_ratio"]
        }
        
        return {
            "original": original_scores,
            "generated": generated_scores,
            "differences": differences
        }

import json
import numpy as np
from scipy.spatial.distance import euclidean

if __name__ == "__main__":
    analyzer = LinguisticDistancingAnalyzer()
    
    input_file = "INPUT_FILE_PATH.jsonl"
    
    ref_vectors = []
    gen_vectors = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            data = json.loads(line)
            
            ref_result = analyzer.analyze(data["reference"])
            gen_result = analyzer.analyze(data["generate"])
            
            ref_vector = [
                ref_result["agentless_passive_density"],
                ref_result["nominalization_density"],
                ref_result["impersonal_subject_ratio"]
            ]
            
            gen_vector = [
                gen_result["agentless_passive_density"],
                gen_result["nominalization_density"],
                gen_result["impersonal_subject_ratio"]
            ]
            
            ref_vectors.append(ref_vector)
            gen_vectors.append(gen_vector)
    
    ref_vectors = np.array(ref_vectors)
    gen_vectors = np.array(gen_vectors)
    
    # Normalize using Reference mean and standard deviation
    ref_mean = ref_vectors.mean(axis=0)
    ref_std = ref_vectors.std(axis=0)
    
    ref_vectors_z = (ref_vectors - ref_mean) / ref_std
    gen_vectors_z = (gen_vectors - ref_mean) / ref_std
    
    # Calculate Euclidean distance for each pair
    distances = []
    for ref, gen in zip(ref_vectors_z, gen_vectors_z):
        distances.append(euclidean(ref, gen))
    distances = np.array(distances)
    
    # Output statistics
    print(f"Average distance: {distances.mean():.4f}")
    print(f"Standard deviation: {distances.std():.4f}")
    print(f"Median: {np.median(distances):.4f}")