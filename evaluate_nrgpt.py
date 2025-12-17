#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import wandb

from spellchecker import SpellChecker
import textstat
import re
import re
from collections import defaultdict
# import mauve

# Initialize tools (do this once, outside your main function)
spell = SpellChecker()

# print("wandb version:", wandb.__version__)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util

REFERENCE_TEXT_PATH = "data/shakespeare_char/input.txt" # Path to your reference text file for MAUVE

# -----------------------------
# Metrics helpers (unchanged)
def distinct_n(sentences, n):
    total_ngrams = 0
    unique_ngrams = set()
    for sent in sentences:
        tokens = sent.split()
        ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

_gpt2_tok = None
_gpt2_model = None
def compute_perplexity(sentences, model_name="gpt2"):
    global _gpt2_tok, _gpt2_model
    if _gpt2_tok is None:
        _gpt2_tok = GPT2Tokenizer.from_pretrained(model_name)
        _gpt2_model = GPT2LMHeadModel.from_pretrained(model_name).eval()
    scores = []
    for s in sentences:
        # ids = _gpt2_tok.encode(s, return_tensors="pt")
        # Get the maximum length from the model's configuration
        max_length = _gpt2_model.config.n_positions

        ids = _gpt2_tok.encode(s, return_tensors="pt")
        # Truncate the input tensor to the model's maximum length
        if ids.shape[1] > max_length:
            ids = ids[:, :max_length]
        # -----------------------

        with torch.no_grad():
            loss = _gpt2_model(ids, labels=ids).loss
        scores.append(torch.exp(loss).item())
    return float(np.mean(scores)) if scores else float("nan")

_sbert = SentenceTransformer("all-MiniLM-L6-v2")
def avg_pairwise_cosine(lines):
    if not lines:
        return 0.0
    emb = _sbert.encode(lines, convert_to_tensor=True)
    sims = []
    for i in range(len(emb)):
        for j in range(i+1, len(emb)):
            sims.append(util.pytorch_cos_sim(emb[i], emb[j]).item())
    return float(np.mean(sims)) if sims else 0.0


# grammatical errors
def robust_grammar_rules(text):
    """Comprehensive grammar error detection using rules"""
    errors = []
    error_details = defaultdict(list)
    
    # Preprocessing - handle edge cases
    if not text or len(text.strip()) < 3:
        return errors
    
    # Clean text for analysis
    clean_text = re.sub(r'\s+', ' ', text.strip())
    
    # 1. SUBJECT-VERB AGREEMENT
    sv_patterns = [
        (r'\b(he|she|it)\s+(are|were)\b', 'Subject-verb disagreement: singular subject with plural verb'),
        (r'\b(they|we|you)\s+(is|was)\b', 'Subject-verb disagreement: plural subject with singular verb'),
        (r'\bthere\s+is\s+\w*s\b', 'There is + plural noun'),
        (r'\bthere\s+are\s+\w*[^s]\b', 'There are + singular noun'),
    ]
    
    # 2. ARTICLE ERRORS
    article_patterns = [
        (r'\ba\s+[aeiouAEIOU]', 'Use "an" before vowel sounds'),
        (r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', 'Use "a" before consonant sounds'),
        (r'\b(a|an)\s+\d+', 'Article before number (usually unnecessary)'),
        (r'\bthe\s+\d+(st|nd|rd|th)', 'Article before ordinal numbers'),
    ]
    
    # 3. PRONOUN ERRORS
    pronoun_patterns = [
        (r'\bmyself\s+and\s+\w+', 'Use "I" instead of "myself" in compound subjects'),
        (r'\b(him|her)\s+and\s+I\b', 'Should be "he/she and I" in subject position'),
        (r'\bbetween\s+you\s+and\s+I\b', 'Should be "between you and me"'),
        (r'\bits\s+\w+ing\b(?!\s+its)', 'Possible "it\'s" contraction needed'),
        (r'\byour\s+(going|coming|doing|being)\b', 'Should be "you\'re"'),
        (r'\bthier\b', 'Misspelling: should be "their"'),
        (r'\bwhos\b(?!\s)', 'Should be "who\'s" (contraction) or "whose" (possessive)'),
    ]
    
    # 4. TENSE CONSISTENCY
    tense_patterns = [
        (r'\bwill\s+went\b', 'Tense mixing: will + past tense'),
        (r'\bdid\s+\w+ed\b', 'Double past tense: did + past participle'),
        (r'\bhave\s+\w+ing\b', 'Have + present participle (should be past participle)'),
    ]
    
    # 5. CAPITALIZATION
    cap_patterns = [
        (r'(?:^|\.\s+)[a-z]', 'Sentence should start with capital letter'),
        (r'\bi\s+', 'Lowercase "I" should be capitalized'),
        (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'Days should be capitalized'),
        (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'Months should be capitalized'),
    ]
        
    # 6. PREPOSITION ERRORS
    prep_patterns = [
        (r'\bdifferent\s+than\b', 'Use "different from" instead of "different than"'),
        (r'\boff\s+of\b', 'Use "off" instead of "off of"'),
        (r'\bcould\s+of\b', 'Should be "could have" or "could\'ve"'),
        (r'\bshould\s+of\b', 'Should be "should have" or "should\'ve"'),
        (r'\bwould\s+of\b', 'Should be "would have" or "would\'ve"'),
    ]
    
    # 7. WORD CHOICE ERRORS
    word_choice_patterns = [
        (r'\baffect\b(?=.*\bas\s+a\s+noun)', 'Use "effect" as noun, "affect" as verb'),
        (r'\bthen\b(?=.*comparison)', 'Use "than" for comparisons'),
        (r'\bloose\b(?=.*\bweight\b)', 'Should be "lose weight"'),
        (r'\baccept\b(?=.*\bfor\b)', 'Should be "except" when meaning "excluding"'),
    ]
    
    # 8. PUNCTUATION ERRORS
    punct_patterns = [
        (r'[a-z]\.[A-Z]', 'Missing space after period'),
        (r'\s+,', 'Space before comma'),
        (r',,+', 'Multiple consecutive commas'),
        (r'\.\.+(?!\.)', 'Multiple periods (use ellipsis ... if intentional)'),
        (r'\?\?+', 'Multiple question marks'),
        (r'!!+', 'Multiple exclamation marks'),
    ]
    
    # 9. DOUBLE WORDS
    double_word_patterns = [
        (r'\b(\w+)\s+\1\b', 'Repeated word'),
    ]
    
    # 10. SENTENCE STRUCTURE
    structure_patterns = [
        (r'^[A-Z][a-z]*ing\s', 'Sentence fragment: starts with gerund'),
        (r'\bbecause\s+of\s+\w+ing\b', 'Use "because" + clause, not "because of" + gerund'),
        (r'\bthe\s+reason\s+is\s+because\b', 'Redundant: use either "the reason is" or "because"'),
    ]
    
    # Combine all pattern groups
    all_patterns = [
        ('Subject-Verb Agreement', sv_patterns),
        ('Articles', article_patterns),
        ('Pronouns', pronoun_patterns),
        ('Tense', tense_patterns),
        ('Capitalization', cap_patterns),
        ('Prepositions', prep_patterns),
        ('Word Choice', word_choice_patterns),
        ('Punctuation', punct_patterns),
        ('Double Words', double_word_patterns),
        ('Sentence Structure', structure_patterns),
    ]
    
    # Apply all patterns
    for category, patterns in all_patterns:
        for pattern, description in patterns:
            try:
                matches = re.finditer(pattern, clean_text, re.IGNORECASE)
                for match in matches:
                    error_info = {
                        'category': category,
                        'description': description,
                        'matched_text': match.group(),
                        'position': match.span(),
                        'severity': get_error_severity(category)
                    }
                    errors.append(error_info)
                    error_details[category].append(error_info)
            except re.error:
                # Skip invalid regex patterns
                continue
    
    return errors, error_details

def get_error_severity(category):
    """Assign severity scores to different error types"""
    severity_map = {
        'Subject-Verb Agreement': 0.9,
        'Tense': 0.8,
        'Pronouns': 0.7,
        'Articles': 0.6,
        'Word Choice': 0.7,
        'Capitalization': 0.4,
        'Punctuation': 0.3,
        'Prepositions': 0.6,
        'Double Words': 0.8,
        'Sentence Structure': 0.9
    }
    return severity_map.get(category, 0.5)

def spell_check(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    misspelled = spell.unknown(words)
    return list(misspelled)

def grammar_quality_score(text):
    """Simple grammar quality assessment"""
    # print('text:', text)
    word_count = len(text.split())
    # print('word_count:', word_count)
    if word_count == 0:
        return {'overall_score': 0}
    
    # Simple grammar checking
    # grammar_errors = simple_grammar_rules(text)
    errors, error_details = robust_grammar_rules(text)

    # Calculate weighted error score based on severity
    total_severity = sum(error['severity'] for error in errors)
    severity_weighted_score = max(0, 1 - (total_severity / word_count)) if word_count > 0 else 0
    
    # Simple error count score
    simple_error_score = max(0, 1 - (len(errors) / word_count)) if word_count > 0 else 0
    
    # Spelling accuracy
    spell_errors = spell_check(text)
    # print('spell_errors:', spell_errors)
    spelling_score = max(0, 1 - (len(spell_errors) / word_count))
    # print('spelling_score:', spelling_score)
    
    # Readability
    readability = textstat.flesch_kincaid_grade(text)
    # print('readability:', readability)
    readability_score = 1.0 if 8 <= readability <= 12 else 0.7
    # print('readability_score:', readability_score)
    
    overall_score = (severity_weighted_score * 0.5 + spelling_score * 0.3 + readability_score * 0.2)
    # print('overall_score:', overall_score)
    
    # return {
    #     'overall_score': overall_score,
    #     'grammar_errors': len(grammar_errors),
    #     'spelling_errors': len(spell_errors),
    #     'readability_grade': readability
    # }

    return overall_score

# # calcualte mauve score
# _reference_lines = None
# def get_reference_lines():
#     """Loads and caches the reference text for MAUVE."""
#     global _reference_lines
#     if _reference_lines is None:
#         try:
#             with open(REFERENCE_TEXT_PATH, "r", encoding="utf-8") as f:
#                 _reference_lines = [line.strip() for line in f if line.strip()]
#         except FileNotFoundError:
#             print(f"Warning: MAUVE reference text not found at '{REFERENCE_TEXT_PATH}'. MAUVE score will be 0.")
#             _reference_lines = []
#     return _reference_lines

# # Step 1: Compute features for reference texts ONCE
# p_features = mauve.compute_featurizations(
#     texts=get_reference_lines(),  # Your reference texts
#     featurize_model_name="gpt2",
#     model_device_id=0,
#     batch_size=32,
#     verbose=False
# )

# def compute_mauve(generated_lines):
#     """Computes the MAUVE score against the reference text."""
 
#     out = mauve.compute_mauve(p_text=p_features, q_text=generated_lines, device_id=0, verbose=False, batch_size=32, featurize_model_name="gpt2",kmeans_num_redo=3,                 # Fewer k-means iterations
#     kmeans_max_iter=100)
#     return out.mauve

# evalute the generations
def evaluate_generation(text):
    d1 = round(distinct_n([text], 1), 4)
    d2 = round(distinct_n([text], 2), 4)
    ppl = round(compute_perplexity([text]), 4)
    sim = round(avg_pairwise_cosine(text.splitlines()), 4)
    gqs = round(grammar_quality_score(text), 4)
    # mauve_score = round(compute_mauve(text), 4)
    return {
        "distinct-1": d1,
        "distinct-2": d2,
        "perplexity": ppl,
        "avg_semantic_sim": sim,
        "avg_semantic_sim": sim,
        # "mauve": mauve_score,
        "gqs": gqs,
    }

# Helper functions
def load_table_from_run(run, tag):
    """Load table data from run summary."""
    tbl_info = run.summary.get(tag)
    if not tbl_info:
        return None, None
    
    try:
        art = run.file(tbl_info["path"])
        local_path = art.download(replace=True)
        if not isinstance(local_path, str):
            local_path = local_path.name
        with open(local_path, "r") as f:
            tbl = json.load(f)
        # print('tbl.get("data", []):', tbl.get("data", []))
        return tbl.get("columns", []), tbl.get("data", [])
    except Exception as e:
        print(f"  Could not load {tag}: {e}")
        return None, None

def find_best_generation(data, gen_idx):
    """Find generation with lowest perplexity."""
    best_metrics, best_idx, count = None, -1, 0
    for i, row in enumerate(data):
        try:
            text = row[gen_idx]
            # print('before')
            m = evaluate_generation(text)
            # print('after')
            # print('text:', text)
            # print('m:', m)
            count += 1
            if best_metrics is None or m["perplexity"] < best_metrics["perplexity"]:
                best_metrics, best_idx = m, i
        except Exception:
            continue
    return best_metrics, best_idx, count