import pandas as pd
import numpy as np
import time
import os
import unicodedata
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TfidfModel:
    def __init__(self):
        self.THRESHOLD = 0.85
        self.MIN_OCCURRENCE =2
        self.human_memory = defaultdict(list)
        self.human_memory_set = defaultdict(set)
        self.human_counter = defaultdict(lambda: defaultdict(int))  # human_value counter
        self.vectorizers = {}
        self.vector_matrices = {}
        self.lookup_dict = defaultdict(dict)  # field -> ocr_value -> last human_value
        logger.info("TF-IDF model initialized.")

    def learn(self, field, ocr_value, human_value, suggestion):
        logger.debug(f"Learning for field '{field}' with OCR value '{ocr_value}', human value '{human_value}', suggestion '{suggestion}'")
        # Normalize inputs
        if not human_value:
            return

        # update lookup_dict
        self.lookup_dict[field][ocr_value] = human_value

        # update human_counter
        self.human_counter[field][human_value] += 1
        logger.debug(f"Updated human counter for field '{field}': {self.human_counter[field]}")

        # Chỉ học nếu human_value lặp lại >= MIN_OCCURRENCE và chưa học, và suggest != human
        if (
            self.human_counter[field][human_value] >= self.MIN_OCCURRENCE
            and human_value not in self.human_memory_set[field]
            and suggestion != human_value
        ):
            self.human_memory[field].append(human_value)
            self.human_memory_set[field].add(human_value)
            self._update_vectorizer(field)

        logger.debug(f"Learned for field '{field}': {human_value} (suggestion: {suggestion})")

    def _update_vectorizer(self, field):
        logger.debug(f"Updating vectorizer for field '{field}' with {len(self.human_memory[field])} human values")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        mat = vectorizer.fit_transform(self.human_memory[field])
        self.vectorizers[field] = vectorizer
        self.vector_matrices[field] = mat
        logger.debug(f"Vectorizer for field '{field}' updated with {len(self.human_memory[field])} human values")

    def suggest(self, field, ocr_value):
        logger.debug(f"Suggesting for field '{field}' with OCR value '{ocr_value}'")
        suggestion = None
        confidence = 0.0

        if not self.human_memory[field]:
            return suggestion, confidence
        
        vec = self.vectorizers[field].transform([ocr_value])
        sim = cosine_similarity(vec, self.vector_matrices[field])[0]
        best_idx = np.argmax(sim)
        best_score = sim[best_idx]
        best_score = round(float(best_score), 4)
        suggestion = self.human_memory[field][best_idx]

        if best_score >=self.THRESHOLD:
            logger.debug(f"✅  High confidence for field '{field}' with OCR value '{ocr_value}': {best_score} - suggestion value: {suggestion}")
            suggestion = self.human_memory[field][best_idx]
            confidence = best_score
        else:          # Fallback if confidence < THRESHOLD
            logger.debug(f"⚠️  Low confidence for field '{field}' with OCR value '{ocr_value}': {best_score} - suggestion value: {suggestion}. Using fallback.")
            
            # reset suggestion and confidence
            suggestion = None
            confidence = 0.0
            
            fallback = self.fallback_lookup(field, ocr_value)
            if fallback:
                suggestion = fallback
                confidence = 1.0  # Fallback alway has  confidence = 1.0
                logger.debug(f"🔄 Fallback suggestion for field '{field}' with OCR value '{ocr_value}': {suggestion}")
            else:
                suggestion = None
                confidence = 0.0
        logger.debug(f"Suggestion for field '{field}' with OCR value '{ocr_value}': {suggestion} - confidence: {confidence}")
        return suggestion, confidence

    def fallback_lookup(self, field, ocr_value):
        return self.lookup_dict[field].get(ocr_value, None)
