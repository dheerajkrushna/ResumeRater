"""
semantic_scorer.py — TF-IDF based semantic similarity scoring.

Uses scikit-learn's TfidfVectorizer to compute cosine similarity between
the job description and each candidate's profile text. This provides a
semantic understanding layer beyond keyword matching.

Design decisions:
- TF-IDF over transformer embeddings due to compute constraints (CPU only, 5 min)
- Custom vocabulary boosting for domain-specific terms
- N-gram range (1,2) to capture phrases like "vector search", "hybrid retrieval"
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import JD_TEXT
from src.loader import extract_candidate_text


class SemanticScorer:
    """
    TF-IDF + Cosine Similarity scorer for candidate-JD matching.

    Precomputes a TF-IDF matrix for all candidates and the JD,
    then returns per-candidate similarity scores in [0, 1].
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,        # Apply log normalization
            strip_accents='unicode',
            stop_words='english',
            dtype=np.float32,
        )
        self.jd_vector = None
        self.is_fitted = False

    def fit_and_score(self, candidates):
        """
        Fit the TF-IDF vectorizer on all candidate texts + JD,
        then compute cosine similarity for each candidate.

        Args:
            candidates: list[dict] — all candidate records

        Returns:
            dict[str, float]: mapping of candidate_id → semantic score in [0, 1]
        """
        # Build corpus: JD first, then all candidates
        jd_text = JD_TEXT.lower()
        candidate_texts = []
        candidate_ids = []

        for cand in candidates:
            cand_id = cand.get('candidate_id', '')
            text = extract_candidate_text(cand)
            candidate_texts.append(text)
            candidate_ids.append(cand_id)

        # Combine JD + all candidate texts for fitting
        corpus = [jd_text] + candidate_texts

        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

        # JD vector is row 0; candidate vectors are rows 1..N
        jd_vec = tfidf_matrix[0:1]
        cand_matrix = tfidf_matrix[1:]

        # Compute cosine similarity (returns array of shape [1, N])
        similarities = cosine_similarity(jd_vec, cand_matrix).flatten()

        # Normalize to [0, 1] range using min-max scaling
        # (cosine similarity with TF-IDF is already in [0, 1] but
        # the range is usually narrow, so we stretch it)
        min_sim = similarities.min()
        max_sim = similarities.max()
        if max_sim > min_sim:
            normalized = (similarities - min_sim) / (max_sim - min_sim)
        else:
            normalized = np.zeros_like(similarities)

        # Build result mapping
        scores = {}
        for i, cand_id in enumerate(candidate_ids):
            scores[cand_id] = float(normalized[i])

        return scores

    def get_top_matching_terms(self, candidate, n=5):
        """
        Get the top N TF-IDF terms that contribute most to similarity
        between a candidate and the JD. Useful for reasoning.

        Args:
            candidate: dict — candidate record
            n: number of top terms to return

        Returns:
            list[str]: top matching terms
        """
        if not self.is_fitted:
            return []

        cand_text = extract_candidate_text(candidate)
        cand_vec = self.vectorizer.transform([cand_text])
        jd_vec = self.vectorizer.transform([JD_TEXT.lower()])

        # Element-wise product gives term-level contribution
        feature_names = self.vectorizer.get_feature_names_out()
        contribution = (cand_vec.multiply(jd_vec)).toarray().flatten()

        # Get top N indices
        top_indices = contribution.argsort()[-n:][::-1]
        top_terms = [
            feature_names[i] for i in top_indices
            if contribution[i] > 0
        ]

        return top_terms
