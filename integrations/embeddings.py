from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Free local embeddings using sentence-transformers.
    No API calls, runs entirely on your machine.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model name
                - all-MiniLM-L6-v2: Fast, small (80MB), good for general use
                - all-mpnet-base-v2: More accurate, larger (420MB)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load embedding model (lazy loading)."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✓ Embedding model loaded")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.info("Embeddings will be disabled")
            self.model = None

    def embed(self, text: str) -> np.ndarray:
        """
        Create embedding vector for text.

        Args:
            text: Text to embed (error message, code, etc.)

        Returns:
            Numpy array (384 dimensions for MiniLM)
        """
        if not self.model:
            return np.zeros(384)  # Fallback

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(384)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts at once (faster).

        Args:
            texts: List of text strings

        Returns:
            2D numpy array (num_texts x embedding_dim)
        """
        if not self.model:
            return np.zeros((len(texts), 384))

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return np.zeros((len(texts), 384))

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def find_most_similar(self,
                         query: str,
                         candidates: List[str],
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar texts from a list.

        Args:
            query: Query text (e.g., current error)
            candidates: List of candidate texts (e.g., known errors)
            top_k: Return top K most similar

        Returns:
            List of (index, similarity_score) tuples
        """
        if not candidates:
            return []

        # Embed query
        query_emb = self.embed(query)

        # Embed all candidates
        candidate_embs = self.embed_batch(candidates)

        # Calculate similarities
        similarities = []
        for i, cand_emb in enumerate(candidate_embs):
            sim = np.dot(query_emb, cand_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cand_emb)
            )
            similarities.append((i, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class SemanticPatternMatcher:
    """
    Match error patterns using semantic similarity.
    More intelligent than simple keyword matching.
    """

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.known_errors: List[str] = []
        self.known_solutions: List[str] = []

    def add_pattern(self, error_text: str, solution: str):
        """
        Add an error-solution pattern.

        Args:
            error_text: Error message/description
            solution: Solution that worked
        """
        self.known_errors.append(error_text)
        self.known_solutions.append(solution)

    def find_similar_errors(self, current_error: str, top_k: int = 3) -> List[dict]:
        """
        Find similar errors from knowledge base.

        Args:
            current_error: Current error message
            top_k: Number of results

        Returns:
            List of {error, solution, similarity} dicts
        """
        if not self.known_errors:
            return []

        # Find most similar errors
        matches = self.embedding_engine.find_most_similar(
            query=current_error,
            candidates=self.known_errors,
            top_k=top_k
        )

        results = []
        for idx, similarity in matches:
            if similarity > 0.6:  # Only return if reasonably similar
                results.append({
                    "error": self.known_errors[idx],
                    "solution": self.known_solutions[idx],
                    "similarity": similarity
                })

        return results

    def cluster_errors(self, errors: List[str], threshold: float = 0.75) -> List[List[int]]:
        """
        Group similar errors together.

        Args:
            errors: List of error messages
            threshold: Similarity threshold for clustering

        Returns:
            List of clusters (each cluster is list of indices)
        """
        if not errors:
            return []

        # Get embeddings for all errors
        embeddings = self.embedding_engine.embed_batch(errors)

        # Simple clustering: group by similarity
        clusters = []
        assigned = set()

        for i in range(len(errors)):
            if i in assigned:
                continue

            cluster = [i]
            assigned.add(i)

            # Find similar errors
            for j in range(i + 1, len(errors)):
                if j in assigned:
                    continue

                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )

                if sim >= threshold:
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters
