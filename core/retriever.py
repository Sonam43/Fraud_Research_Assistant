"""Section-aware, evidence-grounded hybrid retrieval (Notebook 09 compatible).

Loads an existing FAISS index, BM25 index, and passage metadata. The BM25
pickle must come only from your own trusted Colab project; never unpickle
third-party files. Retrieval uses BM25 + dense BGE embeddings + RRF.

CLI (from project root):
    python -m core.retriever --verify-only
    python -m core.retriever --query "How is XGBoost used in fraud detection?"

Historical settings from Notebook 09:
    EMBEDDING_MODEL = BAAI/bge-small-en-v1.5
    FIRST_STAGE_K = 20
    RRF_K = 60
    FINAL_K = 10
    normalize_embeddings = True
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[1] / "data" / "indexes" / "section_aware"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FIRST_STAGE_K = 20
RRF_K = 60
FINAL_K = 10


def bm25_tokenize(text: str) -> list[str]:
    """Reproduce the exact BM25 tokenizer used in Notebook 09."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def metadata_fingerprint(records: list[dict[str, Any]]) -> str:
    """Match the SHA-256 metadata fingerprint stored in the project pickle."""
    digest = hashlib.sha256()
    for record in records:
        line = (
            f"{int(record['vector_id'])}\t"
            f"{record['chunk_id']}\t"
            f"{record['article_id']}\t"
            f"{record['text']}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _find_bm25_model(payload: Any, bm25_type: type, depth: int = 0) -> Any:
    if isinstance(payload, bm25_type):
        return payload
    if isinstance(payload, dict) and depth < 3:
        for key in ("bm25", "bm25_model", "model", "index", "retriever"):
            if key in payload:
                result = _find_bm25_model(payload[key], bm25_type, depth + 1)
                if result is not None:
                    return result
    return None


def _normalize_pages(value: Any) -> list[int]:
    """Convert page_numbers field into a list of positive PDF page numbers."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        pages = []
        for item in value:
            pages.extend(_normalize_pages(item))
        return list(dict.fromkeys(pages))
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("none", "null", "nan"):
            return []
        try:
            return _normalize_pages(json.loads(value))
        except json.JSONDecodeError:
            try:
                return _normalize_pages(ast.literal_eval(value))
            except (ValueError, SyntaxError):
                raise ValueError(f"Unrecognized page_numbers field: {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Unrecognized page_numbers type: {type(value).__name__}")
    if int(value) != value or value < 1:
        raise ValueError(f"Invalid PDF page number: {value!r}")
    return [int(value)]


class HybridRetriever:
    """Verified index loader and first-stage BM25 + FAISS + RRF retriever."""

    def __init__(
        self,
        asset_dir: Path | str = ASSET_DIR,
        *,
        full_bm25_check: bool = False,
        embedding_model: Any = None,
    ) -> None:
        try:
            import faiss
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "Missing retrieval dependency. Run: "
                "python -m pip install -r requirements-retrieval.txt"
            ) from exc

        self.asset_dir = Path(asset_dir)
        paths = {
            "FAISS index": self.asset_dir / "index.faiss",
            "BM25 index": self.asset_dir / "bm25_index.pkl",
            "passage metadata": self.asset_dir / "metadata.jsonl",
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

        self.metadata: list[dict[str, Any]] = []
        with paths["passage metadata"].open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at metadata line {line_no}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected metadata object at line {line_no}")
                self.metadata.append(record)
        if not self.metadata:
            raise ValueError("metadata.jsonl has no passage records.")

        required = {"vector_id", "chunk_id", "article_id", "title", "text", "page_numbers"}
        chunk_ids = set()
        for position, record in enumerate(self.metadata):
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Metadata row {position} missing fields: {sorted(missing)}")
            if int(record["vector_id"]) != position:
                raise ValueError(f"Metadata row {position}: vector_id must match row position")
            chunk_id = str(record["chunk_id"])
            if not chunk_id or chunk_id in chunk_ids:
                raise ValueError(f"Missing or duplicate chunk_id at metadata row {position}")
            chunk_ids.add(chunk_id)
            if not str(record["article_id"]).strip() or not str(record["text"]).strip():
                raise ValueError(f"Empty article_id or passage text at metadata row {position}")

        self.faiss_index = faiss.read_index(str(paths["FAISS index"]))
        if not self.faiss_index.is_trained:
            raise ValueError("FAISS index is not trained.")

        # SECURITY: Pickle can execute code. Never load an untrusted file.
        with paths["BM25 index"].open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("BM25 pickle should contain a dictionary, as in Notebook 09")
        self.bm25_index = _find_bm25_model(payload, BM25Okapi)
        if self.bm25_index is None:
            raise TypeError("No BM25Okapi instance found in the saved dictionary")
        if payload.get("strategy") != "section_aware":
            raise ValueError(
                "BM25 chunking strategy mismatch: "
                f"expected section_aware, found {payload.get('strategy')!r}"
            )
        expected_fingerprint = payload.get("fingerprint")
        if not expected_fingerprint:
            raise ValueError("BM25 pickle has no metadata fingerprint")
        if metadata_fingerprint(self.metadata) != expected_fingerprint:
            raise ValueError(
                "BM25 fingerprint mismatch: these indexes and metadata "
                "may come from different indexing runs"
            )
        if self.faiss_index.ntotal != len(self.metadata):
            raise ValueError("FAISS passage count differs from metadata count")
        if self.bm25_index.corpus_size != len(self.metadata):
            raise ValueError("BM25 passage count differs from metadata count")
        if len(self.bm25_index.doc_freqs) != len(self.metadata):
            raise ValueError("BM25 document frequency count differs from metadata count")
        if len(self.bm25_index.doc_len) != len(self.metadata):
            raise ValueError("BM25 document-length count differs from metadata count")

        if full_bm25_check:
            for position, record in enumerate(self.metadata):
                tokens = bm25_tokenize(record["text"])
                if (
                    Counter(tokens) != Counter(self.bm25_index.doc_freqs[position])
                    or len(tokens) != self.bm25_index.doc_len[position]
                ):
                    raise ValueError(f"BM25 token mismatch at metadata row {position}")

        self._embedding_model = embedding_model  # Lazy-load only for live queries.

    def verify_summary(self) -> dict[str, Any]:
        """Report validated file alignment (not real-world answer quality)."""
        return {
            "status": "VERIFIED",
            "strategy": "section_aware",
            "passages": len(self.metadata),
            "faiss_vectors": int(self.faiss_index.ntotal),
            "bm25_documents": int(self.bm25_index.corpus_size),
            "faiss_dimension": int(self.faiss_index.d),
            "metadata_fingerprint": "MATCH",
        }

    def _embed_query(self, question: str) -> np.ndarray:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers to run live queries: "
                    "python -m pip install sentence-transformers"
                ) from exc
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        encoded = self._embedding_model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vector = np.asarray(encoded, dtype="float32").reshape(1, -1)
        if vector.shape[1] != self.faiss_index.d:
            raise ValueError(
                f"FAISS dimension mismatch: expected {self.faiss_index.d}, "
                f"embedding returned {vector.shape[1]}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite values in query embedding")
        return vector

    def search(
        self,
        question: str,
        *,
        top_k: int = FINAL_K,
        first_stage_k: int = FIRST_STAGE_K,
        rrf_k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        """Retrieve citation-ready passages using Notebook 09's System E method."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please provide a nonempty research question")
        if top_k < 1 or first_stage_k < 1 or rrf_k < 1:
            raise ValueError("top_k, first_stage_k, and rrf_k must be positive")
        if top_k > first_stage_k:
            raise ValueError("top_k must not exceed first_stage_k")

        tokens = bm25_tokenize(question)
        if not tokens:
            raise ValueError("The question contains no valid BM25 tokens")
        bm25_scores = np.asarray(self.bm25_index.get_scores(tokens), dtype="float64")
        if len(bm25_scores) != len(self.metadata) or not np.isfinite(bm25_scores).all():
            raise ValueError("BM25 scores are missing or invalid")
        bm25_positions = np.argsort(-bm25_scores, kind="stable")[:first_stage_k]

        query_vector = self._embed_query(question)
        dense_scores, dense_positions = self.faiss_index.search(
            query_vector, min(first_stage_k, len(self.metadata))
        )
        if not np.isfinite(dense_scores[dense_positions >= 0]).all():
            raise ValueError("FAISS returned non-finite scores")

        fused: defaultdict[int, float] = defaultdict(float)
        for rank, position in enumerate(bm25_positions, start=1):
            fused[int(position)] += 1.0 / (rrf_k + rank)
        for rank, position in enumerate(dense_positions[0], start=1):
            if position >= 0:
                fused[int(position)] += 1.0 / (rrf_k + rank)

        ordered = sorted(fused, key=lambda position: (-fused[position], position))[:top_k]
        results: list[dict[str, Any]] = []
        for rank, position in enumerate(ordered, start=1):
            record = self.metadata[position]
            results.append({
                "source_id": f"S{rank}",
                "rank": rank,
                "score": float(fused[position]),
                "vector_id": int(record["vector_id"]),
                "chunk_id": str(record["chunk_id"]),
                "article_id": str(record["article_id"]),
                "title": str(record.get("title", "")),
                "page_numbers": _normalize_pages(record.get("page_numbers")),
                "text": str(record["text"]),
            })
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or test section-aware hybrid retrieval")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true", help="Verify index/metadata alignment")
    mode.add_argument("--query", type=str, help="Retrieve passages for one question")
    parser.add_argument("--full-check", action="store_true", help="Recompute BM25 token frequencies")
    parser.add_argument("--top-k", type=int, default=5, help="Passages shown for --query (default: 5)")
    args = parser.parse_args()

    retriever = HybridRetriever(full_bm25_check=args.full_check)
    print("\nRETRIEVAL ASSET VALIDATION")
    print(json.dumps(retriever.verify_summary(), indent=2))
    if args.query:
        print("\nTOP RETRIEVED PASSAGES")
        for result in retriever.search(args.query, top_k=args.top_k):
            print(
                f"\n[{result['source_id']}] {result['title']} "
                f"| Article: {result['article_id']} "
                f"| PDF pages: {result['page_numbers']} "
                f"| RRF score: {result['score']:.6f}"
            )
            print(result["text"][:500].replace("\n", " ").strip(), "...")


if __name__ == "__main__":
    main()
