"""
Self-hosted AI Search Engine for the Swarms framework.

Provides a cost-free, privacy-preserving local document search
engine using TF-IDF indexing. No paid APIs or external services
required.

Features:
    - Index local files and directories (txt, md, pdf text, csv, json)
    - TF-IDF based full-text search with BM25-style ranking
    - Thread-safe document store with persistence
    - Designed as agent-compatible tools for Swarms integration

Usage:
    from swarms.tools.search_engine import DocumentStore

    store = DocumentStore()
    store.index_directory("/path/to/docs")
    results = store.search("my query", top_k=5)
"""

import json
import math
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="search_engine")

# Supported file extensions for indexing
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".py",
    ".js",
    ".html",
    ".xml",
    ".yaml",
    ".yml",
    ".rst",
    ".cfg",
    ".ini",
    ".toml",
}

# Common English stop words for filtering
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "up",
        "out",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "as",
        "from",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "because",
        "until",
        "while",
        "how",
        "where",
        "when",
        "why",
        "am",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens,
    filtering stop words."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOP_WORDS and len(tok) > 1
    ]


def _read_file_content(filepath: str) -> Optional[str]:
    """Safely read text content from a file.

    Args:
        filepath: Path to the file to read.

    Returns:
        File content as a string, or None if unreadable.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (OSError, PermissionError) as exc:
        logger.warning(f"Cannot read {filepath}: {exc}")
        return None


class SearchResult:
    """Represents a single search result.

    Attributes:
        doc_id: Unique document identifier.
        title: Document title.
        content: Snippet of matching content.
        score: Relevance score.
        metadata: Additional document metadata.
    """

    __slots__ = (
        "doc_id",
        "title",
        "content",
        "score",
        "metadata",
    )

    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.score = score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"SearchResult(doc_id={self.doc_id!r}, "
            f"title={self.title!r}, score={self.score:.4f})"
        )


class DocumentStore:
    """Thread-safe local document store with TF-IDF search.

    Indexes documents and supports BM25-style relevance ranking.
    Entirely local — no external APIs or services required.

    Args:
        name: Name for this document store.
        persist_path: Optional path to persist the index.
        bm25_k1: BM25 term frequency saturation parameter.
        bm25_b: BM25 length normalization parameter.

    Example:
        >>> store = DocumentStore(name="my_docs")
        >>> store.add_document("doc1", "Hello World", "This is a test document about AI.")
        >>> results = store.search("AI test")
        >>> print(results[0].title)
        'Hello World'
    """

    def __init__(
        self,
        name: str = "default",
        persist_path: Optional[str] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.name = name
        self.persist_path = persist_path
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._lock = threading.Lock()
        # doc_id -> {title, content, metadata, tokens}
        self._documents: Dict[str, Dict[str, Any]] = {}
        # token -> set of doc_ids
        self._inverted_index: Dict[str, set] = defaultdict(set)
        # doc_id -> {token -> count}
        self._term_freqs: Dict[str, Dict[str, int]] = {}
        # token -> document frequency count
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._avg_doc_len: float = 0.0

        if persist_path and os.path.isfile(persist_path):
            self._load(persist_path)

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the store and update the index.

        Args:
            doc_id: Unique identifier for the document.
            title: Document title.
            content: Full text content of the document.
            metadata: Optional metadata dictionary.
        """
        tokens = _tokenize(f"{title} {content}")
        tf: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1

        with self._lock:
            # Remove old version if exists
            if doc_id in self._documents:
                self._remove_doc_from_index(doc_id)

            self._documents[doc_id] = {
                "title": title,
                "content": content,
                "metadata": metadata or {},
                "token_count": len(tokens),
            }
            self._term_freqs[doc_id] = dict(tf)

            for tok in tf:
                self._inverted_index[tok].add(doc_id)
                self._doc_freqs[tok] += 1

            self._update_avg_doc_len()

        logger.debug(
            f"Indexed document {doc_id!r} "
            f"({len(tokens)} tokens)"
        )

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the store.

        Args:
            doc_id: The document to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if doc_id not in self._documents:
                return False
            self._remove_doc_from_index(doc_id)
            self._update_avg_doc_len()
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        snippet_length: int = 300,
    ) -> List[SearchResult]:
        """Search the document store using BM25 ranking.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            snippet_length: Max characters for content snippets.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)
        n = len(self._documents)
        if n == 0:
            return []

        with self._lock:
            for tok in query_tokens:
                if tok not in self._inverted_index:
                    continue
                df = self._doc_freqs.get(tok, 0)
                # IDF with smoothing
                idf = math.log(
                    (n - df + 0.5) / (df + 0.5) + 1.0
                )
                for did in self._inverted_index[tok]:
                    tf = self._term_freqs[did].get(tok, 0)
                    dl = self._documents[did]["token_count"]
                    avg_dl = self._avg_doc_len or 1.0
                    # BM25 scoring
                    numerator = tf * (self.bm25_k1 + 1)
                    denominator = tf + self.bm25_k1 * (
                        1
                        - self.bm25_b
                        + self.bm25_b * (dl / avg_dl)
                    )
                    scores[did] += idf * (
                        numerator / denominator
                    )

            ranked = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            results = []
            for did, score in ranked:
                doc = self._documents[did]
                snippet = self._get_snippet(
                    doc["content"],
                    query_tokens,
                    snippet_length,
                )
                results.append(
                    SearchResult(
                        doc_id=did,
                        title=doc["title"],
                        content=snippet,
                        score=score,
                        metadata=doc["metadata"],
                    )
                )

        return results

    def index_file(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Index a single file.

        Args:
            filepath: Path to the file.
            metadata: Optional metadata to attach.

        Returns:
            True if indexed successfully, False otherwise.
        """
        path = Path(filepath)
        if not path.is_file():
            logger.warning(f"File not found: {filepath}")
            return False

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug(
                f"Skipping unsupported file type: {path.suffix}"
            )
            return False

        content = _read_file_content(str(path))
        if content is None:
            return False

        doc_id = str(path.resolve())
        title = path.name
        meta = {
            "filepath": str(path.resolve()),
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            meta.update(metadata)

        self.add_document(doc_id, title, content, meta)
        return True

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[set] = None,
    ) -> int:
        """Index all supported files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.
            extensions: Optional set of extensions to include.
                Defaults to SUPPORTED_EXTENSIONS.

        Returns:
            Number of files successfully indexed.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(
                f"Directory not found: {directory}"
            )

        exts = extensions or SUPPORTED_EXTENSIONS
        count = 0
        pattern = "**/*" if recursive else "*"

        for fpath in dir_path.glob(pattern):
            if (
                fpath.is_file()
                and fpath.suffix.lower() in exts
            ):
                if self.index_file(str(fpath)):
                    count += 1

        logger.info(
            f"Indexed {count} files from {directory!r}"
        )
        return count

    def save(
        self, filepath: Optional[str] = None
    ) -> None:
        """Persist the document store to disk.

        Args:
            filepath: Path to save to. Uses persist_path if not
                specified.
        """
        path = filepath or self.persist_path
        if not path:
            raise ValueError(
                "No filepath specified for saving"
            )

        with self._lock:
            data = {
                "name": self.name,
                "bm25_k1": self.bm25_k1,
                "bm25_b": self.bm25_b,
                "documents": {},
            }
            for did, doc in self._documents.items():
                data["documents"][did] = {
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                }

        os.makedirs(
            os.path.dirname(os.path.abspath(path)),
            exist_ok=True,
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved document store to {path!r}")

    def _load(self, filepath: str) -> None:
        """Load a persisted document store from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.name = data.get("name", self.name)
        self.bm25_k1 = data.get("bm25_k1", self.bm25_k1)
        self.bm25_b = data.get("bm25_b", self.bm25_b)

        for did, doc in data.get("documents", {}).items():
            self.add_document(
                doc_id=did,
                title=doc["title"],
                content=doc["content"],
                metadata=doc.get("metadata"),
            )

        logger.info(
            f"Loaded {self.document_count} documents "
            f"from {filepath!r}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the document store.

        Returns:
            Dictionary with store statistics.
        """
        with self._lock:
            return {
                "name": self.name,
                "document_count": len(self._documents),
                "unique_terms": len(self._inverted_index),
                "avg_document_length": round(
                    self._avg_doc_len, 2
                ),
            }

    def _remove_doc_from_index(self, doc_id: str) -> None:
        """Remove a document from internal index structures.
        Caller must hold ``self._lock``."""
        old_tf = self._term_freqs.pop(doc_id, {})
        for tok in old_tf:
            self._inverted_index[tok].discard(doc_id)
            self._doc_freqs[tok] -= 1
            if self._doc_freqs[tok] <= 0:
                del self._doc_freqs[tok]
                del self._inverted_index[tok]
        del self._documents[doc_id]

    def _update_avg_doc_len(self) -> None:
        """Recalculate average document length.
        Caller must hold ``self._lock``."""
        if self._documents:
            total = sum(
                d["token_count"]
                for d in self._documents.values()
            )
            self._avg_doc_len = total / len(self._documents)
        else:
            self._avg_doc_len = 0.0

    @staticmethod
    def _get_snippet(
        content: str,
        query_tokens: List[str],
        max_length: int,
    ) -> str:
        """Extract a relevant snippet from document content."""
        if len(content) <= max_length:
            return content

        content_lower = content.lower()
        best_pos = 0
        best_count = 0

        # Slide a window to find the area with most query
        # term matches
        window = max_length
        for i in range(0, len(content) - window, window // 4):
            chunk = content_lower[i : i + window]
            count = sum(
                1 for t in query_tokens if t in chunk
            )
            if count > best_count:
                best_count = count
                best_pos = i

        start = max(0, best_pos)
        end = min(len(content), start + max_length)
        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet


def create_search_tool_functions(
    store: DocumentStore,
) -> Dict[str, Any]:
    """Create tool functions compatible with Swarms agents.

    Returns a dictionary of callable tool functions that agents
    can use for searching and indexing.

    Args:
        store: The DocumentStore instance to use.

    Returns:
        Dictionary mapping function names to callables.
    """

    def search_documents(
        query: str, top_k: int = 5
    ) -> str:
        """Search indexed documents for relevant results.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            Formatted string of search results.
        """
        results = store.search(query, top_k=top_k)
        if not results:
            return "No results found for the query."

        output_parts = []
        for i, r in enumerate(results, 1):
            output_parts.append(
                f"[Result {i}] (Score: {r.score:.4f})\n"
                f"Title: {r.title}\n"
                f"Content: {r.content}\n"
            )
        return "\n".join(output_parts)

    def index_file(filepath: str) -> str:
        """Index a file for searching.

        Args:
            filepath: Path to the file to index.

        Returns:
            Status message.
        """
        success = store.index_file(filepath)
        if success:
            return f"Successfully indexed: {filepath}"
        return f"Failed to index: {filepath}"

    def index_directory(
        directory: str, recursive: bool = True
    ) -> str:
        """Index all supported files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Search subdirectories.

        Returns:
            Status message with count of indexed files.
        """
        count = store.index_directory(
            directory, recursive=recursive
        )
        return (
            f"Indexed {count} files from {directory}"
        )

    def get_index_stats() -> str:
        """Get statistics about the current document index.

        Returns:
            Formatted statistics string.
        """
        stats = store.get_stats()
        return (
            f"Document Store: {stats['name']}\n"
            f"Documents: {stats['document_count']}\n"
            f"Unique Terms: {stats['unique_terms']}\n"
            f"Avg Doc Length: {stats['avg_document_length']}"
        )

    return {
        "search_documents": search_documents,
        "index_file": index_file,
        "index_directory": index_directory,
        "get_index_stats": get_index_stats,
    }
