"""Tests for the self-hosted search engine module."""

import json
import os
import tempfile

import pytest

from swarms.tools.search_engine import (
    DocumentStore,
    SearchResult,
    _tokenize,
    create_search_tool_functions,
    SUPPORTED_EXTENSIONS,
)


# ── Tokenizer Tests ──────────────────────────────────────


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello World Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_single_char_tokens_removed(self):
        tokens = _tokenize("I a b c test")
        assert "test" in tokens
        # Single characters should be filtered
        assert "b" not in tokens
        assert "c" not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numeric_tokens(self):
        tokens = _tokenize("test 123 value 42")
        assert "test" in tokens
        assert "123" in tokens
        assert "value" in tokens
        assert "42" in tokens


# ── SearchResult Tests ───────────────────────────────────


class TestSearchResult:
    def test_to_dict(self):
        r = SearchResult(
            doc_id="d1",
            title="Test",
            content="content",
            score=1.5,
            metadata={"key": "val"},
        )
        d = r.to_dict()
        assert d["doc_id"] == "d1"
        assert d["title"] == "Test"
        assert d["content"] == "content"
        assert d["score"] == 1.5
        assert d["metadata"] == {"key": "val"}

    def test_repr(self):
        r = SearchResult(
            doc_id="d1",
            title="Test",
            content="c",
            score=2.0,
        )
        assert "d1" in repr(r)
        assert "Test" in repr(r)

    def test_default_metadata(self):
        r = SearchResult(
            doc_id="d1",
            title="T",
            content="C",
            score=0.0,
        )
        assert r.metadata == {}


# ── DocumentStore Tests ──────────────────────────────────


class TestDocumentStore:
    @pytest.fixture
    def store(self):
        return DocumentStore(name="test_store")

    @pytest.fixture
    def populated_store(self, store):
        store.add_document(
            "doc1",
            "Python Guide",
            "Python is a programming language "
            "used for machine learning and data science.",
        )
        store.add_document(
            "doc2",
            "Java Basics",
            "Java is an object-oriented programming "
            "language widely used in enterprise.",
        )
        store.add_document(
            "doc3",
            "Machine Learning",
            "Machine learning is a subset of "
            "artificial intelligence that uses "
            "algorithms to learn from data.",
        )
        return store

    def test_add_document(self, store):
        store.add_document(
            "d1", "Title", "Content here"
        )
        assert store.document_count == 1

    def test_add_multiple_documents(self, store):
        for i in range(5):
            store.add_document(
                f"d{i}", f"Title {i}", f"Content {i}"
            )
        assert store.document_count == 5

    def test_add_duplicate_replaces(self, store):
        store.add_document(
            "d1", "Title1", "Content1"
        )
        store.add_document(
            "d1", "Title2", "Content2"
        )
        assert store.document_count == 1

    def test_remove_document(self, store):
        store.add_document(
            "d1", "Title", "Content"
        )
        assert store.remove_document("d1") is True
        assert store.document_count == 0

    def test_remove_nonexistent(self, store):
        assert store.remove_document("nope") is False

    def test_search_basic(self, populated_store):
        results = populated_store.search("python")
        assert len(results) > 0
        assert results[0].title == "Python Guide"

    def test_search_relevance_ranking(
        self, populated_store
    ):
        results = populated_store.search(
            "machine learning"
        )
        assert len(results) > 0
        titles = [r.title for r in results]
        assert "Machine Learning" in titles

    def test_search_no_results(self, populated_store):
        results = populated_store.search(
            "xyznonexistent"
        )
        assert len(results) == 0

    def test_search_empty_query(self, populated_store):
        results = populated_store.search("")
        assert len(results) == 0

    def test_search_stop_words_only(
        self, populated_store
    ):
        results = populated_store.search("the and or")
        assert len(results) == 0

    def test_search_top_k(self, populated_store):
        results = populated_store.search(
            "programming", top_k=1
        )
        assert len(results) <= 1

    def test_search_scores_descending(
        self, populated_store
    ):
        results = populated_store.search("programming")
        for i in range(len(results) - 1):
            assert (
                results[i].score >= results[i + 1].score
            )

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert len(results) == 0

    def test_get_stats(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["name"] == "test_store"
        assert stats["document_count"] == 3
        assert stats["unique_terms"] > 0
        assert stats["avg_document_length"] > 0

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["document_count"] == 0
        assert stats["unique_terms"] == 0
        assert stats["avg_document_length"] == 0


# ── File Indexing Tests ──────────────────────────────────


class TestFileIndexing:
    @pytest.fixture
    def store(self):
        return DocumentStore(name="file_test")

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            # Create test files
            with open(
                os.path.join(d, "readme.txt"), "w"
            ) as f:
                f.write(
                    "This is a readme about the project."
                )
            with open(
                os.path.join(d, "notes.md"), "w"
            ) as f:
                f.write(
                    "# Notes\n\nImportant notes about "
                    "machine learning algorithms."
                )
            with open(
                os.path.join(d, "data.csv"), "w"
            ) as f:
                f.write("name,value\ntest,42\n")
            # Create subdirectory
            sub = os.path.join(d, "sub")
            os.makedirs(sub)
            with open(
                os.path.join(sub, "deep.txt"), "w"
            ) as f:
                f.write("Deep nested document content.")
            # Unsupported file
            with open(
                os.path.join(d, "image.png"), "wb"
            ) as f:
                f.write(b"\x89PNG")
            yield d

    def test_index_file(self, store, temp_dir):
        path = os.path.join(temp_dir, "readme.txt")
        result = store.index_file(path)
        assert result is True
        assert store.document_count == 1

    def test_index_nonexistent_file(self, store):
        result = store.index_file("/nonexistent/file.txt")
        assert result is False

    def test_index_unsupported_extension(
        self, store, temp_dir
    ):
        path = os.path.join(temp_dir, "image.png")
        result = store.index_file(path)
        assert result is False

    def test_index_directory(self, store, temp_dir):
        count = store.index_directory(temp_dir)
        # readme.txt, notes.md, data.csv, deep.txt = 4
        assert count == 4

    def test_index_directory_non_recursive(
        self, store, temp_dir
    ):
        count = store.index_directory(
            temp_dir, recursive=False
        )
        # Only top-level: readme.txt, notes.md, data.csv
        assert count == 3

    def test_index_invalid_directory(self, store):
        with pytest.raises(ValueError):
            store.index_directory("/nonexistent/dir")

    def test_search_indexed_files(
        self, store, temp_dir
    ):
        store.index_directory(temp_dir)
        results = store.search("machine learning")
        assert len(results) > 0

    def test_index_file_with_metadata(
        self, store, temp_dir
    ):
        path = os.path.join(temp_dir, "readme.txt")
        store.index_file(
            path, metadata={"category": "docs"}
        )
        results = store.search("readme project")
        assert len(results) > 0
        assert (
            results[0].metadata["category"] == "docs"
        )


# ── Persistence Tests ────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.json")

            store1 = DocumentStore(
                name="persist_test",
                persist_path=path,
            )
            store1.add_document(
                "d1", "Test Doc", "Test content here"
            )
            store1.add_document(
                "d2",
                "Another",
                "More content about testing",
            )
            store1.save()

            store2 = DocumentStore(
                name="loaded",
                persist_path=path,
            )
            assert store2.document_count == 2
            results = store2.search("test")
            assert len(results) > 0

    def test_save_no_path_raises(self):
        store = DocumentStore(name="no_path")
        with pytest.raises(ValueError):
            store.save()

    def test_save_creates_directories(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(
                d, "sub", "dir", "index.json"
            )
            store = DocumentStore(
                name="deep_save"
            )
            store.add_document(
                "d1", "Title", "Content"
            )
            store.save(path)
            assert os.path.isfile(path)


# ── Tool Functions Tests ─────────────────────────────────


class TestToolFunctions:
    @pytest.fixture
    def tools(self):
        store = DocumentStore(name="tool_test")
        store.add_document(
            "d1",
            "Python Tutorial",
            "Learn Python programming basics.",
        )
        store.add_document(
            "d2",
            "Java Guide",
            "Getting started with Java development.",
        )
        return create_search_tool_functions(store)

    def test_search_documents(self, tools):
        result = tools["search_documents"]("Python")
        assert "Python Tutorial" in result
        assert "Score:" in result

    def test_search_no_results(self, tools):
        result = tools["search_documents"](
            "xyznonexistent"
        )
        assert "No results" in result

    def test_get_index_stats(self, tools):
        result = tools["get_index_stats"]()
        assert "tool_test" in result
        assert "Documents: 2" in result

    def test_index_file_tool(self, tools):
        result = tools["index_file"](
            "/nonexistent/file.txt"
        )
        assert "Failed" in result

    def test_index_file_success(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False
        ) as f:
            f.write("Test content for tool indexing.")
            f.flush()
            path = f.name

        try:
            store = DocumentStore(name="tool_idx")
            tools = create_search_tool_functions(store)
            result = tools["index_file"](path)
            assert "Successfully" in result
        finally:
            os.unlink(path)

    def test_index_directory_tool(self):
        with tempfile.TemporaryDirectory() as d:
            with open(
                os.path.join(d, "test.txt"), "w"
            ) as f:
                f.write("Test document content.")

            store = DocumentStore(name="dir_tool")
            tools = create_search_tool_functions(store)
            result = tools["index_directory"](d)
            assert "Indexed 1 files" in result
