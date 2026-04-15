"""Tests for the SearchSwarm orchestration module."""

import os
import tempfile

import pytest

from swarms.tools.search_engine import DocumentStore
from swarms.structs.search_swarm import SearchSwarm


class TestSearchSwarmInit:
    """Test SearchSwarm initialization."""

    def test_default_init(self):
        swarm = SearchSwarm(agent_mode=False)
        assert swarm.name == "SearchSwarm"
        assert swarm.agent_mode is False
        assert swarm.document_store is not None
        assert swarm.document_store.document_count == 0

    def test_custom_name(self):
        swarm = SearchSwarm(
            name="MySearch", agent_mode=False
        )
        assert swarm.name == "MySearch"

    def test_with_document_store(self):
        store = DocumentStore(name="custom")
        store.add_document(
            "d1", "Test", "Test content"
        )
        swarm = SearchSwarm(
            document_store=store, agent_mode=False
        )
        assert swarm.document_store.document_count == 1

    def test_with_documents_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with open(
                os.path.join(d, "test.txt"), "w"
            ) as f:
                f.write("Test document content.")

            swarm = SearchSwarm(
                documents_directory=d,
                agent_mode=False,
            )
            assert (
                swarm.document_store.document_count == 1
            )

    def test_get_stats(self):
        swarm = SearchSwarm(agent_mode=False)
        stats = swarm.get_stats()
        assert "swarm_id" in stats
        assert "swarm_name" in stats
        assert stats["agent_mode"] is False
        assert stats["document_count"] == 0


class TestSearchSwarmSearch:
    """Test SearchSwarm search functionality."""

    @pytest.fixture
    def swarm(self):
        store = DocumentStore(name="test")
        store.add_document(
            "doc1",
            "Python Guide",
            "Python is a versatile programming "
            "language for AI and data science.",
        )
        store.add_document(
            "doc2",
            "Java Basics",
            "Java is an enterprise programming "
            "language for building applications.",
        )
        store.add_document(
            "doc3",
            "Deep Learning",
            "Deep learning uses neural networks "
            "for artificial intelligence tasks.",
        )
        return SearchSwarm(
            document_store=store,
            agent_mode=False,
        )

    def test_search_returns_results(self, swarm):
        results = swarm.search("python")
        assert len(results) > 0
        assert results[0].title == "Python Guide"

    def test_search_relevance(self, swarm):
        results = swarm.search("artificial intelligence")
        assert len(results) > 0

    def test_search_no_results(self, swarm):
        results = swarm.search("xyznonexistent")
        assert len(results) == 0

    def test_run_without_agents(self, swarm):
        result = swarm.run("python programming")
        assert isinstance(result, str)
        assert "Python Guide" in result

    def test_run_no_results(self, swarm):
        result = swarm.run("xyznonexistent")
        assert "No results" in result

    def test_search_top_k(self, swarm):
        results = swarm.search(
            "programming", top_k=1
        )
        assert len(results) <= 1


class TestSearchSwarmIndexing:
    """Test SearchSwarm file indexing."""

    @pytest.fixture
    def swarm(self):
        return SearchSwarm(agent_mode=False)

    def test_index_file(self, swarm):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False
        ) as f:
            f.write("Test content for indexing.")
            f.flush()
            path = f.name

        try:
            result = swarm.index_file(path)
            assert result is True
            assert (
                swarm.document_store.document_count == 1
            )
        finally:
            os.unlink(path)

    def test_index_directory(self, swarm):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                with open(
                    os.path.join(d, f"doc{i}.txt"),
                    "w",
                ) as f:
                    f.write(f"Document {i} content.")

            count = swarm.index_directory(d)
            assert count == 3

    def test_save_and_load_index(self, swarm):
        swarm.document_store.add_document(
            "d1", "Test", "Content"
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.json")
            swarm.save_index(path)
            assert os.path.isfile(path)


class TestSearchSwarmMergeResults:
    """Test result merging logic."""

    def test_merge_deduplicates(self):
        from swarms.tools.search_engine import (
            SearchResult,
        )

        primary = [
            SearchResult(
                "d1", "T1", "C1", 2.0
            ),
            SearchResult(
                "d2", "T2", "C2", 1.5
            ),
        ]
        secondary = [
            SearchResult(
                "d2", "T2", "C2", 1.8
            ),
            SearchResult(
                "d3", "T3", "C3", 1.0
            ),
        ]
        merged = SearchSwarm._merge_results(
            primary, secondary, top_k=10
        )
        ids = [r.doc_id for r in merged]
        assert ids.count("d2") == 1
        assert len(merged) == 3

    def test_merge_respects_top_k(self):
        from swarms.tools.search_engine import (
            SearchResult,
        )

        primary = [
            SearchResult(
                f"d{i}", f"T{i}", f"C{i}", float(i)
            )
            for i in range(5)
        ]
        merged = SearchSwarm._merge_results(
            primary, [], top_k=3
        )
        assert len(merged) == 3

    def test_merge_sorted_by_score(self):
        from swarms.tools.search_engine import (
            SearchResult,
        )

        primary = [
            SearchResult("d1", "T1", "C1", 1.0)
        ]
        secondary = [
            SearchResult("d2", "T2", "C2", 3.0)
        ]
        merged = SearchSwarm._merge_results(
            primary, secondary, top_k=10
        )
        assert merged[0].score >= merged[1].score
