"""
SearchSwarm: AI swarm-based search orchestration.

A self-hosted search engine built on the Swarms framework that
orchestrates multiple agents to process queries, search local
document indices, and synthesize results.

Operates entirely locally with no paid API dependencies.
Compatible with free/local LLM providers (e.g., Ollama) via
LiteLLM.

Example:
    >>> from swarms.structs.search_swarm import SearchSwarm
    >>> swarm = SearchSwarm(
    ...     documents_directory="/path/to/docs",
    ...     model_name="ollama/llama3",
    ... )
    >>> results = swarm.run("What is machine learning?")
"""

import uuid
from typing import Any, Dict, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.tools.search_engine import (
    DocumentStore,
    SearchResult,
    create_search_tool_functions,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="search_swarm")

QUERY_ANALYZER_PROMPT = """\
You are a Query Analyzer agent in a search engine system.
Your role is to analyze user search queries and produce
improved, expanded queries for better search results.

Given a user query, you should:
1. Identify the core intent and key concepts.
2. Expand abbreviations or ambiguous terms.
3. Suggest related terms that could improve recall.
4. Output a refined query string optimized for a
   text-based search engine.

Respond ONLY with the refined query text. Do not include
explanations or formatting.
"""

SYNTHESIZER_PROMPT = """\
You are a Search Results Synthesizer agent. Your role is
to take raw search results and produce a clear, well-organized
summary for the user.

Given the original query and search results, you should:
1. Analyze the relevance of each result.
2. Extract and consolidate key information.
3. Present a coherent, well-structured answer.
4. Cite the source documents when possible.
5. If results are insufficient, clearly state that.

Be concise, accurate, and helpful.
"""


class SearchSwarm:
    """AI swarm-based local search engine.

    Orchestrates agents to analyze queries, search a local
    document index, and synthesize results. Fully self-hosted
    with no paid API dependencies.

    Args:
        name: Name of the search swarm.
        description: Description of the swarm.
        documents_directory: Path to directory of documents
            to index on initialization.
        document_store: Pre-configured DocumentStore instance.
            If not provided, a new one is created.
        model_name: LLM model to use for agents. Use local
            models like ``"ollama/llama3"`` for free operation.
        max_loops: Maximum agent reasoning loops.
        persist_path: Path to persist the document index.
        agent_mode: When True, uses LLM agents for query
            refinement and result synthesis. When False, performs
            direct search only (no LLM required).

    Example:
        >>> swarm = SearchSwarm(
        ...     documents_directory="./my_docs",
        ...     model_name="ollama/llama3",
        ... )
        >>> result = swarm.run("What is deep learning?")
        >>> print(result)
    """

    def __init__(
        self,
        name: str = "SearchSwarm",
        description: str = "AI swarm-based local search engine",
        documents_directory: Optional[str] = None,
        document_store: Optional[DocumentStore] = None,
        model_name: str = "ollama/llama3",
        max_loops: int = 1,
        persist_path: Optional[str] = None,
        agent_mode: bool = True,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.model_name = model_name
        self.max_loops = max_loops
        self.agent_mode = agent_mode
        self.persist_path = persist_path

        # Initialize document store
        self.document_store = document_store or DocumentStore(
            name=f"{name}_store",
            persist_path=persist_path,
        )

        # Index initial documents if directory provided
        if documents_directory:
            self.document_store.index_directory(
                documents_directory
            )

        # Initialize conversation history
        self.conversation = Conversation(
            name=f"{name}_conversation"
        )

        # Create search tool functions
        self._tools = create_search_tool_functions(
            self.document_store
        )

        # Initialize agents if in agent mode
        self._query_agent: Optional[Agent] = None
        self._synthesizer_agent: Optional[Agent] = None

        if self.agent_mode:
            self._init_agents()

        logger.info(
            f"SearchSwarm '{name}' initialized "
            f"(agent_mode={agent_mode}, "
            f"docs={self.document_store.document_count})"
        )

    def _init_agents(self) -> None:
        """Initialize the query analyzer and synthesizer agents."""
        self._query_agent = Agent(
            agent_name="QueryAnalyzer",
            agent_description=(
                "Analyzes and refines search queries "
                "for better results."
            ),
            system_prompt=QUERY_ANALYZER_PROMPT,
            model_name=self.model_name,
            max_loops=self.max_loops,
            verbose=False,
            streaming_on=False,
        )

        self._synthesizer_agent = Agent(
            agent_name="ResultSynthesizer",
            agent_description=(
                "Synthesizes search results into "
                "clear, organized responses."
            ),
            system_prompt=SYNTHESIZER_PROMPT,
            model_name=self.model_name,
            max_loops=self.max_loops,
            verbose=False,
            streaming_on=False,
        )

    def run(
        self,
        query: str,
        top_k: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Execute a search query through the swarm.

        Args:
            query: The user's search query.
            top_k: Maximum number of results to retrieve.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Search results as a formatted string, or a
            dictionary when raw results are requested.
        """
        logger.info(f"Processing query: {query!r}")
        self.conversation.add(role="User", content=query)

        # Step 1: Refine query (agent mode) or use as-is
        refined_query = query
        if self.agent_mode and self._query_agent:
            try:
                refined_query = self._query_agent.run(
                    f"Refine this search query: {query}"
                )
                logger.info(
                    f"Refined query: {refined_query!r}"
                )
                self.conversation.add(
                    role="QueryAnalyzer",
                    content=f"Refined query: {refined_query}",
                )
            except Exception as exc:
                logger.warning(
                    f"Query refinement failed, using "
                    f"original query: {exc}"
                )
                refined_query = query

        # Step 2: Search the document store
        results = self.document_store.search(
            refined_query, top_k=top_k
        )

        # Also search with original query and merge
        if refined_query != query:
            original_results = self.document_store.search(
                query, top_k=top_k
            )
            results = self._merge_results(
                results, original_results, top_k
            )

        if not results:
            no_results_msg = (
                f"No results found for: {query!r}"
            )
            self.conversation.add(
                role="SearchSwarm", content=no_results_msg
            )
            return no_results_msg

        # Format raw results
        raw_output = self._format_results(results)
        self.conversation.add(
            role="SearchEngine",
            content=raw_output,
        )

        # Step 3: Synthesize results (agent mode)
        if self.agent_mode and self._synthesizer_agent:
            try:
                synthesis_prompt = (
                    f"Original query: {query}\n\n"
                    f"Search results:\n{raw_output}\n\n"
                    f"Please synthesize these results into "
                    f"a clear, helpful response."
                )
                synthesized = self._synthesizer_agent.run(
                    synthesis_prompt
                )
                self.conversation.add(
                    role="ResultSynthesizer",
                    content=synthesized,
                )
                return synthesized
            except Exception as exc:
                logger.warning(
                    f"Synthesis failed, returning raw "
                    f"results: {exc}"
                )

        return raw_output

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Direct search without agent processing.

        Args:
            query: Search query string.
            top_k: Maximum results to return.

        Returns:
            List of SearchResult objects.
        """
        return self.document_store.search(
            query, top_k=top_k
        )

    def index_file(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Index a single file.

        Args:
            filepath: Path to the file.
            metadata: Optional metadata.

        Returns:
            True if successfully indexed.
        """
        return self.document_store.index_file(
            filepath, metadata=metadata
        )

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> int:
        """Index all supported files in a directory.

        Args:
            directory: Path to directory.
            recursive: Include subdirectories.

        Returns:
            Count of indexed files.
        """
        return self.document_store.index_directory(
            directory, recursive=recursive
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get search swarm statistics.

        Returns:
            Dictionary with swarm and index statistics.
        """
        store_stats = self.document_store.get_stats()
        return {
            "swarm_id": self.id,
            "swarm_name": self.name,
            "agent_mode": self.agent_mode,
            "model_name": self.model_name,
            **store_stats,
        }

    def save_index(
        self, filepath: Optional[str] = None
    ) -> None:
        """Save the document index to disk.

        Args:
            filepath: Path to save. Uses persist_path if omitted.
        """
        self.document_store.save(filepath)

    @staticmethod
    def _merge_results(
        primary: List[SearchResult],
        secondary: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Merge two result lists, removing duplicates."""
        seen = {r.doc_id for r in primary}
        merged = list(primary)
        for r in secondary:
            if r.doc_id not in seen:
                merged.append(r)
                seen.add(r.doc_id)
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    @staticmethod
    def _format_results(
        results: List[SearchResult],
    ) -> str:
        """Format search results as a readable string."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"--- Result {i} "
                f"(Score: {r.score:.4f}) ---\n"
                f"Title: {r.title}\n"
                f"Content: {r.content}\n"
            )
        return "\n".join(parts)
