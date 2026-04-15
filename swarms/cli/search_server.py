"""
Self-hosted Search Engine REST API server.

Provides HTTP endpoints for the SearchSwarm search engine.
Uses only Python standard library (http.server) — no extra
dependencies required.

Endpoints:
    POST /search          - Execute a search query
    POST /index/file      - Index a single file
    POST /index/directory - Index a directory of files
    GET  /stats           - Get index statistics
    GET  /health          - Health check

Usage:
    python -m swarms.cli.search_server --port 8000 \\
        --docs /path/to/documents

    Or programmatically:
        from swarms.cli.search_server import run_search_server
        run_search_server(port=8000, docs_dir="/path/to/docs")
"""

import argparse
import json
import sys
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

from swarms.structs.search_swarm import SearchSwarm
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="search_server")


def _json_response(
    handler: BaseHTTPRequestHandler,
    data: Dict[str, Any],
    status: int = 200,
) -> None:
    """Send a JSON response."""
    body = json.dumps(data, indent=2, default=str).encode(
        "utf-8"
    )
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header(
        "Access-Control-Allow-Origin", "*"
    )
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(
    handler: BaseHTTPRequestHandler,
) -> Optional[Dict[str, Any]]:
    """Read and parse a JSON request body."""
    length = int(
        handler.headers.get("Content-Length", 0)
    )
    if length == 0:
        return None
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


class SearchRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the search engine API."""

    # Set by the factory function
    swarm: SearchSwarm

    def log_message(
        self, format: str, *args: Any
    ) -> None:
        """Route HTTP logs through loguru."""
        logger.info(format % args)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header(
            "Access-Control-Allow-Origin", "*"
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            _json_response(
                self, {"status": "ok"}
            )
        elif self.path == "/stats":
            stats = self.swarm.get_stats()
            _json_response(self, stats)
        elif self.path == "/":
            _json_response(
                self,
                {
                    "service": "SearchSwarm API",
                    "version": "1.0.0",
                    "endpoints": {
                        "POST /search": (
                            "Search documents"
                        ),
                        "POST /index/file": (
                            "Index a file"
                        ),
                        "POST /index/directory": (
                            "Index a directory"
                        ),
                        "GET /stats": (
                            "Index statistics"
                        ),
                        "GET /health": "Health check",
                    },
                },
            )
        else:
            _json_response(
                self,
                {"error": "Not found"},
                status=404,
            )

    def do_POST(self) -> None:
        """Handle POST requests."""
        try:
            body = _read_json_body(self)
        except (json.JSONDecodeError, ValueError) as exc:
            _json_response(
                self,
                {"error": f"Invalid JSON: {exc}"},
                status=400,
            )
            return

        if self.path == "/search":
            self._handle_search(body)
        elif self.path == "/index/file":
            self._handle_index_file(body)
        elif self.path == "/index/directory":
            self._handle_index_directory(body)
        else:
            _json_response(
                self,
                {"error": "Not found"},
                status=404,
            )

    def _handle_search(
        self, body: Optional[Dict[str, Any]]
    ) -> None:
        """Handle search requests."""
        if not body or "query" not in body:
            _json_response(
                self,
                {"error": "Missing 'query' field"},
                status=400,
            )
            return

        query = body["query"]
        top_k = body.get("top_k", 10)

        results = self.swarm.search(query, top_k=top_k)
        _json_response(
            self,
            {
                "query": query,
                "result_count": len(results),
                "results": [
                    r.to_dict() for r in results
                ],
            },
        )

    def _handle_index_file(
        self, body: Optional[Dict[str, Any]]
    ) -> None:
        """Handle file indexing requests."""
        if not body or "filepath" not in body:
            _json_response(
                self,
                {"error": "Missing 'filepath' field"},
                status=400,
            )
            return

        filepath = body["filepath"]
        success = self.swarm.index_file(filepath)
        _json_response(
            self,
            {
                "filepath": filepath,
                "indexed": success,
            },
        )

    def _handle_index_directory(
        self, body: Optional[Dict[str, Any]]
    ) -> None:
        """Handle directory indexing requests."""
        if not body or "directory" not in body:
            _json_response(
                self,
                {"error": "Missing 'directory' field"},
                status=400,
            )
            return

        directory = body["directory"]
        recursive = body.get("recursive", True)
        count = self.swarm.index_directory(
            directory, recursive=recursive
        )
        _json_response(
            self,
            {
                "directory": directory,
                "files_indexed": count,
            },
        )


def _make_handler_class(
    swarm: SearchSwarm,
) -> type:
    """Create a handler class bound to a SearchSwarm instance."""

    class BoundHandler(SearchRequestHandler):
        pass

    BoundHandler.swarm = swarm
    return BoundHandler


def run_search_server(
    port: int = 8000,
    host: str = "0.0.0.0",
    docs_dir: Optional[str] = None,
    persist_path: Optional[str] = None,
    model_name: str = "ollama/llama3",
    agent_mode: bool = False,
) -> None:
    """Start the search engine HTTP server.

    Args:
        port: Port to listen on.
        host: Host address to bind to.
        docs_dir: Directory of documents to index on startup.
        persist_path: Path to persist/load the document index.
        model_name: LLM model name for agent mode.
        agent_mode: Enable AI agent query processing.
    """
    swarm = SearchSwarm(
        name="SearchServer",
        documents_directory=docs_dir,
        model_name=model_name,
        persist_path=persist_path,
        agent_mode=agent_mode,
    )

    handler_class = _make_handler_class(swarm)
    server = HTTPServer((host, port), handler_class)

    stats = swarm.get_stats()
    logger.info(
        f"Search server starting on http://{host}:{port}"
    )
    logger.info(
        f"Documents indexed: {stats['document_count']}"
    )
    logger.info(
        f"Agent mode: {'enabled' if agent_mode else 'disabled'}"
    )

    print(
        f"\n  SearchSwarm API Server"
        f"\n  ====================="
        f"\n  URL:       http://{host}:{port}"
        f"\n  Documents: {stats['document_count']}"
        f"\n  Agent Mode: {agent_mode}"
        f"\n\n  Endpoints:"
        f"\n    POST /search          - Search documents"
        f"\n    POST /index/file      - Index a file"
        f"\n    POST /index/directory - Index a directory"
        f"\n    GET  /stats           - Index statistics"
        f"\n    GET  /health          - Health check"
        f"\n\n  Press Ctrl+C to stop.\n"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down search server")
        if persist_path:
            swarm.save_index()
            logger.info("Index saved")
        server.server_close()


def main() -> None:
    """CLI entry point for the search server."""
    parser = argparse.ArgumentParser(
        description="SearchSwarm API Server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--docs",
        dest="docs_dir",
        help="Directory of documents to index",
    )
    parser.add_argument(
        "--persist",
        dest="persist_path",
        help="Path to persist the document index",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default="ollama/llama3",
        help=(
            "LLM model name "
            "(default: ollama/llama3)"
        ),
    )
    parser.add_argument(
        "--agent-mode",
        action="store_true",
        help="Enable AI agent query processing",
    )

    args = parser.parse_args()
    run_search_server(
        port=args.port,
        host=args.host,
        docs_dir=args.docs_dir,
        persist_path=args.persist_path,
        model_name=args.model_name,
        agent_mode=args.agent_mode,
    )


if __name__ == "__main__":
    main()
