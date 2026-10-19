"""MCP server exposing Phase 1 read-only tools."""

from __future__ import annotations

from typing import Optional

from .file_ops import glob_files, grep_files, list_dir, read_file, stat_path
from .git_tools import git_diff, git_status


def create_server() -> object:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the `mcp` package to run the File-Ops MCP server.") from exc

    mcp = FastMCP("chat-software-file-ops")

    @mcp.tool()
    def list_directory(path: str = ".", roots: Optional[list[str]] = None) -> dict:
        return list_dir(path=path, roots=roots)

    @mcp.tool()
    def stat(path: str, roots: Optional[list[str]] = None) -> dict:
        return stat_path(path=path, roots=roots)

    @mcp.tool()
    def read(path: str, roots: Optional[list[str]] = None, max_bytes: int = 200000) -> dict:
        return read_file(path=path, roots=roots, max_bytes=max_bytes)

    @mcp.tool()
    def glob(pattern: str, root: str = ".", roots: Optional[list[str]] = None) -> dict:
        return glob_files(pattern=pattern, root=root, roots=roots)

    @mcp.tool()
    def grep(query: str, root: str = ".", roots: Optional[list[str]] = None) -> dict:
        return grep_files(query=query, root=root, roots=roots)

    @mcp.tool()
    def git_status_tool(repo_path: str = ".", roots: Optional[list[str]] = None) -> dict:
        return git_status(repo_path=repo_path, roots=roots)

    @mcp.tool()
    def git_diff_tool(repo_path: str = ".", roots: Optional[list[str]] = None) -> dict:
        return git_diff(repo_path=repo_path, roots=roots)

    return mcp


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
