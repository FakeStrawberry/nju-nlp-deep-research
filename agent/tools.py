import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def retrieve_once(
    searcher: BrowseCompBM25Searcher,
    query: str,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    docs = searcher.search(query, k=k)
    return [
        {
            "docid": doc["docid"],
            "score": doc["score"],
            "snippet": snippetize(doc["text"], snippet_max_chars),
            "url": doc.get("url", ""),
        }
        for doc in docs
    ]


def open_doc(
    searcher: BrowseCompBM25Searcher,
    docid: str,
    max_chars: int = 5000,
) -> Dict[str, Any]:
    doc = searcher.get_document(str(docid))
    if doc is None:
        return {"docid": str(docid), "error": "document not found"}
    text = doc.get("text", "")
    return {
        "docid": doc["docid"],
        "url": doc.get("url", ""),
        "text": snippetize(text, max_chars),
        "truncated": bool(max_chars and max_chars > 0 and len(text) > max_chars),
        "full_text_chars": len(text),
    }


def find_in_doc(
    searcher: BrowseCompBM25Searcher,
    docid: str,
    keyword: str,
    window_chars: int = 500,
    max_matches: int = 5,
) -> Dict[str, Any]:
    doc = searcher.get_document(str(docid))
    if doc is None:
        return {"docid": str(docid), "keyword": keyword, "error": "document not found"}

    keyword = str(keyword).strip()
    if not keyword:
        return {"docid": str(docid), "keyword": keyword, "error": "empty keyword"}

    text = doc.get("text", "")
    pattern = re.compile(re.escape(keyword), flags=re.IGNORECASE)
    matches = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - window_chars)
        end = min(len(text), match.end() + window_chars)
        matches.append(
            {
                "start": match.start(),
                "end": match.end(),
                "snippet": text[start:end].strip(),
            }
        )
        if len(matches) >= max_matches:
            break

    return {
        "docid": doc["docid"],
        "url": doc.get("url", ""),
        "keyword": keyword,
        "num_matches_returned": len(matches),
        "matches": matches,
    }


def format_rag_context(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for rank, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"docid: {item['docid']}",
                    f"score: {item['score']}",
                    f"url: {item.get('url', '')}",
                    item["snippet"],
                ]
            )
        )
    return "\n\n".join(blocks)


def get_search_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str) -> List[Dict[str, Any]]:
        return retrieve_once(searcher=searcher, query=query, k=k, snippet_max_chars=snippet_max_chars)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    f"Search the BrowseComp-Plus BM25 index and return top-{k} results "
                    "with docid, score, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    return tools, {"search": search}


def get_deep_research_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
    open_doc_max_chars: int = 5000,
    find_window_chars: int = 500,
    max_find_matches: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            requested_k = k if top_k is None else max(1, min(int(top_k), 20))
        except (TypeError, ValueError):
            requested_k = k
        return retrieve_once(
            searcher=searcher,
            query=query,
            k=requested_k,
            snippet_max_chars=snippet_max_chars,
        )

    def open_doc_tool(docid: str) -> Dict[str, Any]:
        return open_doc(searcher=searcher, docid=docid, max_chars=open_doc_max_chars)

    def get_document(docid: str) -> Dict[str, Any]:
        return open_doc_tool(docid)

    def find_in_doc_tool(docid: str, keyword: str) -> Dict[str, Any]:
        return find_in_doc(
            searcher=searcher,
            docid=docid,
            keyword=keyword,
            window_chars=find_window_chars,
            max_matches=max_find_matches,
        )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the BrowseComp-Plus BM25 index and return ranked results "
                    "with docid, score, snippet, and url. Use this to explore a new clue."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {
                            "type": "integer",
                            "description": f"Optional number of results to return. Default is {k}; max is 20.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_doc",
                "description": "Open a document by docid and return a truncated full-text view.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                    },
                    "required": ["docid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_doc",
                "description": "Find exact keyword occurrences inside one document and return local context windows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document id"},
                        "keyword": {
                            "type": "string",
                            "description": "Case-insensitive keyword or phrase to locate in the document.",
                        },
                    },
                    "required": ["docid", "keyword"],
                },
            },
        },
    ]
    return tools, {
        "search": search,
        "open_doc": open_doc_tool,
        "find_in_doc": find_in_doc_tool,
        "get_document": get_document,
    }


def get_agent_tool_specs_and_registry(
    searcher: BrowseCompBM25Searcher,
    k: int = 5,
    snippet_max_chars: int = 1200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    return get_deep_research_tool_specs_and_registry(
        searcher=searcher,
        k=k,
        snippet_max_chars=snippet_max_chars,
    )
