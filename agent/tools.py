import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .browsecomp_searcher import BrowseCompBM25Searcher, snippetize


QUERY_SNIPPET_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "another",
    "because",
    "before",
    "being",
    "between",
    "could",
    "during",
    "first",
    "following",
    "given",
    "identify",
    "including",
    "later",
    "mentioned",
    "other",
    "question",
    "second",
    "should",
    "something",
    "their",
    "there",
    "these",
    "third",
    "those",
    "through",
    "under",
    "which",
    "while",
    "would",
}


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)


def _query_terms(query: str) -> List[str]:
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", query.lower()):
        token = token.strip("'’")
        if token.endswith(("'s", "’s")):
            token = token[:-2]
        if not token:
            continue
        if token in QUERY_SNIPPET_STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token) or len(token) >= 4:
            if token not in terms:
                terms.append(token)
    return terms


def _document_lead(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    frontmatter_end = text.find("\n---", 4) if text.startswith("---\n") else -1
    if frontmatter_end != -1:
        next_break = text.find("\n\n", frontmatter_end + 4)
        end = next_break if next_break != -1 else frontmatter_end + 4
        return snippetize(text[:end].strip(), max_chars)
    first_break = text.find("\n\n")
    if first_break != -1 and first_break < max_chars:
        return text[:first_break].strip()
    return snippetize(text, max_chars)


def _slice_around(text: str, center: int, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    start = max(0, center - max_chars // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)

    if start > 0:
        whitespace = text.find(" ", start, min(end, start + 80))
        if whitespace != -1:
            start = whitespace + 1
    if end < len(text):
        whitespace = text.rfind(" ", max(start, end - 80), end)
        if whitespace != -1:
            end = whitespace

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def query_aware_snippet(text: str, query: str, max_chars: int = 1200) -> str:
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text

    terms = _query_terms(query)
    if not terms:
        return snippetize(text, max_chars)

    lower_text = text.lower()
    candidates: List[int] = []
    for term in terms[:20]:
        start = 0
        seen = 0
        while seen < 8:
            pos = lower_text.find(term, start)
            if pos == -1:
                break
            candidates.append(pos + len(term) // 2)
            start = pos + len(term)
            seen += 1

    if not candidates:
        return snippetize(text, max_chars)

    window = max(500, max_chars)
    best_center = candidates[0]
    best_score = -1
    for center in candidates:
        start = max(0, center - window // 2)
        end = min(len(text), center + window // 2)
        chunk = lower_text[start:end]
        score = 0
        for term in terms:
            if term in chunk:
                score += 3 if any(ch.isdigit() for ch in term) else 1
        if score > best_score:
            best_score = score
            best_center = center

    lead_budget = min(360, max_chars // 3)
    lead = _document_lead(text, lead_budget)
    if lead and best_center > len(lead) + 200 and max_chars >= 700:
        passage_budget = max_chars - len(lead) - 8
        passage = _slice_around(text, best_center, passage_budget)
        return f"{lead.rstrip()}\n...\n{passage}"

    return _slice_around(text, best_center, max_chars)


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
            "snippet": query_aware_snippet(doc["text"], query=query, max_chars=snippet_max_chars),
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
