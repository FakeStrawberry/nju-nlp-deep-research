import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .dataset_utils import load_jsonl
from .tools import build_searcher, get_deep_research_tool_specs_and_registry
from .vllm_client import VLLMClient


SYSTEM_PROMPT = """You are a Deep Research Agent for BrowseComp-Plus.

You must answer complex questions using only the local BrowseComp-Plus corpus tools.
Do not use external search engines, web APIs, or unsupported background knowledge.

Available tools:
- search(query, top_k): search the BM25 corpus for a clue or entity.
- open_doc(docid): inspect a retrieved document.
- find_in_doc(docid, keyword): locate a phrase inside a document.

Workflow:
1. Search before giving a final answer.
2. Track confirmed facts, unresolved clues, and document ids.
3. If evidence is insufficient, reformulate the query and continue.
4. Avoid repeated searches that return the same documents unless you are checking a specific clue.
5. Stop when the answer is supported, the search budget is exhausted, or new searches add no new information.

Final answer format:
Explanation: <brief evidence-based reasoning>
Exact Answer: <short final answer only>
Confidence: <0-100>%
Evidence: <docid list or short citations>
"""


FINALIZER_PROMPT = """Stop searching now.
Based only on the evidence already collected in the conversation, produce the best final answer.
If the evidence is insufficient, say so clearly.

Use exactly this format:
Explanation: <brief evidence-based reasoning>
Exact Answer: <short final answer only>
Confidence: <0-100>%
Evidence: <docid list or short citations>
"""


SPECIAL_TOKEN_RE = re.compile(r"\[unused\d+\]\s*")
EXACT_ANSWER_RE = re.compile(
    r"(?:Exact Answer|Final Answer|Answer)\s*:\s*(.+?)(?:\n[A-Z][A-Za-z ]{1,30}\s*:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_model_text(text: Any) -> str:
    if text is None:
        return ""
    return SPECIAL_TOKEN_RE.sub("", str(text)).strip()


def extract_exact_answer(text: str) -> str:
    cleaned = clean_model_text(text)
    match = EXACT_ANSWER_RE.search(cleaned)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r"\s+", " ", answer)
        return answer.strip("`\"' ")

    non_empty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if non_empty_lines:
        return non_empty_lines[-1].strip("`\"' ")
    return cleaned


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": parsed}
    return {"_raw": value}


def normalize_tool_call(tool_call: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    name = function.get("name") or tool_call.get("name") or ""
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, dict):
        arguments = json_dumps(arguments)
    elif arguments is None:
        arguments = "{}"

    return {
        "id": str(tool_call.get("id") or fallback_id),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": str(name),
            "arguments": str(arguments),
        },
    }


def normalize_assistant_message(message: Dict[str, Any], fallback_prefix: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    content = clean_model_text(message.get("content", ""))
    normalized: Dict[str, Any] = {"role": "assistant", "content": content}
    if message.get("reasoning_content"):
        normalized["reasoning_content"] = clean_model_text(message.get("reasoning_content"))

    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = [
        normalize_tool_call(tool_call, fallback_id=f"{fallback_prefix}_{idx}")
        for idx, tool_call in enumerate(raw_tool_calls, start=1)
    ]
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized, tool_calls


def canonical_action(tool_call: Dict[str, Any]) -> str:
    function = tool_call.get("function", {})
    name = function.get("name", "")
    args = parse_json_object(function.get("arguments", "{}"))
    return f"{name}:{json_dumps(args)}"


@dataclass
class ResearchState:
    seen_actions: set[str] = field(default_factory=set)
    seen_queries: List[str] = field(default_factory=list)
    seen_docids: set[str] = field(default_factory=set)
    evidence_notes: List[str] = field(default_factory=list)
    tool_call_count: int = 0
    no_new_info_rounds: int = 0

    def remember_tool_result(self, tool_name: str, args: Dict[str, Any], result: Any) -> bool:
        self.tool_call_count += 1
        before_docs = set(self.seen_docids)
        before_notes = len(self.evidence_notes)

        if tool_name == "search":
            query = str(args.get("query", "")).strip()
            if query and query not in self.seen_queries:
                self.seen_queries.append(query)
            if isinstance(result, list):
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    docid = str(item.get("docid", "")).strip()
                    if docid:
                        self.seen_docids.add(docid)
                    snippet = clean_model_text(item.get("snippet", ""))
                    if docid and snippet:
                        self._add_note(f"doc {docid}: {truncate_text(snippet, 260)}")
        elif tool_name in {"open_doc", "get_document"} and isinstance(result, dict):
            docid = str(result.get("docid", args.get("docid", ""))).strip()
            if docid:
                self.seen_docids.add(docid)
            text = clean_model_text(result.get("text", ""))
            if docid and text:
                self._add_note(f"opened doc {docid}: {truncate_text(text, 320)}")
        elif tool_name == "find_in_doc" and isinstance(result, dict):
            docid = str(result.get("docid", args.get("docid", ""))).strip()
            if docid:
                self.seen_docids.add(docid)
            matches = result.get("matches") or []
            for match in matches[:3]:
                if isinstance(match, dict):
                    snippet = clean_model_text(match.get("snippet", ""))
                    if docid and snippet:
                        self._add_note(f"match in doc {docid}: {truncate_text(snippet, 260)}")

        return before_docs != self.seen_docids or before_notes != len(self.evidence_notes)

    def _add_note(self, note: str) -> None:
        if note not in self.evidence_notes:
            self.evidence_notes.append(note)
        if len(self.evidence_notes) > 18:
            self.evidence_notes = self.evidence_notes[-18:]

    def to_prompt(self) -> str:
        searches = self.seen_queries[-8:]
        docids = sorted(self.seen_docids)
        notes = self.evidence_notes[-10:]
        return "\n".join(
            [
                "Compact research state:",
                f"- tool calls so far: {self.tool_call_count}",
                f"- recent searches: {json_dumps(searches)}",
                f"- seen docids: {', '.join(docids[:40]) if docids else '(none)'}",
                "- evidence notes:",
                *[f"  - {note}" for note in notes],
            ]
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "tool_call_count": self.tool_call_count,
            "seen_queries": list(self.seen_queries),
            "seen_docids": sorted(self.seen_docids),
            "evidence_notes": list(self.evidence_notes),
            "no_new_info_rounds": self.no_new_info_rounds,
        }


class DeepResearchAgent:
    def __init__(
        self,
        client: VLLMClient,
        model: str,
        tool_specs: List[Dict[str, Any]],
        tool_registry: Dict[str, Callable[..., Any]],
        *,
        max_rounds: int = 6,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_history_messages: int = 18,
        max_tool_result_chars: int = 7000,
        top_k: int = 5,
        bootstrap_search: bool = True,
        max_no_new_info_rounds: int = 2,
    ) -> None:
        self.client = client
        self.model = model
        self.tool_specs = tool_specs
        self.tool_registry = tool_registry
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_history_messages = max_history_messages
        self.max_tool_result_chars = max_tool_result_chars
        self.top_k = top_k
        self.bootstrap_search = bootstrap_search
        self.max_no_new_info_rounds = max_no_new_info_rounds

    def answer(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        state = ResearchState()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        if self.bootstrap_search:
            self._append_bootstrap_search(messages, state, question)

        stop_reason = "completed"
        try:
            for round_id in range(1, self.max_rounds + 1):
                response = self.client.simple_chat(
                    model=self.model,
                    messages=self._context_messages(messages, state),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.tool_specs,
                    tool_choice="auto",
                )
                raw_message = response["choices"][0]["message"]
                assistant_message, tool_calls = normalize_assistant_message(raw_message, f"round{round_id}_call")
                messages.append(assistant_message)

                if not tool_calls:
                    final_text = assistant_message.get("content", "")
                    return self._build_record(
                        query_id=query_id,
                        question=question,
                        status="completed",
                        predicted_answer=extract_exact_answer(final_text),
                        messages=messages,
                        state=state,
                        stop_reason="model_final_answer",
                    )

                round_has_new_info = False
                for tool_call in tool_calls:
                    action = canonical_action(tool_call)
                    state.seen_actions.add(action)
                    result = self._execute_tool_call(tool_call)
                    tool_name = tool_call.get("function", {}).get("name", "")
                    args = parse_json_object(tool_call.get("function", {}).get("arguments", "{}"))
                    if state.remember_tool_result(tool_name, args, result):
                        round_has_new_info = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": truncate_text(json_dumps(result), self.max_tool_result_chars),
                        }
                    )

                if round_has_new_info:
                    state.no_new_info_rounds = 0
                else:
                    state.no_new_info_rounds += 1
                if state.no_new_info_rounds >= self.max_no_new_info_rounds:
                    stop_reason = "no_new_information"
                    break

            if stop_reason == "completed":
                stop_reason = "max_rounds_reached"
            final_text = self._force_final_answer(messages, state)
            return self._build_record(
                query_id=query_id,
                question=question,
                status="completed" if final_text else stop_reason,
                predicted_answer=extract_exact_answer(final_text),
                messages=messages,
                state=state,
                stop_reason=stop_reason,
            )
        except Exception as exc:
            error_text = f"Agent failed: {type(exc).__name__}: {exc}"
            messages.append({"role": "assistant", "content": error_text})
            return self._build_record(
                query_id=query_id,
                question=question,
                status="error",
                predicted_answer="",
                messages=messages,
                state=state,
                stop_reason="exception",
                error=error_text,
            )

    def _append_bootstrap_search(self, messages: List[Dict[str, Any]], state: ResearchState, question: str) -> None:
        tool_call = {
            "id": "bootstrap_search_1",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json_dumps({"query": question, "top_k": self.top_k}),
            },
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        result = self._execute_tool_call(tool_call)
        state.remember_tool_result("search", {"query": question, "top_k": self.top_k}, result)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": truncate_text(json_dumps(result), self.max_tool_result_chars),
            }
        )

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        args = parse_json_object(function.get("arguments", "{}"))
        args = self._normalize_tool_args(name, args)
        tool = self.tool_registry.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}", "available_tools": sorted(self.tool_registry)}
        try:
            return tool(**args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "tool": name, "arguments": args}

    def _normalize_tool_args(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(args)
        if "doc_id" in args and "docid" not in args:
            args["docid"] = args.pop("doc_id")
        if "document_id" in args and "docid" not in args:
            args["docid"] = args.pop("document_id")
        if name == "search":
            if "q" in args and "query" not in args:
                args["query"] = args.pop("q")
            if "k" in args and "top_k" not in args:
                args["top_k"] = args.pop("k")
            args.pop("_raw", None)
        if name == "find_in_doc":
            for alias in ("phrase", "term", "query"):
                if alias in args and "keyword" not in args:
                    args["keyword"] = args.pop(alias)
                    break
        if name in {"open_doc", "get_document", "find_in_doc"}:
            args.pop("_raw", None)
        return args

    def _context_messages(self, messages: List[Dict[str, Any]], state: ResearchState) -> List[Dict[str, Any]]:
        if len(messages) <= self.max_history_messages + 2:
            return messages

        head = messages[:2]
        tail = list(messages[2:])[-self.max_history_messages :]
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        return head + [{"role": "user", "content": state.to_prompt()}] + tail

    def _force_final_answer(self, messages: List[Dict[str, Any]], state: ResearchState) -> str:
        messages.append({"role": "user", "content": FINALIZER_PROMPT + "\n\n" + state.to_prompt()})
        response = self.client.simple_chat(
            model=self.model,
            messages=self._context_messages(messages, state),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw_message = response["choices"][0]["message"]
        assistant_message, _ = normalize_assistant_message(raw_message, "final")
        messages.append(assistant_message)
        return assistant_message.get("content", "")

    def _build_record(
        self,
        *,
        query_id: Optional[str],
        question: str,
        status: str,
        predicted_answer: str,
        messages: List[Dict[str, Any]],
        state: ResearchState,
        stop_reason: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "query_id": query_id,
            "status": status,
            "predicted_answer": predicted_answer,
            "messages": messages,
            "state_summary": state.to_record(),
            "current_subgoal": "Answer the original BrowseComp-Plus question from local evidence.",
            "next_action_plan": "finished" if status == "completed" else stop_reason,
            "stop_reason": stop_reason,
        }
        if error:
            record["error"] = error
        return record


def existing_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_id = row.get("query_id")
            if query_id is not None:
                ids.add(str(query_id))
    return ids


def iter_rows(dataset_path: str, limit: Optional[int], skip_ids: set[str]) -> Iterable[Dict[str, Any]]:
    rows = load_jsonl(dataset_path, limit=limit)
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if query_id in skip_ids:
            continue
        yield row


def run_submission(args: argparse.Namespace) -> None:
    searcher = build_searcher(args.index_path)
    tool_specs, tool_registry = get_deep_research_tool_specs_and_registry(
        searcher=searcher,
        k=args.top_k,
        snippet_max_chars=args.snippet_max_chars,
        open_doc_max_chars=args.open_doc_max_chars,
        find_window_chars=args.find_window_chars,
        max_find_matches=args.max_find_matches,
    )
    client = VLLMClient(base_url=args.base_url, api_key=args.api_key)
    agent = DeepResearchAgent(
        client=client,
        model=args.model,
        tool_specs=tool_specs,
        tool_registry=tool_registry,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_history_messages=args.max_history_messages,
        max_tool_result_chars=args.max_tool_result_chars,
        top_k=args.top_k,
        bootstrap_search=args.bootstrap_search,
        max_no_new_info_rounds=args.max_no_new_info_rounds,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skip_ids = existing_query_ids(output_path) if args.resume else set()
    rows = list(iter_rows(args.dataset, args.limit, skip_ids))
    mode = "a" if args.resume else "w"

    with output_path.open(mode, encoding="utf-8") as fout:
        for index, row in enumerate(rows, start=1):
            query_id = str(row.get("query_id", ""))
            question = str(row.get("query", ""))
            print(f"[{index}/{len(rows)}] query_id={query_id}", flush=True)
            record = agent.answer(question, query_id=query_id)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"  status={record['status']} stop={record['stop_reason']} "
                f"answer={truncate_text(record['predicted_answer'], 100)}",
                flush=True,
            )

    print(f"Saved {len(rows)} records to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-round BrowseComp-Plus deep research agent.")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="Input JSONL dataset.")
    parser.add_argument("--index-path", default="indexes/browsecomp_plus_bm25.sqlite", help="SQLite BM25 index path.")
    parser.add_argument("--output", default="runs/submission.jsonl", help="Output submission JSONL path.")
    parser.add_argument("--model", default="qwen_auto", help="Served vLLM model name.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="dummy", help="API key for the vLLM endpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset rows.")
    parser.add_argument("--top-k", type=int, default=5, help="Default top-k for search.")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum model/tool rounds after bootstrap.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for each model response.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--snippet-max-chars", type=int, default=1200, help="Max chars per search snippet.")
    parser.add_argument("--open-doc-max-chars", type=int, default=5000, help="Max chars returned by open_doc.")
    parser.add_argument("--find-window-chars", type=int, default=500, help="Context chars around find_in_doc matches.")
    parser.add_argument("--max-find-matches", type=int, default=5, help="Maximum find_in_doc matches returned.")
    parser.add_argument("--max-history-messages", type=int, default=18, help="Recent messages retained in model context.")
    parser.add_argument("--max-tool-result-chars", type=int, default=7000, help="Max serialized chars per tool result.")
    parser.add_argument(
        "--max-no-new-info-rounds",
        type=int,
        default=2,
        help="Force final answer after this many rounds without new docs or notes.",
    )
    parser.add_argument("--resume", action="store_true", help="Append and skip query_ids already present in output.")
    parser.add_argument(
        "--no-bootstrap-search",
        dest="bootstrap_search",
        action="store_false",
        help="Disable the deterministic first search of the original question.",
    )
    parser.set_defaults(bootstrap_search=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_submission(args)


if __name__ == "__main__":
    main(sys.argv[1:])
