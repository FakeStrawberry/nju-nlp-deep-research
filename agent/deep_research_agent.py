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
6. Prefer the best-supported specific answer with low confidence over "Insufficient evidence" when at least one concrete candidate is supported by retrieved evidence.

Final answer format:
Explanation: <brief evidence-based reasoning>
Exact Answer: <short final answer only>
Confidence: <0-100>%
Evidence: <docid list or short citations>
"""


FINALIZER_PROMPT = """Stop searching now.
Based only on the evidence already collected in the conversation, produce the best final answer.
If the evidence is insufficient, say so clearly.
Prefer a best-supported specific answer with low confidence over "Insufficient evidence" when the evidence contains a concrete candidate.

Use exactly this format:
Explanation: <brief evidence-based reasoning>
Exact Answer: <short final answer only>
Confidence: <0-100>%
Evidence: <docid list or short citations>
"""


VERIFIER_PROMPT = """Verify the candidate final answer before submission.

Use only the existing conversation evidence and the verification search results just added.
Do not use outside knowledge and do not invent missing evidence.

Rules:
- If the candidate is directly supported, keep it.
- If the candidate is a ticker, abbreviation, or shorthand, normalize it to the full entity name when the evidence supports that normalization.
- If another short answer is better supported by the evidence, replace the candidate with that answer.
- Use "Insufficient evidence" only when no specific answer is supported by the evidence.
- Prefer a low-confidence specific answer over "Insufficient evidence" when retrieved evidence contains a plausible name, title, organization, number, date, country, or phrase.
- Keep Exact Answer short: a name, title, organization, number, date, country, or phrase.

Use exactly this format:
Explanation: <brief evidence-based verification>
Exact Answer: <short final answer only>
Confidence: <0-100>%
Evidence: <docid list or short citations>
"""


SPECIAL_TOKEN_RE = re.compile(r"\[unused\d+\]\s*")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
EXACT_ANSWER_RE = re.compile(
    r"(?:Exact Answer|Final Answer|Answer)\s*:\s*(.+?)(?:\n[A-Z][A-Za-z ]{1,30}\s*:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
QUERY_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")

BM25_QUERY_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "also",
    "an",
    "another",
    "answer",
    "around",
    "author",
    "based",
    "became",
    "because",
    "before",
    "being",
    "between",
    "certain",
    "could",
    "during",
    "first",
    "following",
    "given",
    "having",
    "identify",
    "inclusive",
    "information",
    "later",
    "looking",
    "mentioned",
    "name",
    "other",
    "particular",
    "published",
    "question",
    "second",
    "should",
    "specific",
    "submitted",
    "their",
    "there",
    "they",
    "these",
    "third",
    "those",
    "through",
    "under",
    "what",
    "when",
    "which",
    "while",
    "whose",
    "would",
    "worked",
    "written",
}

ENTITY_CONNECTORS = {
    "and",
    "at",
    "by",
    "da",
    "de",
    "del",
    "der",
    "di",
    "du",
    "for",
    "in",
    "la",
    "le",
    "of",
    "on",
    "the",
    "to",
    "van",
    "von",
}


def clean_model_text(text: Any) -> str:
    if text is None:
        return ""
    cleaned = SPECIAL_TOKEN_RE.sub("", str(text))
    cleaned = THINK_BLOCK_RE.sub("", cleaned)
    cleaned = THINK_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def has_exact_answer(text: str) -> bool:
    return bool(EXACT_ANSWER_RE.search(clean_model_text(text)))


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


def is_plausible_short_answer(answer: str) -> bool:
    answer = clean_model_text(answer)
    if not answer:
        return False
    if len(answer) > 160:
        return False
    bad_markers = (
        "wait,",
        "i'm not sure",
        "not sure",
        "maybe",
        "perhaps",
        "the information is not available",
        "unable to identify",
    )
    lowered = answer.lower()
    return not any(marker in lowered for marker in bad_markers)


def is_insufficient_answer(answer: str) -> bool:
    lowered = clean_model_text(answer).lower()
    markers = (
        "insufficient evidence",
        "unable to determine",
        "cannot determine",
        "not enough information",
        "not available",
        "not found",
    )
    return any(marker in lowered for marker in markers)


def extract_json_array(text: str) -> Optional[List[Any]]:
    cleaned = clean_model_text(text)
    start = cleaned.find("[")
    while start != -1:
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            start = cleaned.find("[", start + 1)
            continue
        if isinstance(parsed, list):
            return parsed
        start = cleaned.find("[", start + 1)
    return None


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
    adaptive_searches_used: int = 0

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
            "adaptive_searches_used": self.adaptive_searches_used,
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
        max_tool_result_chars: int = 0,
        top_k: int = 5,
        bootstrap_search: bool = True,
        bootstrap_query_count: int = 4,
        auto_open_top_n: int = 1,
        min_tool_calls: int = 3,
        max_no_new_info_rounds: int = 2,
        verify_final_answer: bool = True,
        verification_top_k: int = 5,
        verification_open_top_n: int = 1,
        adaptive_query_count: int = 1,
        max_adaptive_searches: int = 2,
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
        self.bootstrap_query_count = bootstrap_query_count
        self.auto_open_top_n = auto_open_top_n
        self.min_tool_calls = min_tool_calls
        self.max_no_new_info_rounds = max_no_new_info_rounds
        self.verify_final_answer = verify_final_answer
        self.verification_top_k = verification_top_k
        self.verification_open_top_n = verification_open_top_n
        self.adaptive_query_count = adaptive_query_count
        self.max_adaptive_searches = max_adaptive_searches

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
                    if state.tool_call_count < self.min_tool_calls:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Continue researching before answering. You have not used enough evidence yet. "
                                    "Use search/open_doc/find_in_doc to verify at least one concrete clue, then answer."
                                ),
                            }
                        )
                        continue
                    if not has_exact_answer(final_text) or not is_plausible_short_answer(extract_exact_answer(final_text)):
                        final_text = self._force_final_answer(messages, state)
                    if self._should_adapt_before_final(final_text, state):
                        if self._append_adaptive_searches(
                            messages,
                            state,
                            question,
                            prefix=f"round{round_id}_pre_final",
                        ):
                            continue
                    final_text = self._maybe_verify_final_answer(question, final_text, messages, state)
                    if self._should_adapt_before_final(final_text, state):
                        if self._append_adaptive_searches(
                            messages,
                            state,
                            question,
                            prefix=f"round{round_id}_post_verify",
                        ):
                            continue
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
                    self._append_tool_result(messages, tool_call["id"], result)
                    if tool_name == "search":
                        if self._append_auto_open_calls(messages, state, result, prefix=f"round{round_id}"):
                            round_has_new_info = True

                if not round_has_new_info:
                    round_has_new_info = self._append_adaptive_searches(
                        messages,
                        state,
                        question,
                        prefix=f"round{round_id}_no_info",
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
            final_text = self._maybe_verify_final_answer(question, final_text, messages, state)
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
        queries = self._build_initial_queries(question)
        for index, query in enumerate(queries, start=1):
            self._append_search_call(messages, state, query=query, call_id=f"bootstrap_search_{index}")

    def _build_initial_queries(self, question: str) -> List[str]:
        queries: List[str] = []
        for query in self._bm25_aware_search_queries(question):
            self._add_query(queries, query)
        for query in self._llm_search_plan(question):
            self._add_query(queries, self._compress_bm25_query(query))
        self._add_query(queries, question)
        return queries[: max(1, self.bootstrap_query_count)]

    def _llm_search_plan(self, question: str) -> List[str]:
        prompt = (
            "Create concise search queries for a local BM25 corpus. "
            "Use only entities, dates, quoted phrases, and distinctive clues from the question. "
            "Do not answer the question. Return only a JSON array of strings, no markdown.\n\n"
            f"Question:\n{question}"
        )
        try:
            response = self.client.simple_chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You write high-recall BM25 search queries."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=384,
            )
        except Exception:
            return []

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = extract_json_array(content)
        if not parsed:
            return []
        return [str(item).strip() for item in parsed if isinstance(item, str) and item.strip()]

    def _heuristic_search_queries(self, question: str) -> List[str]:
        return self._bm25_aware_search_queries(question)

    def _bm25_aware_search_queries(self, question: str) -> List[str]:
        queries: List[str] = []
        quoted = self._quoted_phrases(question)
        entities = self._capitalized_phrases(question)
        identifiers = self._identifier_terms(question)
        years = re.findall(r"\b(?:1[5-9]\d{2}|20\d{2})\b", question)
        rare_terms = self._distinctive_terms(question, max_terms=14)

        for phrase in quoted[:4]:
            self._add_query(queries, phrase)

        for identifier in identifiers[:4]:
            self._add_query(queries, identifier)

        for entity in entities[:4]:
            self._add_query(queries, entity)

        if identifiers and rare_terms:
            self._add_query(queries, " ".join(identifiers[:3] + rare_terms[:6]))
        if years and rare_terms:
            self._add_query(queries, " ".join(years[:4] + rare_terms[:8]))
        if entities and rare_terms:
            self._add_query(queries, " ".join(entities[:2] + rare_terms[:8]))
        if rare_terms:
            self._add_query(queries, " ".join(rare_terms[:10]))
        return queries

    @staticmethod
    def _quoted_phrases(text: str) -> List[str]:
        phrases = []
        for phrase in re.findall(r"[\"“”]([^\"“”]{2,100})[\"“”]", text):
            phrase = re.sub(r"\s+", " ", phrase).strip(" .;:")
            if phrase and phrase.lower() not in {p.lower() for p in phrases}:
                phrases.append(phrase)
        return phrases

    @staticmethod
    def _identifier_terms(text: str) -> List[str]:
        identifiers = []
        patterns = (
            r"\b[A-Z]{2,}(?:[-_][A-Z0-9]{2,})+\b",
            r"\b[A-Z]{2,}\d+[A-Z0-9-]*\b",
            r"\b[A-Za-z]+-\d+[A-Za-z0-9-]*\b",
            r"\b\d+[A-Za-z]+[A-Za-z0-9-]*\b",
        )
        for pattern in patterns:
            for match in re.findall(pattern, text):
                cleaned = match.strip(" .;:")
                if re.fullmatch(r"\d{3,4}s?", cleaned, flags=re.IGNORECASE):
                    continue
                if cleaned and cleaned.lower() not in {item.lower() for item in identifiers}:
                    identifiers.append(cleaned)
        return identifiers

    @staticmethod
    def _capitalized_phrases(text: str) -> List[str]:
        tokens = QUERY_WORD_RE.findall(text)
        phrases: List[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i].strip("'’")
            lower = token.lower()
            coded_token = bool(
                re.search(r"[A-Za-z]", token)
                and re.search(r"\d", token)
                and not re.fullmatch(r"\d{3,4}s?", token, flags=re.IGNORECASE)
            )
            starts_entity = (
                len(token) > 1
                and lower not in BM25_QUERY_STOPWORDS
                and (token[:1].isupper() or token.isupper() or coded_token)
            )
            if not starts_entity:
                i += 1
                continue

            phrase = [token]
            j = i + 1
            while j < len(tokens) and len(phrase) < 7:
                nxt = tokens[j].strip("'’")
                nxt_lower = nxt.lower()
                if nxt_lower in ENTITY_CONNECTORS:
                    if j + 1 < len(tokens):
                        after = tokens[j + 1].strip("'’")
                        coded_after = bool(
                            re.search(r"[A-Za-z]", after)
                            and re.search(r"\d", after)
                            and not re.fullmatch(r"\d{3,4}s?", after, flags=re.IGNORECASE)
                        )
                        if after.lower() in BM25_QUERY_STOPWORDS:
                            break
                        if after[:1].isupper() or after.isupper() or coded_after:
                            phrase.append(nxt)
                            j += 1
                            continue
                    break
                if nxt_lower in BM25_QUERY_STOPWORDS:
                    break
                if len(nxt) > 1 and (
                    nxt[:1].isupper()
                    or nxt.isupper()
                    or (
                        bool(re.search(r"[A-Za-z]", nxt) and re.search(r"\d", nxt))
                        and not re.fullmatch(r"\d{3,4}s?", nxt, flags=re.IGNORECASE)
                    )
                ):
                    phrase.append(nxt)
                    j += 1
                    continue
                break

            if len(phrase) >= 2:
                query = " ".join(phrase).strip(" .;:")
                if query.lower() not in {existing.lower() for existing in phrases}:
                    phrases.append(query)
            i = max(j, i + 1)
        return phrases

    @staticmethod
    def _distinctive_terms(text: str, max_terms: int = 12) -> List[str]:
        terms: List[str] = []
        for word in QUERY_WORD_RE.findall(text):
            cleaned = word.strip("'’")
            if cleaned.endswith(("'s", "’s")):
                cleaned = cleaned[:-2]
            lower = cleaned.lower()
            if not cleaned or lower in BM25_QUERY_STOPWORDS:
                continue
            if any(ch.isdigit() for ch in cleaned) or len(cleaned) >= 6:
                if lower not in {term.lower() for term in terms}:
                    terms.append(cleaned)
            if len(terms) >= max_terms:
                break
        return terms

    @classmethod
    def _compress_bm25_query(cls, query: str, max_terms: int = 10) -> str:
        quoted = cls._quoted_phrases(query)
        identifiers = cls._identifier_terms(query)
        rare_terms = cls._distinctive_terms(query, max_terms=max_terms)
        pieces: List[str] = []
        for piece in quoted[:2] + identifiers[:3] + rare_terms:
            if piece.lower() not in {existing.lower() for existing in pieces}:
                pieces.append(piece)
        return " ".join(pieces[:max_terms]) if pieces else query

    @staticmethod
    def _add_query(queries: List[str], query: str) -> None:
        query = re.sub(r"\s+", " ", clean_model_text(query)).strip(" .;:")
        if not query:
            return
        lowered = query.lower()
        if lowered not in {existing.lower() for existing in queries}:
            queries.append(query)

    def _append_search_call(self, messages: List[Dict[str, Any]], state: ResearchState, query: str, call_id: str) -> bool:
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json_dumps({"query": query, "top_k": self.top_k}),
            },
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        result = self._execute_tool_call(tool_call)
        has_new_info = state.remember_tool_result("search", {"query": query, "top_k": self.top_k}, result)
        self._append_tool_result(messages, tool_call["id"], result)
        return self._append_auto_open_calls(messages, state, result, prefix=call_id) or has_new_info

    def _append_auto_open_calls(
        self,
        messages: List[Dict[str, Any]],
        state: ResearchState,
        search_result: Any,
        prefix: str,
    ) -> bool:
        if self.auto_open_top_n <= 0 or not isinstance(search_result, list):
            return False
        tool_calls = []
        for item in search_result:
            if len(tool_calls) >= self.auto_open_top_n:
                break
            if not isinstance(item, dict):
                continue
            docid = str(item.get("docid", "")).strip()
            if not docid or f"open_doc:{json_dumps({'docid': docid})}" in state.seen_actions:
                continue
            tool_call = {
                "id": f"{prefix}_open_{len(tool_calls) + 1}",
                "type": "function",
                "function": {
                    "name": "open_doc",
                    "arguments": json_dumps({"docid": docid}),
                },
            }
            tool_calls.append(tool_call)
            state.seen_actions.add(canonical_action(tool_call))

        if not tool_calls:
            return False
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        has_new_info = False
        for tool_call in tool_calls:
            result = self._execute_tool_call(tool_call)
            args = parse_json_object(tool_call.get("function", {}).get("arguments", "{}"))
            if state.remember_tool_result("open_doc", args, result):
                has_new_info = True
            self._append_tool_result(messages, tool_call["id"], result)
        return has_new_info

    def _should_adapt_before_final(self, final_text: str, state: ResearchState) -> bool:
        if self.adaptive_query_count <= 0 or state.adaptive_searches_used >= self.max_adaptive_searches:
            return False
        candidate = extract_exact_answer(final_text)
        if not has_exact_answer(final_text):
            return True
        if not is_plausible_short_answer(candidate):
            return True
        return is_insufficient_answer(candidate)

    def _append_adaptive_searches(
        self,
        messages: List[Dict[str, Any]],
        state: ResearchState,
        question: str,
        prefix: str,
    ) -> bool:
        if self.adaptive_query_count <= 0 or state.adaptive_searches_used >= self.max_adaptive_searches:
            return False

        queries = self._adaptive_search_queries(question, state)
        if not queries:
            return False

        has_new_info = False
        for index, query in enumerate(queries[: self.adaptive_query_count], start=1):
            if state.adaptive_searches_used >= self.max_adaptive_searches:
                break
            state.adaptive_searches_used += 1
            if self._append_search_call(
                messages,
                state,
                query=query,
                call_id=f"{prefix}_adaptive_search_{index}",
            ):
                has_new_info = True
        return has_new_info

    def _adaptive_search_queries(self, question: str, state: ResearchState) -> List[str]:
        queries: List[str] = []
        seen = {query.lower() for query in state.seen_queries}

        for query in self._bm25_aware_search_queries(question):
            if query.lower() not in seen:
                self._add_query(queries, query)

        for prior_query in reversed(state.seen_queries):
            compressed = self._compress_bm25_query(prior_query, max_terms=7)
            if compressed.lower() not in seen:
                self._add_query(queries, compressed)

        if question.lower() not in seen:
            self._add_query(queries, question)
        return queries

    def _append_tool_result(self, messages: List[Dict[str, Any]], tool_call_id: str, result: Any) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": truncate_text(json_dumps(result), self.max_tool_result_chars),
            }
        )

    def _maybe_verify_final_answer(
        self,
        question: str,
        final_text: str,
        messages: List[Dict[str, Any]],
        state: ResearchState,
    ) -> str:
        if not self.verify_final_answer:
            return final_text

        candidate = extract_exact_answer(final_text)
        candidate_is_valid = (
            has_exact_answer(final_text)
            and bool(candidate)
            and is_plausible_short_answer(candidate)
            and not is_insufficient_answer(candidate)
        )
        verification_candidate = candidate if candidate_is_valid else ""

        verification_queries = self._build_verification_queries(question, verification_candidate, state)
        for index, query in enumerate(verification_queries, start=1):
            self._append_verification_search(messages, state, query, call_id=f"verification_search_{index}")

        candidate_for_prompt = verification_candidate or "No valid candidate answer was produced."
        messages.append(
            {
                "role": "user",
                "content": (
                    VERIFIER_PROMPT
                    + "\n\nOriginal question:\n"
                    + question
                    + "\n\nCandidate final answer:\n"
                    + candidate_for_prompt
                    + "\n\n"
                    + state.to_prompt()
                ),
            }
        )
        response = self.client.simple_chat(
            model=self.model,
            messages=self._context_messages(messages, state),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw_message = response["choices"][0]["message"]
        assistant_message, _ = normalize_assistant_message(raw_message, "verify")
        messages.append(assistant_message)
        verified_text = assistant_message.get("content", "")
        if has_exact_answer(verified_text):
            return verified_text
        return final_text

    def _build_verification_queries(self, question: str, candidate: str, state: ResearchState) -> List[str]:
        queries: List[str] = []
        candidate = clean_model_text(candidate)
        if candidate and not is_insufficient_answer(candidate):
            self._add_query(queries, candidate)

        for prior_query in state.seen_queries[:3]:
            if candidate and not is_insufficient_answer(candidate):
                self._add_query(queries, f"{candidate} {prior_query}")
            elif prior_query:
                self._add_query(queries, prior_query)
            if len(queries) >= 2:
                break

        if not queries:
            self._add_query(queries, question)
        return queries[:2]

    def _append_verification_search(
        self,
        messages: List[Dict[str, Any]],
        state: ResearchState,
        query: str,
        call_id: str,
    ) -> None:
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json_dumps({"query": query, "top_k": self.verification_top_k}),
            },
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        result = self._execute_tool_call(tool_call)
        state.remember_tool_result("search", {"query": query, "top_k": self.verification_top_k}, result)
        self._append_tool_result(messages, tool_call["id"], result)

        old_auto_open_top_n = self.auto_open_top_n
        try:
            self.auto_open_top_n = self.verification_open_top_n
            self._append_auto_open_calls(messages, state, result, prefix=call_id)
        finally:
            self.auto_open_top_n = old_auto_open_top_n

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
        bootstrap_query_count=args.bootstrap_query_count,
        auto_open_top_n=args.auto_open_top_n,
        min_tool_calls=args.min_tool_calls,
        max_no_new_info_rounds=args.max_no_new_info_rounds,
        verify_final_answer=args.verify_final_answer,
        verification_top_k=args.verification_top_k,
        verification_open_top_n=args.verification_open_top_n,
        adaptive_query_count=args.adaptive_query_count,
        max_adaptive_searches=args.max_adaptive_searches,
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
    parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=0,
        help="Max serialized chars per stored tool result. 0 keeps valid full JSON.",
    )
    parser.add_argument(
        "--bootstrap-query-count",
        type=int,
        default=4,
        help="Number of initial planned search queries before the free-form agent loop.",
    )
    parser.add_argument(
        "--auto-open-top-n",
        type=int,
        default=1,
        help="Automatically open this many top documents after each search.",
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=3,
        help="Do not accept a model final answer before this many tool calls.",
    )
    parser.add_argument(
        "--max-no-new-info-rounds",
        type=int,
        default=2,
        help="Force final answer after this many rounds without new docs or notes.",
    )
    parser.add_argument(
        "--verification-top-k",
        type=int,
        default=5,
        help="Top-k for final candidate answer verification searches.",
    )
    parser.add_argument(
        "--verification-open-top-n",
        type=int,
        default=1,
        help="Automatically open this many top verification search results.",
    )
    parser.add_argument(
        "--adaptive-query-count",
        type=int,
        default=1,
        help="Run this many deterministic fallback searches when the agent stalls or wants to answer with insufficient evidence.",
    )
    parser.add_argument(
        "--max-adaptive-searches",
        type=int,
        default=2,
        help="Maximum deterministic fallback searches per question.",
    )
    parser.add_argument(
        "--no-verify-final-answer",
        dest="verify_final_answer",
        action="store_false",
        help="Disable final candidate answer verification.",
    )
    parser.add_argument("--resume", action="store_true", help="Append and skip query_ids already present in output.")
    parser.add_argument(
        "--no-bootstrap-search",
        dest="bootstrap_search",
        action="store_false",
        help="Disable the deterministic first search of the original question.",
    )
    parser.set_defaults(bootstrap_search=True)
    parser.set_defaults(verify_final_answer=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_submission(args)


if __name__ == "__main__":
    main(sys.argv[1:])
