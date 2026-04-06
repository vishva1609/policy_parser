"""
Clause Comparator — Semantic matching + LLM compliance reasoning + gap detection.

KEY OOP IMPROVEMENT: Dependency Injection
    The reference repo (qa_apis.py) creates the LLM config ONCE in the views layer
    and INJECTS it into the service, which passes it to the model:

        # views/qa_apis.py:
        embeddings = OpenAIEmbeddings()         ← created once
        chat = ChatOpenAI(...)                  ← created once
        response = process_request(request, embeddings, chat)  ← injected

    We mirror this: ComplianceReasoner now ACCEPTS a pre-built MistralConfig
    instead of creating its own:

        # views/pipeline_runner.py:
        config = MistralConfig(api_key=..., model=...)   ← created once
        reasoner = ComplianceReasoner(config=config)     ← injected

Components:
    ClauseMatcher       — top-k semantic search via ChromaDB (injected embedder)
    ComplianceReasoner  — Mistral LLM judgment (accepts injected MistralConfig)
    GapDetector         — rules engine combining similarity + LLM output
"""
import os
import time
from typing import List, Optional

from ..common.schemas import PolicyClause, ClauseMatch, ComplianceResult, ComplianceStatus
from .embedder import EmbeddingPipeline
from ..common.llm_client import MistralConfig, LANGCHAIN_AVAILABLE

try:
    from langchain_core.output_parsers import JsonOutputParser
except ImportError:
    pass

# ── Thresholds ─────────────────────────────────────────────────────────────────
MISSING_THRESHOLD = 0.30
GAP_THRESHOLD     = 0.50


# ── ClauseMatcher ──────────────────────────────────────────────────────────────

class ClauseMatcher:
    """
    Retrieves top-k semantically similar clauses from the *target* policy
    using the target policy's EmbeddingPipeline (ChromaDB cosine search).
    """

    def __init__(self, target_embedder: EmbeddingPipeline):
        self.embedder = target_embedder

    def find_matches(self, clause: PolicyClause, top_k: int = 5) -> List[ClauseMatch]:
        query = f"{clause.title} {clause.text}"[:1000]
        try:
            results = self.embedder.search(query, n_results=min(top_k, 10))
        except Exception as e:
            print(f"    [Matcher] Search error: {str(e)[:80]}")
            return []

        docs      = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        ids       = results.get('ids', [[]])[0]

        matches: List[ClauseMatch] = []
        for doc, dist, meta, cid in zip(docs, distances, metadatas, ids):
            similarity = max(0.0, 1.0 - float(dist))

            section = meta.get('section_title', '')
            if not section:
                ctx = meta.get('parent_context', '')
                section = ctx[0] if isinstance(ctx, list) else str(ctx)

            page = meta.get('page', 0)
            if not page:
                pages = meta.get('pages', [0])
                page  = pages[0] if isinstance(pages, list) and pages else 0

            filename     = meta.get('filename', '')
            clause_title = meta.get('clause_title', '')
            policy_label = meta.get('policy_label', '')

            matches.append(ClauseMatch(
                text=doc,
                clause_id=meta.get('clause_id', str(cid)),
                similarity_score=round(similarity, 4),
                section=section,
                page=page,
                filename=filename,
                clause_title=clause_title,
                policy_label=policy_label,
                file_id=str(meta.get('file_id', '') or ''),
                sub_clause_id=str(meta.get('sub_clause_id', '') or ''),
                parent_clause_id=str(meta.get('parent_clause_id', '') or ''),
                metadata=meta,
            ))

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches


# ── ComplianceReasoner ─────────────────────────────────────────────────────────

class ComplianceReasoner:
    """
    Uses Mistral LLM to determine whether the target policy adequately
    covers a source policy requirement.

    DEPENDENCY INJECTION: Accepts an optional pre-built MistralConfig.
    This mirrors how qa_apis.py injects `chat` into process_request:

        # qa_apis.py (reference repo):
        chat = ChatOpenAI(...)                      ← created once in views
        response = process_request(request, embeddings, chat)  ← injected

        # Our views/pipeline_runner.py:
        self.config = MistralConfig(api_key=..., model=...)   ← created once
        reasoner = ComplianceReasoner(config=self.config)      ← injected

    Falls back to creating its own MistralConfig if none is injected
    (backward compatibility).
    """

    _SYSTEM = (
        "You are an expert compliance auditor comparing two security/privacy policies. "
        "Determine whether the target policy adequately covers the source requirement. "
        "Pay attention to specific standards (e.g. AES-256, TLS 1.2), thresholds, "
        "and mandatory vs. advisory language (must vs. should). "
        "Respond ONLY with valid JSON."
    )

    _PROMPT = """\
[SOURCE POLICY — requirement to satisfy]
Title : {source_title}
Text  : \"{source_text}\"

[TARGET POLICY — candidates found via semantic search]
{candidates_block}

Does the target policy adequately cover the source requirement?

Return JSON exactly:
{{
  "status": "Compliant" | "Partial" | "Non-Compliant" | "Missing",
  "reason": "<2-3 sentence explanation>",
  "gap": "<specific missing requirement, or empty string if Compliant>",
  "best_candidate_index": <1..{n} or 0 if none relevant>,
  "confidence": <0.0..1.0>,
  "evidence_quote": "<most relevant verbatim quote from target, or empty string>"
}}

Definitions:
  Compliant     — target fully satisfies the source requirement
  Partial       — target addresses the topic but omits specifics
  Non-Compliant — target contradicts or clearly falls short of the source
  Missing       — no relevant clause exists in the target at all
"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        config: "MistralConfig" = None,
    ):
        """
        Initialise the reasoner.

        Preferred usage (dependency injection — mirrors reference repo):
            config = MistralConfig(api_key=..., model=...)
            reasoner = ComplianceReasoner(config=config)

        Legacy usage (creates its own config — backward compatible):
            reasoner = ComplianceReasoner(api_key=..., model=...)

        Args:
            api_key:     Mistral API key (only used if config is None).
            model:       Model name (only used if config is None).
            temperature: Temperature (only used if config is None).
            max_tokens:  Max tokens (only used if config is None).
            config:      Pre-built MistralConfig (preferred — dependency injection).
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("Install langchain-mistralai: pip install langchain-mistralai")

        # Use injected config if provided; otherwise create one (backward compat)
        if config is not None:
            _config = config
        else:
            _config = MistralConfig(
                api_key=api_key or os.getenv("MISTRAL_API_KEY"),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # get_comparison_chain mirrors get_qa_chain(vector_db) in OpenAIConfig
        self.chain = _config.get_comparison_chain(
            embedder=None,           # no retriever needed — we handle retrieval separately
            system_prompt=self._SYSTEM,
        )
        self.model_name = _config.model_name
        print(f"  ComplianceReasoner ready: {self.model_name}")

    def compare(
        self,
        source_clause: PolicyClause,
        candidates: List[ClauseMatch],
        max_retries: int = 3,
    ) -> dict:
        if not candidates:
            return {
                "status": "Missing",
                "reason": f"No relevant clauses found in target policy for: {source_clause.title}",
                "gap": f"Target policy has no clause addressing: {source_clause.title}",
                "best_candidate_index": 0,
                "confidence": 0.95,
                "evidence_quote": "",
            }

        candidates_block = ""
        for i, c in enumerate(candidates[:5], 1):
            snippet = c.text[:400] + ("…" if len(c.text) > 400 else "")
            candidates_block += (
                f"{i}. [Section: {c.section} | Similarity: {c.similarity_score:.2f}]\n"
                f'   "{snippet}"\n\n'
            )

        input_text = self._PROMPT.format(
            source_title=source_clause.title[:120],
            source_text=source_clause.text[:600],
            candidates_block=candidates_block.strip(),
            n=len(candidates[:5]),
        )

        for attempt in range(max_retries):
            try:
                raw = self.chain.invoke({"input": input_text})
                return self._normalize(raw)
            except Exception as e:
                print(f"    [Reasoner] attempt {attempt+1} error: {str(e)[:100]}")
                time.sleep(5 * (2 ** attempt))

        return {
            "status": "Partial",
            "reason": "LLM unavailable; similarity score used as proxy.",
            "gap": "",
            "best_candidate_index": 1,
            "confidence": 0.3,
            "evidence_quote": candidates[0].text[:200] if candidates else "",
        }

    @staticmethod
    def _normalize(raw: dict) -> dict:
        valid = {"Compliant", "Partial", "Non-Compliant", "Missing"}
        status = raw.get("status", "Partial")
        if status not in valid:
            status = "Partial"
        try:
            idx = int(raw.get("best_candidate_index", 1))
        except (ValueError, TypeError):
            idx = 1
        try:
            conf = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
        except (ValueError, TypeError):
            conf = 0.5
        return {
            "status": status,
            "reason": str(raw.get("reason", ""))[:500],
            "gap": str(raw.get("gap", ""))[:300],
            "best_candidate_index": idx,
            "confidence": conf,
            "evidence_quote": str(raw.get("evidence_quote", ""))[:300],
        }


# ── GapDetector ────────────────────────────────────────────────────────────────

class GapDetector:
    """
    Rules engine combining similarity thresholds + LLM judgement
    to produce a final ComplianceResult.

    Priority rules (evaluated in order):
        1. No candidates OR max similarity < missing_threshold  → Missing
        2. LLM explicitly says "Missing"                        → Missing
        3. max similarity < gap_threshold AND LLM "Non-Compliant" → Non-Compliant
        4. Otherwise use LLM status directly
    """

    def __init__(
        self,
        missing_threshold: Optional[float] = None,
        gap_threshold: Optional[float] = None,
    ):
        self.missing_threshold = (
            MISSING_THRESHOLD if missing_threshold is None else missing_threshold
        )
        self.gap_threshold = GAP_THRESHOLD if gap_threshold is None else gap_threshold

    def detect(
        self,
        source_clause: PolicyClause,
        candidates: List[ClauseMatch],
        llm_result: dict,
        direction: str = "A→B",
    ) -> ComplianceResult:
        max_sim = max((c.similarity_score for c in candidates), default=0.0)
        best_match: Optional[ClauseMatch] = candidates[0] if candidates else None

        llm_status = llm_result.get("status", "Partial")
        reason     = llm_result.get("reason", "")
        gap        = llm_result.get("gap", "")
        evidence   = llm_result.get("evidence_quote", "")
        confidence = llm_result.get("confidence", 0.5)

        if max_sim < self.missing_threshold or not candidates:
            status = ComplianceStatus.MISSING
            reason = f"No relevant clause found (max similarity: {max_sim:.2f}). {reason}".strip()
            gap    = gap or f"Missing coverage for: {source_clause.title}"
            evidence = ""
        elif llm_status == "Missing":
            status = ComplianceStatus.MISSING
        elif max_sim < self.gap_threshold and llm_status == "Non-Compliant":
            status = ComplianceStatus.NON_COMPLIANT
        else:
            _map = {
                "Compliant":     ComplianceStatus.COMPLIANT,
                "Partial":       ComplianceStatus.PARTIAL,
                "Non-Compliant": ComplianceStatus.NON_COMPLIANT,
                "Missing":       ComplianceStatus.MISSING,
            }
            status = _map.get(llm_status, ComplianceStatus.PARTIAL)

        idx = llm_result.get("best_candidate_index", 1)
        if idx and 1 <= idx <= len(candidates):
            best_match = candidates[idx - 1]

        return ComplianceResult(
            clause_a=source_clause,
            best_match=best_match,
            all_candidates=candidates,
            status=status,
            reason=reason,
            gap_description=gap,
            confidence=confidence,
            evidence_quote=evidence,
            direction=direction,
        )
