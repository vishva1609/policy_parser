"""
Automated compliance question generation using LangChain + Mistral.
"""

from typing import List, Optional
from dataclasses import dataclass
import os
import json
import time
from pathlib import Path

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("WARNING: langchain-mistralai not installed. Install with: pip install langchain-mistralai")

from .policy_analyzer import PolicyStatement


@dataclass
class ComplianceQuestion:
    """Generated compliance question."""
    question: str
    category: str
    question_type: str
    expected_evidence: str
    source_statement: str
    rationale: str
    page: int
    section: str


class QuestionGenerator:
    """Generate compliance questions using LangChain + Mistral."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 1.0
    ):
        """
        Initialize question generator.

        Args:
            api_key: Mistral API key (falls back to MISTRAL_API_KEY env var)
            model: Mistral model name (mistral-small-latest, mistral-medium-latest, mistral-large-latest)
            temperature: Controls randomness (0.0=deterministic, 1.0=creative). Default 0.3.
            max_tokens: Maximum response length in tokens (~4 chars each). Default 2048.
            top_p: Nucleus sampling — only considers tokens in top_p probability mass (0.0-1.0). Default 1.0.
                   Don't change both temperature and top_p at the same time.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-mistralai required. Install: pip install langchain-mistralai")

        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set")

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.llm = ChatMistralAI(
            model=model,
            mistral_api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a compliance auditor analyzing policy documents. "
                "Generate specific, actionable compliance questions for each policy statement. "
                "Respond ONLY with valid JSON, no extra text."
            )),
            ("human", "{input}")
        ])

        self.chain = self.prompt | self.llm | self.parser

        print(f"Initialized LangChain question generator")
        print(f"  Model: {model} | Temperature: {temperature} | Max Tokens: {max_tokens} | Top P: {top_p}")

    # ---------------------------
    # Main Question Generator
    # ---------------------------
    def generate_questions_from_statements(
        self,
        statements: List[PolicyStatement],
        questions_per_statement: int = 2,
        batch_size: int = 5
    ) -> List[ComplianceQuestion]:

        if not statements:
            return []

        all_questions: List[ComplianceQuestion] = []

        print("\nGenerating compliance questions...")
        print(f"Model: {self.model_name} | Temperature: {self.temperature} | Max Tokens: {self.max_tokens} | Top P: {self.top_p}")
        print(f"Statements: {len(statements)}")
        print(f"Batch size: {batch_size}")

        for i in range(0, len(statements), batch_size):
            batch = statements[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\nProcessing batch {batch_num}...")

            user_input = self._build_user_input(batch, questions_per_statement)
            questions = self._invoke_chain(user_input, batch)
            all_questions.extend(questions)
            print(f"Generated {len(questions)} questions.")

            time.sleep(1)

        print(f"\nTotal questions generated: {len(all_questions)}")
        return all_questions

    # ---------------------------
    # Build User Input
    # ---------------------------
    def _build_user_input(
        self,
        statements: List[PolicyStatement],
        questions_per_statement: int
    ) -> str:

        text = f"For each policy statement below, generate {questions_per_statement} specific compliance questions.\n\n"
        text += "Question types: implementation, audit, requirement, verification\n\n"
        text += "POLICY STATEMENTS:\n"

        for i, stmt in enumerate(statements, 1):
            text += f"\n{i}. [{stmt.category}] {stmt.text}\n"

        text += """
Respond in this JSON format. IMPORTANT: statement_number must match the number of the statement above (1, 2, 3, etc.):

[
  {
    "statement_number": 1,
    "questions": [
      {
        "question": "question text",
        "type": "implementation|audit|requirement|verification",
        "rationale": "why this question matters"
      }
    ]
  }
]
"""
        return text

    # ---------------------------
    # Invoke LangChain Chain
    # ---------------------------
    def _invoke_chain(
        self,
        user_input: str,
        statements: List[PolicyStatement],
        max_retries: int = 3
    ) -> List[ComplianceQuestion]:

        for attempt in range(max_retries):
            try:
                data = self.chain.invoke({"input": user_input})
                return self._parse_data(data, statements)

            except Exception as e:
                print(f"LangChain error (attempt {attempt + 1}): {str(e)[:150]}")
                wait_time = 5 * (2 ** attempt)
                time.sleep(wait_time)

        print("Failed to generate questions for this batch.")
        return []

    # ---------------------------
    # Parse Structured Output
    # ---------------------------
    def _parse_data(
        self,
        data,
        statements: List[PolicyStatement]
    ) -> List[ComplianceQuestion]:

        questions: List[ComplianceQuestion] = []

        if not isinstance(data, list):
            return questions

        for item in data:
            # Match by numeric index (1-based from prompt)
            stmt_num = item.get("statement_number", 0)
            if 1 <= stmt_num <= len(statements):
                matching_stmt = statements[stmt_num - 1]
            else:
                # Fallback: try text matching
                stmt_text = item.get("statement_id", "")
                matching_stmt = next(
                    (s for s in statements if s.text in stmt_text or stmt_text in s.text),
                    None
                )

            for q_data in item.get("questions", []):
                questions.append(
                    ComplianceQuestion(
                        question=q_data.get("question", ""),
                        question_type=q_data.get("type", "requirement"),
                        category=matching_stmt.category.value if matching_stmt else "General",
                        expected_evidence="",
                        source_statement=matching_stmt.text[:300] if matching_stmt else "",
                        rationale=q_data.get("rationale", ""),
                        page=matching_stmt.page if matching_stmt else 0,
                        section=matching_stmt.section if matching_stmt else ""
                    )
                )

        return questions

    # ---------------------------
    # Export Questions
    # ---------------------------
    def export_questions(self, questions: List[ComplianceQuestion]) -> dict:
        """Export questions to dictionary format."""
        data = {
            "total_questions": len(questions),
            "questions_by_type": {},
            "questions_by_category": {},
            "questions": [vars(q) for q in questions]
        }

        for q in questions:
            data["questions_by_type"][q.question_type] = \
                data["questions_by_type"].get(q.question_type, 0) + 1

            data["questions_by_category"][q.category] = \
                data["questions_by_category"].get(q.category, 0) + 1

        return data

    # ---------------------------
    # Save to File
    # ---------------------------
    def save_questions(
        self,
        questions: List[ComplianceQuestion],
        output_path: str
    ):
        """Save questions to JSON file."""
        data = self.export_questions(questions)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(questions)} questions to {output_path}")
