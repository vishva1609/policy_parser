"""
Automated question generation using Mistral API.
"""

from typing import List, Optional
from dataclasses import dataclass
import os
import json
import time
from pathlib import Path

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("WARNING: mistralai not installed. Install with: pip install mistralai")

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
    """Generate compliance questions using Mistral API."""

    def __init__(self, api_key: str = None, model: str = "mistral-small-latest"):
        if not MISTRAL_AVAILABLE:
            raise ImportError("mistralai package required. Install: pip install mistralai")

        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set")

        self.model_name = model
        self.client = Mistral(api_key=self.api_key)

        print(f"Initialized question generator with model: {model}")

    # ---------------------------
    # Mistral API Call
    # ---------------------------
    def _call_mistral(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Mistral API with retry logic."""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048,
                )

                return response.choices[0].message.content

            except Exception as e:
                print(f"Mistral API error (attempt {attempt+1}): {str(e)[:150]}")

                # Exponential backoff for rate limits
                wait_time = 5 * (2 ** attempt)
                time.sleep(wait_time)

        return None

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
        print(f"Model: {self.model_name}")
        print(f"Statements: {len(statements)}")
        print(f"Batch size: {batch_size}")

        for i in range(0, len(statements), batch_size):
            batch = statements[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\nProcessing batch {batch_num}...")

            prompt = self._create_batch_prompt(batch, questions_per_statement)

            response_text = self._call_mistral(prompt)

            if response_text:
                questions = self._parse_response(response_text, batch)
                all_questions.extend(questions)
                print(f"Generated {len(questions)} questions.")
            else:
                print("Failed to generate questions for this batch.")

            time.sleep(1)  # small delay to avoid rate limits

        print(f"\nTotal questions generated: {len(all_questions)}")
        return all_questions

    # ---------------------------
    # Prompt Builder
    # ---------------------------
    def _create_batch_prompt(
        self,
        statements: List[PolicyStatement],
        questions_per_statement: int
    ) -> str:

        prompt = f"""
You are a compliance auditor analyzing policy documents.

For each policy statement below, generate {questions_per_statement} specific compliance questions.

Question categories:
- implementation
- audit
- requirement
- verification

POLICY STATEMENTS:
"""

        for i, stmt in enumerate(statements, 1):
            prompt += f"\n{i}. [{stmt.category}] {stmt.text}\n"

        prompt += """
Respond ONLY in this JSON format:

[
  {
    "statement_id": "statement text",
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

        return prompt

    # ---------------------------
    # Response Parser
    # ---------------------------
    def _parse_response(
        self,
        response_text: str,
        statements: List[PolicyStatement]
    ) -> List[ComplianceQuestion]:

        questions: List[ComplianceQuestion] = []

        try:
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1

            if json_start == -1 or json_end == -1:
                return []

            data = json.loads(response_text[json_start:json_end])

            for item in data:
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
                            source_statement=stmt_text[:300],
                            rationale=q_data.get("rationale", ""),
                            page=matching_stmt.page if matching_stmt else 0,
                            section=matching_stmt.section if matching_stmt else ""
                        )
                    )

        except Exception as e:
            print(f"Failed to parse response: {e}")

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
