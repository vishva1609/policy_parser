"""
Automated compliance question generation, auto-tuning, and model comparison.

Merges former question_generator.py and model_comparison.py.
Includes bug fixes for:
- Diversity scoring formula (Bug 2)
- AdaptiveTuner.get_optimal_params() missing score (Bug 3)
- AutoTuner._save_results() missing keys (Bug 4)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import os
import json
import time
import itertools
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


# ============================================================
# Data Classes
# ============================================================

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


@dataclass
class ModelResult:
    """Results from a single model evaluation."""
    model_name: str
    num_questions: int
    avg_question_length: float
    question_types: Dict[str, int]
    categories: Dict[str, int]
    generation_time: float
    tokens_used: int  # Approximate
    quality_score: float
    sample_questions: List[str]
    error: Optional[str] = None


@dataclass
class ComparisonReport:
    """Full comparison report across multiple models."""
    timestamp: str
    document_name: str
    num_statements_tested: int
    models_tested: List[str]
    results: List[ModelResult]
    best_model: str
    recommendation: str


# Available models for comparison
MISTRAL_MODELS = {
    "mistral-small-latest": {
        "name": "Mistral Small",
        "speed": "Fast",
        "quality": "Good",
        "cost": "Low",
        "description": "Best for rapid prototyping and cost-sensitive applications"
    },
    "mistral-medium-latest": {
        "name": "Mistral Medium",
        "speed": "Medium",
        "quality": "Better",
        "cost": "Medium",
        "description": "Balanced choice for most production use cases"
    },
    "mistral-large-latest": {
        "name": "Mistral Large",
        "speed": "Slower",
        "quality": "Best",
        "cost": "High",
        "description": "Highest quality for critical compliance applications"
    },
    "open-mistral-7b": {
        "name": "Open Mistral 7B",
        "speed": "Fast",
        "quality": "Good",
        "cost": "Free (open-source)",
        "description": "Open-source option, good for self-hosting"
    },
    "open-mixtral-8x7b": {
        "name": "Open Mixtral 8x7B",
        "speed": "Medium",
        "quality": "Better",
        "cost": "Free (open-source)",
        "description": "MoE architecture, good balance"
    },
    "open-mixtral-8x22b": {
        "name": "Open Mixtral 8x22B",
        "speed": "Slower",
        "quality": "Best (open)",
        "cost": "Free (open-source)",
        "description": "Largest open-source option"
    }
}

# Max number of distinct question types used for diversity scoring
MAX_QUESTION_TYPES = 4


# ============================================================
# Question Generator
# ============================================================

class QuestionGenerator:
    """Generate compliance questions using LangChain + Mistral."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize question generator.

        Args:
            api_key: Mistral API key (falls back to MISTRAL_API_KEY env var)
            model: Mistral model name
            temperature: Controls randomness (0.0=deterministic, 1.0=creative). Default 0.3.
            max_tokens: Maximum response length in tokens. Default 2048.
            top_p: Nucleus sampling threshold (0.0-1.0). Default 1.0.
            frequency_penalty: Penalizes repeated tokens (-2.0 to 2.0). Default 0.0.
            presence_penalty: Penalizes already-present topics (-2.0 to 2.0). Default 0.0.
            seed: Optional random seed for reproducibility.
            top_k: Optional top-k sampling cutoff.
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
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.top_k = top_k

        model_kwargs = {
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if seed is not None:
            model_kwargs["seed"] = seed
        if top_k is not None:
            model_kwargs["top_k"] = top_k

        self.llm = ChatMistralAI(
            model=model,
            mistral_api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model_kwargs=model_kwargs,
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

        print("Initialized LangChain question generator")
        print(
            f"  Model: {model} | Temperature: {temperature} | Max Tokens: {max_tokens} | "
            f"Top P: {top_p} | Freq Penalty: {frequency_penalty} | Presence Penalty: {presence_penalty} | "
            f"Seed: {seed} | Top K: {top_k}"
        )

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
        print(
            f"Model: {self.model_name} | Temperature: {self.temperature} | Max Tokens: {self.max_tokens} | "
            f"Top P: {self.top_p} | Freq Penalty: {self.frequency_penalty} | Presence Penalty: {self.presence_penalty} | "
            f"Seed: {self.seed} | Top K: {self.top_k}"
        )
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


# ============================================================
# Auto Tuner
# ============================================================

class AutoTuner:
    """
    Automatically tune hyperparameters for optimal question generation quality.
    
    This class tests different hyperparameter combinations and scores them
    based on question quality metrics like specificity, coverage, and actionability.
    """
    
    # Default hyperparameter search space
    DEFAULT_SEARCH_SPACE = {
        'temperature': [0.1, 0.2, 0.3, 0.5, 0.7],
        'top_p': [0.9, 0.95, 1.0],
        'max_tokens': [1024, 2048, 4096]
    }
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        search_space: dict = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize auto-tuner.
        
        Args:
            api_key: Mistral API key
            model: Model to use for tuning
            search_space: Custom hyperparameter search space
            frequency_penalty: Frequency penalty forwarded to generator
            presence_penalty: Presence penalty forwarded to generator
            seed: Optional seed forwarded to generator
            top_k: Optional top-k forwarded to generator
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-mistralai required for auto-tuning")
        
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        
        self.model = model
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.top_k = top_k
        
        print(f"AutoTuner initialized with model: {model}")
    
    def tune(
        self,
        sample_chunks: list,
        max_trials: int = 9,
        output_dir: str = "./output"
    ) -> tuple:
        """
        Run hyperparameter tuning on a sample of document chunks.
        
        Args:
            sample_chunks: List of document chunks to test on
            max_trials: Maximum number of parameter combinations to try
            output_dir: Directory to save tuning results
            
        Returns:
            Tuple of (best_params_dict, all_results_list)
        """
        from .policy_analyzer import PolicyAnalyzer
        
        # Extract policy statements for testing
        analyzer = PolicyAnalyzer()
        statements = analyzer.analyze_document(sample_chunks)
        requirements = analyzer.get_requirements(statements, min_confidence=0.5)
        
        if len(requirements) < 3:
            print("Warning: Not enough requirement statements for tuning, using all statements")
            test_statements = statements[:5]
        else:
            test_statements = requirements[:5]  # Use subset for faster tuning
        
        print(f"Using {len(test_statements)} statements for tuning")
        
        results = []
        best_score = -1
        best_params = None
        
        # Generate parameter combinations
        param_combinations = self._generate_combinations(max_trials)
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n[Trial {i}/{len(param_combinations)}] "
                  f"temp={params['temperature']}, top_p={params['top_p']}, tokens={params['max_tokens']}")
            
            try:
                # Generate questions with these parameters
                generator = QuestionGenerator(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    top_p=params['top_p'],
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    seed=self.seed,
                    top_k=self.top_k,
                )
                
                questions = generator.generate_questions_from_statements(
                    test_statements,
                    questions_per_statement=2,
                    batch_size=5
                )
                
                # Score the results
                score = self._score_questions(questions, test_statements)
                
                results.append({
                    **params,
                    'frequency_penalty': self.frequency_penalty,
                    'presence_penalty': self.presence_penalty,
                    'seed': self.seed,
                    'top_k': self.top_k,
                    'score': score,
                    'num_questions': len(questions)
                })
                
                print(f"  Score: {score:.2f} | Questions: {len(questions)}")
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        **params,
                        'frequency_penalty': self.frequency_penalty,
                        'presence_penalty': self.presence_penalty,
                        'seed': self.seed,
                        'top_k': self.top_k,
                        'score': score
                    }
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"  Error: {str(e)[:100]}")
                results.append({**params, 'score': 0, 'error': str(e)[:100]})
        
        # Save tuning results
        self._save_results(results, best_params, output_dir)
        
        return best_params, results
    
    def _generate_combinations(self, max_trials: int) -> list:
        """Generate hyperparameter combinations to test."""
        all_combos = list(itertools.product(
            self.search_space['temperature'],
            self.search_space['top_p'],
            self.search_space['max_tokens']
        ))
        
        # If too many combinations, sample strategically
        if len(all_combos) > max_trials:
            # Include extremes and middle values
            step = len(all_combos) // max_trials
            all_combos = all_combos[::step][:max_trials]
        
        return [
            {'temperature': t, 'top_p': p, 'max_tokens': m}
            for t, p, m in all_combos
        ]
    
    def _score_questions(self, questions, source_statements):
        """Score generated questions based on coverage, specificity, diversity, and actionability."""
        if not questions or not source_statements:
            return 0.0

        score = 0.0

        # Coverage (max 30 points)
        source_texts = {s.text[:300] for s in source_statements}
        covered_texts = {q.source_statement for q in questions if q.source_statement}
        coverage = len(covered_texts & source_texts) / len(source_statements)
        coverage_score = coverage * 30
        score += coverage_score

        # Specificity - word count (max 25 points)
        avg_words = sum(len(q.question.split()) for q in questions) / len(questions)
        specificity_score = min(avg_words / 15, 1.0) * 25
        score += specificity_score

        # Diversity - unique question types (max 20 points)
        # BUG FIX #2: divide by MAX_QUESTION_TYPES (4), not by total question count
        question_types = [q.question_type for q in questions if q.question_type]
        if question_types:
            unique_types = len(set(question_types))
            diversity = unique_types / MAX_QUESTION_TYPES
            diversity_score = min(diversity, 1.0) * 20
            score += diversity_score

        # Actionability (max 25 points)
        action_words = [
            "how", "what", "which", "who", "when", "where",
            "explain", "describe", "verify", "ensure",
            "implement", "document", "demonstrate"
        ]

        actionable = sum(
            1 for q in questions
            if any(word in q.question.lower() for word in action_words)
        )

        actionability_score = (actionable / len(questions)) * 25
        score += actionability_score

        return round(score, 2)

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """Create a filesystem-safe model name for output filenames."""
        safe = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe = safe.replace(" ", "_")
        return "".join(ch for ch in safe if ch.isalnum() or ch in "-_.")

    @classmethod
    def migrate_old_tuning_file(cls, output_dir: str) -> None:
        """
        Migrate old tuning_results.json to new tuning_results_*.json structure.
        
        Old structure: output/tuning_results.json
        New structure: output/tuning/tuning_results_{model}.json
        
        This ensures cached tuning results are found and reused across runs.
        """
        old_path = Path(output_dir) / "tuning_results.json"
        
        if old_path.exists():
            try:
                with open(old_path, 'r') as f:
                    data = json.load(f)
                
                # Only migrate if file has content and model info
                if data and isinstance(data, dict) and data.get('model'):
                    model_name = data.get('model', 'mistral-small-latest')
                    
                    # Create new path and save
                    new_path = cls.get_tuning_output_path(output_dir, model_name)
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(new_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    
                    # Remove old file after successful migration
                    old_path.unlink()
                    print(f"✓ Migrated tuning file from {old_path.name} to {new_path.name}")
                else:
                    # Old file is empty or invalid - just remove it
                    old_path.unlink()
                    print(f"✓ Removed empty old tuning file: {old_path.name}")
            except Exception as e:
                print(f"⚠ Warning: Failed to migrate old tuning file: {str(e)[:80]}")

    @classmethod
    def get_tuning_output_path(cls, output_dir: str, model_name: str) -> Path:
        """Get model-specific tuning results path in output/tuning."""
        # Migrate old tuning file if it exists (one-time operation)
        cls.migrate_old_tuning_file(output_dir)
        
        tuning_dir = Path(output_dir) / "tuning"
        safe_model = cls._sanitize_model_name(model_name)
        return tuning_dir / f"tuning_results_{safe_model}.json"

    def _save_results(self, results: list, best_params: dict, output_dir: str):
        """Save tuning results to model-specific file."""
        output_path = self.get_tuning_output_path(output_dir, self.model)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # BUG FIX #4: Always include model and all hyperparameter keys
        data = {
            'model': self.model,
            'best_params': best_params,
            'all_trials': results,
            'tuning_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"\nTuning results saved to: {output_path}")


# ============================================================
# Adaptive Tuner
# ============================================================

class AdaptiveTuner:
    """
    Dynamic tuning for production deployment.
    
    Loads pre-tuned configs and adapts based on context.
    """
    
    # Pre-tuned configurations for different document types
    PRESET_CONFIGS = {
        'policy': {
            'temperature': 0.2,
            'max_tokens': 2048,
            'top_p': 0.95,
            'description': 'Conservative, focused output for compliance policies'
        },
        'technical': {
            'temperature': 0.3,
            'max_tokens': 3072,
            'top_p': 0.9,
            'description': 'Balanced for technical documentation'
        },
        'general': {
            'temperature': 0.4,
            'max_tokens': 2048,
            'top_p': 1.0,
            'description': 'General-purpose document processing'
        },
        'creative': {
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.95,
            'description': 'More varied, exploratory questions'
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize adaptive tuner.
        
        Args:
            config_path: Path to custom tuning config (JSON)
        """
        self.config_path = config_path
        self.custom_config = None
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.custom_config = json.load(f)
            print(f"Loaded custom tuning config from: {config_path}")
    
    def detect_document_type(self, chunks: list) -> str:
        """
        Detect document type based on content analysis.
        
        Returns one of: 'policy', 'technical', 'general', 'creative'
        """
        # Combine text from first few chunks
        sample_text = ' '.join(c.text[:500] for c in chunks[:5]).lower()
        
        # Policy document indicators
        policy_keywords = ['policy', 'compliance', 'shall', 'must', 'requirement',
                          'regulation', 'standard', 'procedure', 'guideline']
        policy_score = sum(1 for kw in policy_keywords if kw in sample_text)
        
        # Technical document indicators
        tech_keywords = ['api', 'function', 'implementation', 'architecture',
                        'system', 'module', 'interface', 'database', 'code']
        tech_score = sum(1 for kw in tech_keywords if kw in sample_text)
        
        if policy_score >= 3:
            return 'policy'
        elif tech_score >= 3:
            return 'technical'
        else:
            return 'general'
    
    def get_optimal_params(self, chunks: list = None, doc_type: str = None) -> dict:
        """
        Get optimal parameters for the given document.
        
        Args:
            chunks: Document chunks (for auto-detection)
            doc_type: Explicit document type override
            
        Returns:
            Dictionary with optimal hyperparameters (always includes 'score')
        """
        # Use custom config if available (from previous tuning)
        if self.custom_config and 'best_params' in self.custom_config:
            params = self.custom_config['best_params']
            # BUG FIX #3: Include score in returned dict
            return {
                'temperature': params.get('temperature', 0.3),
                'max_tokens': params.get('max_tokens', 2048),
                'top_p': params.get('top_p', 1.0),
                'frequency_penalty': params.get('frequency_penalty', 0.0),
                'presence_penalty': params.get('presence_penalty', 0.0),
                'seed': params.get('seed'),
                'top_k': params.get('top_k'),
                'score': params.get('score', 0.0),
            }
        
        # Detect or use provided document type
        if not doc_type and chunks:
            doc_type = self.detect_document_type(chunks)
            print(f"Detected document type: {doc_type}")
        
        doc_type = doc_type or 'general'
        config = self.PRESET_CONFIGS.get(doc_type, self.PRESET_CONFIGS['general'])
        
        return {
            'temperature': config['temperature'],
            'max_tokens': config['max_tokens'],
            'top_p': config['top_p'],
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'seed': None,
            'top_k': None,
            'score': 0.0,
        }
    
    @staticmethod
    def save_feedback(
        questions: List[ComplianceQuestion],
        feedback: dict,
        output_dir: str = "./output"
    ):
        """Save user feedback for future tuning improvements."""
        feedback_path = Path(output_dir) / "user_feedback.json"
        
        existing = []
        if feedback_path.exists():
            with open(feedback_path, 'r') as f:
                existing = json.load(f)
        
        existing.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_questions': len(questions),
            'feedback': feedback
        })
        
        with open(feedback_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"Feedback saved to: {feedback_path}")


# ============================================================
# Model Comparator (formerly model_comparison.py)
# ============================================================

class ModelComparator:
    """
    Compare different LLM models for compliance question generation.
    
    Evaluates models based on question quality, speed, token efficiency,
    and coverage across categories.
    """
    
    def __init__(
        self,
        api_key: str = None,
        models: List[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize model comparator.
        
        Args:
            api_key: Mistral API key
            models: List of model names to compare
            temperature: Consistent temperature for all models
            max_tokens: Consistent max tokens for all models
            top_p: Consistent top_p for all models
            frequency_penalty: Consistent frequency penalty for all models
            presence_penalty: Consistent presence penalty for all models
            seed: Optional deterministic seed
            top_k: Optional top-k sampling cutoff
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-mistralai required")
        
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        
        self.models = models or ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.top_k = top_k
        
        print(f"ModelComparator initialized")
        print(f"  Models to compare: {', '.join(self.models)}")
    
    def compare(
        self,
        chunks: list,
        questions_per_statement: int = 2,
        max_statements: int = 10,
        output_path: str = None
    ) -> ComparisonReport:
        """
        Run comparison across all specified models.
        
        Args:
            chunks: Document chunks to process
            questions_per_statement: Questions to generate per statement
            max_statements: Maximum statements to test (for speed)
            output_path: Path to save comparison report
            
        Returns:
            ComparisonReport with all results
        """
        from .policy_analyzer import PolicyAnalyzer
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Extract statements for testing
        analyzer = PolicyAnalyzer()
        statements = analyzer.analyze_document(chunks)
        requirements = analyzer.get_requirements(statements, min_confidence=0.5)
        
        test_statements = requirements[:max_statements] if len(requirements) >= max_statements else statements[:max_statements]
        print(f"\nUsing {len(test_statements)} statements for comparison")
        
        results: List[ModelResult] = []
        
        for i, model_name in enumerate(self.models, 1):
            print(f"\n[{i}/{len(self.models)}] Testing: {model_name}")
            print("-" * 50)
            
            result = self._evaluate_model(model_name, test_statements, questions_per_statement)
            results.append(result)
            
            # Rate limiting between models
            if i < len(self.models):
                print("Waiting before next model...")
                time.sleep(3)
        
        # Determine best model
        valid_results = [r for r in results if not r.error]
        if valid_results:
            best_result = max(valid_results, key=lambda r: r.quality_score)
            best_model = best_result.model_name
            recommendation = self._generate_recommendation(results)
        else:
            best_model = "N/A"
            recommendation = "All models failed to generate questions."
        
        # Create report
        report = ComparisonReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            document_name=chunks[0].document if chunks else "Unknown",
            num_statements_tested=len(test_statements),
            models_tested=self.models,
            results=results,
            best_model=best_model,
            recommendation=recommendation
        )
        
        # Print summary
        self._print_summary(report)
        
        # Save report
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _evaluate_model(
        self,
        model_name: str,
        statements: List[PolicyStatement],
        questions_per_statement: int
    ) -> ModelResult:
        """Evaluate a single model."""
        
        start_time = time.time()
        
        try:
            generator = QuestionGenerator(
                api_key=self.api_key,
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                seed=self.seed,
                top_k=self.top_k,
            )
            
            questions = generator.generate_questions_from_statements(
                statements,
                questions_per_statement=questions_per_statement,
                batch_size=5
            )
            
            generation_time = time.time() - start_time
            
            if not questions:
                return ModelResult(
                    model_name=model_name,
                    num_questions=0,
                    avg_question_length=0,
                    question_types={},
                    categories={},
                    generation_time=generation_time,
                    tokens_used=0,
                    quality_score=0,
                    sample_questions=[],
                    error="No questions generated"
                )
            
            # Calculate metrics
            avg_length = sum(len(q.question) for q in questions) / len(questions)
            
            question_types = {}
            for q in questions:
                question_types[q.question_type] = question_types.get(q.question_type, 0) + 1
            
            categories = {}
            for q in questions:
                categories[q.category] = categories.get(q.category, 0) + 1
            
            # Estimate tokens used (rough approximation)
            tokens_used = sum(len(q.question.split()) * 1.3 for q in questions)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(questions, statements)
            
            return ModelResult(
                model_name=model_name,
                num_questions=len(questions),
                avg_question_length=round(avg_length, 1),
                question_types=question_types,
                categories=categories,
                generation_time=round(generation_time, 2),
                tokens_used=int(tokens_used),
                quality_score=round(quality_score, 2),
                sample_questions=[q.question for q in questions[:3]]
            )
            
        except Exception as e:
            return ModelResult(
                model_name=model_name,
                num_questions=0,
                avg_question_length=0,
                question_types={},
                categories={},
                generation_time=time.time() - start_time,
                tokens_used=0,
                quality_score=0,
                sample_questions=[],
                error=str(e)[:200]
            )
    
    def _calculate_quality_score(
        self,
        questions: List[ComplianceQuestion],
        statements: List[PolicyStatement]
    ) -> float:
        """Calculate overall quality score (0-100)."""
        
        if not questions:
            return 0.0
        
        score = 0.0
        
        # Coverage (30 points)
        expected_questions = len(statements) * 2
        coverage = min(len(questions) / expected_questions, 1.0) * 30
        score += coverage
        
        # Specificity - question length (25 points)
        avg_length = sum(len(q.question) for q in questions) / len(questions)
        specificity = min(avg_length / 120, 1.0) * 25
        score += specificity
        
        # Diversity - question types (20 points)
        unique_types = len(set(q.question_type for q in questions))
        diversity = min(unique_types / MAX_QUESTION_TYPES, 1.0) * 20
        score += diversity
        
        # Actionability (25 points)
        action_words = ['how', 'what', 'demonstrate', 'verify', 'document', 
                       'implement', 'ensure', 'process', 'mechanism']
        actionable = sum(
            1 for q in questions 
            if any(word in q.question.lower() for word in action_words)
        )
        actionability = (actionable / len(questions)) * 25
        score += actionability
        
        return score
    
    def _generate_recommendation(self, results: List[ModelResult]) -> str:
        """Generate recommendation based on comparison results."""
        
        valid = [r for r in results if not r.error]
        if not valid:
            return "No successful model runs. Check API key and rate limits."
        
        # Sort by quality
        by_quality = sorted(valid, key=lambda r: r.quality_score, reverse=True)
        best_quality = by_quality[0]
        
        # Sort by speed
        by_speed = sorted(valid, key=lambda r: r.generation_time)
        fastest = by_speed[0]
        
        # Find best value (quality/time ratio)
        value_scores = [(r, r.quality_score / max(r.generation_time, 0.1)) for r in valid]
        best_value = max(value_scores, key=lambda x: x[1])[0]
        
        lines = [
            f"BEST QUALITY: {best_quality.model_name} (score: {best_quality.quality_score})",
            f"FASTEST: {fastest.model_name} ({fastest.generation_time}s)",
            f"BEST VALUE: {best_value.model_name}",
            "",
            "Recommendations by use case:",
            f"  - Production/Critical: {best_quality.model_name}",
            f"  - Development/Testing: {fastest.model_name}",
            f"  - Cost-Optimized: {best_value.model_name}"
        ]
        
        return "\n".join(lines)
    
    def _print_summary(self, report: ComparisonReport):
        """Print comparison summary."""
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        # Table header
        print(f"\n{'Model':<25} {'Questions':<10} {'Quality':<10} {'Time (s)':<10}")
        print("-" * 55)
        
        for result in report.results:
            status = "✓" if not result.error else "✗"
            print(f"{status} {result.model_name:<23} {result.num_questions:<10} "
                  f"{result.quality_score:<10} {result.generation_time:<10}")
        
        print("\n" + "-"*55)
        print(f"\nBEST MODEL: {report.best_model}")
        print(f"\n{report.recommendation}")
    
    def _save_report(self, report: ComparisonReport, output_path: str):
        """Save comparison report to JSON."""
        
        # Convert dataclasses to dicts
        data = {
            'timestamp': report.timestamp,
            'document_name': report.document_name,
            'num_statements_tested': report.num_statements_tested,
            'models_tested': report.models_tested,
            'results': [asdict(r) for r in report.results],
            'best_model': report.best_model,
            'recommendation': report.recommendation
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nComparison report saved to: {output_path}")


# ============================================================
# CLI Helper
# ============================================================

def list_available_models():
    """Print available models and their characteristics."""
    
    print("\n" + "="*70)
    print("AVAILABLE MISTRAL MODELS")
    print("="*70 + "\n")
    
    for model_id, info in MISTRAL_MODELS.items():
        print(f"{info['name']} ({model_id})")
        print(f"  Speed: {info['speed']} | Quality: {info['quality']} | Cost: {info['cost']}")
        print(f"  {info['description']}")
        print()
