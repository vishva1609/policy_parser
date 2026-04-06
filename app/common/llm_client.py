"""
MistralConfig — Shared LLM configuration and chain factory.

This directly mirrors OpenAIConfig in common/openapi.py from archit1012/qa-bot-llm:

    class OpenAIConfig:
        def __init__(self):
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(...)

        def get_qa_chain(self, vector_db):       ← takes built db, returns chain
            return RetrievalQA.from_chain_type(...)

MistralConfig follows the exact same pattern:
    - __init__ creates self.llm once (like OpenAIConfig.__init__)
    - get_comparison_chain(embedder) takes a built EmbeddingPipeline and returns
      a ready-to-use chain (like get_qa_chain(vector_db))

KEY OOP PRINCIPLE — Dependency Injection:
    The reference repo creates OpenAIConfig ONCE in the views layer, then
    INJECTS it into the service layer. We do exactly the same:

        # In views/pipeline_runner.py:
        self.config = MistralConfig(api_key=..., model=...)   ← created ONCE
        service.process(file, embedder=self.embedder,          ← injected down
                        config=self.config)

Usage::

    config = MistralConfig(api_key="...", model="mistral-small-latest")
    embedder = EmbeddingPipeline(...)

    # Build a comparison chain (equivalent to get_qa_chain in reference)
    chain = config.get_comparison_chain(embedder, system_prompt="...")

    # Or build a segmentation chain
    chain = config.build_chain(system_prompt="...", output_parser=JsonOutputParser())
"""
import os
from typing import Optional

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class MistralConfig:
    """
    Mistral LLM configuration — shared config object.

    Mirrors OpenAIConfig from archit1012/qa-bot-llm/common/openapi.py.

    Created ONCE in the views layer and injected into the service layer,
    which passes it into models. This is dependency injection — no class
    creates its own LLM instance independently.

    Attributes:
        model_name:   The Mistral model identifier.
        api_key:      Mistral API key (from env var if not provided).
        llm:          The shared ChatMistralAI instance.
        temperature:  Sampling temperature.
        max_tokens:   Max output tokens per call.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: Optional[float] = None,
    ):
        """
        Initialise the shared Mistral LLM.

        Args:
            api_key:     Mistral API key. Falls back to MISTRAL_API_KEY env var.
            model:       Mistral model name.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens:  Max output tokens per call.
            top_p:       Nucleus sampling; omit or None to use provider default.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Install langchain-mistralai: pip install langchain-mistralai"
            )

        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY env variable not set and no api_key provided."
            )

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        llm_kwargs = dict(
            model=model,
            mistral_api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if top_p is not None:
            llm_kwargs["top_p"] = top_p

        # Build the shared LLM instance — created once, shared everywhere
        self.llm = ChatMistralAI(**llm_kwargs)

    def get_comparison_chain(self, embedder, system_prompt: str):
        """
        Build a compliance comparison chain for a given embedder/vector db.

        This directly mirrors get_qa_chain(vector_db) from OpenAIConfig:
            def get_qa_chain(self, vector_db):
                return RetrievalQA.from_chain_type(llm=self.llm, retriever=...)

        Here we wire: system_prompt + self.llm → JSON-parsing chain.
        The embedder is the "vector_db" equivalent — it provides retrieval.

        Args:
            embedder:      An EmbeddingPipeline (or BaseEmbedder) instance.
                           Acts as the retriever — equivalent to vector_db.
            system_prompt: System message content that sets the LLM's role.

        Returns:
            A runnable LangChain chain: prompt | llm | JsonOutputParser
        """
        return self.build_chain(system_prompt, output_parser=JsonOutputParser())

    def build_chain(self, system_prompt: str, output_parser=None):
        """
        Build a LangChain chain with the given system prompt and parser.

        Args:
            system_prompt: The system message content.
            output_parser: LangChain output parser. Defaults to JsonOutputParser.

        Returns:
            A runnable chain: prompt | llm | parser
        """
        parser = output_parser if output_parser is not None else JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        return prompt | self.llm | parser

    def build_str_chain(self, system_prompt: str):
        """Convenience: build a chain returning plain string output."""
        return self.build_chain(system_prompt, output_parser=StrOutputParser())

    def __repr__(self) -> str:
        return f"MistralConfig(model={self.model_name!r}, temp={self.temperature})"


# Backward-compat alias so existing code importing MistralClient still works
MistralClient = MistralConfig


def load_mistral_config_optional(api_key: Optional[str] = None) -> Optional[MistralConfig]:
    """
    Build MistralConfig from ``config.yaml`` ``llm:`` block + env.

    - Model: ``MISTRAL_MODEL`` env overrides YAML ``llm.model``.
    - Temperature: ``MISTRAL_TEMPERATURE`` env overrides YAML ``llm.temperature`` (optional).
    Returns None if langchain is missing or no API key.
    """
    if not LANGCHAIN_AVAILABLE:
        return None
    key = api_key if api_key is not None else os.getenv("MISTRAL_API_KEY")
    if not key:
        return None
    from .config_loader import load_pipeline_config

    llm = load_pipeline_config().get("llm") or {}
    model = os.getenv("MISTRAL_MODEL") or llm.get("model") or "mistral-small-latest"
    env_temp = os.getenv("MISTRAL_TEMPERATURE")
    if env_temp is not None and env_temp.strip() != "":
        temperature = float(env_temp)
    else:
        temperature = float(llm.get("temperature", 0.1))
    max_tokens = int(llm.get("max_tokens", 2048))
    raw_tp = llm.get("top_p")
    top_p = float(raw_tp) if raw_tp is not None else None
    try:
        return MistralConfig(
            api_key=key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    except ValueError:
        return None
