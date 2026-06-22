"""
Single source of truth for tunable values. Every module that previously
hardcoded a default (sample size, token budget, threshold, model ID) should
import from here instead - this is what makes tuning the agent a one-file
edit instead of a grep across the codebase.
"""
from dataclasses import dataclass, field


@dataclass
class IngestionConfig:
    sample_size: int = 10000          # reservoir sample rows held in memory/parquet
    sample_dir: str = "/tmp/eda_agent_samples"


@dataclass
class DataContextConfig:
    token_budget: int = 3000          # ~chars/4 estimate; ceiling for to_prompt_context()


@dataclass
class TargetAnalysisConfig:
    sparse_threshold: float = 0.05        # minority class ratio below this -> sparse_target flag
    extreme_sparse_threshold: float = 0.01  # below this -> anomaly-detection framing suggested
    null_warning_threshold: float = 0.05  # target null % above this -> target_nulls flag


@dataclass
class MemoryConfig:
    keep_recent: int = 12             # chat turns kept verbatim before folding into summary
    checkpoint_db_path: str = "/tmp/eda_agent_memory.sqlite"


@dataclass
class SandboxConfig:
    # NOT YET WIRED IN (sandbox.py not built) - placeholders for when it is.
    max_cpu_seconds: int = 10
    max_memory_mb: int = 512
    timeout_seconds: int = 15
    max_self_correct_attempts: int = 2


@dataclass
class LLMConfig:
    # [Unverified] model IDs - check console.groq.com/docs/models before relying on these.
    models: dict = field(default_factory=lambda: {
        "fast": "llama-3.1-8b-instant",
        "reasoning": "llama-3.3-70b-versatile",
        "vision": "llama-3.2-90b-vision-preview",
    })


@dataclass
class FeatureModelConfig:
    skew_threshold: float = 1.0           # |skew| above this -> transform suggestion
    high_cardinality_threshold: int = 50  # distinct values above this -> target/frequency encoding, not one-hot
    high_null_threshold: float = 0.3      # null % above this -> imputation/drop suggestion
    small_dataset_rows: int = 1000        # below this -> prefer simpler/regularized models
    extreme_sparse_threshold_for_anomaly: float = 0.01  # mirrors TargetAnalysisConfig; below this, lead with anomaly detection


@dataclass
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    data_context: DataContextConfig = field(default_factory=DataContextConfig)
    target_analysis: TargetAnalysisConfig = field(default_factory=TargetAnalysisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    feature_model: FeatureModelConfig = field(default_factory=FeatureModelConfig)


# Module-level singleton - import `CONFIG` directly rather than instantiating Config()
# yourself, so every module shares the same (overridable) settings.
CONFIG = Config()
