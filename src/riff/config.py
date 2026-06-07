"""Configuration dataclasses for Riff AI co-host."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class MemoryConfig:
    max_context_tokens: int = 8000
    working_ratio: float = 0.5
    semantic_ratio: float = 0.3
    episodic_ratio: float = 0.2
    min_relevance_score: float = 0.4
    dedup_threshold: float = 0.90
    chars_per_token: int = 4
    moss_timeout_ms: int = 500
    ambient_top_k: int = 5
    deep_recall_top_k: int = 8

    def __post_init__(self):
        ratios = self.working_ratio + self.semantic_ratio + self.episodic_ratio
        if abs(ratios - 1.0) > 1e-9:
            raise ValueError(
                f"Ratios must sum to 1.0, got {ratios:.4f} "
                f"(working={self.working_ratio}, semantic={self.semantic_ratio}, episodic={self.episodic_ratio})"
            )
        if self.max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be positive, got {self.max_context_tokens}")
        if not (0.0 <= self.min_relevance_score <= 1.0):
            raise ValueError(f"min_relevance_score must be between 0 and 1, got {self.min_relevance_score}")
        if not (0.0 <= self.dedup_threshold <= 1.0):
            raise ValueError(f"dedup_threshold must be between 0 and 1, got {self.dedup_threshold}")
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be positive, got {self.chars_per_token}")
        if self.moss_timeout_ms <= 0:
            raise ValueError(f"moss_timeout_ms must be positive, got {self.moss_timeout_ms}")
        if self.ambient_top_k <= 0:
            raise ValueError(f"ambient_top_k must be positive, got {self.ambient_top_k}")
        if self.deep_recall_top_k <= 0:
            raise ValueError(f"deep_recall_top_k must be positive, got {self.deep_recall_top_k}")
        for name in ("working_ratio", "semantic_ratio", "episodic_ratio"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> "MemoryConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        return cls(**filtered)


@dataclass
class AgentConfig:
    cooldown_seconds: float = 8.0
    min_word_count: int = 3
    quality_threshold: float = 7.0
    max_tokens_reply: int = 50

    def __post_init__(self):
        if self.cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds must be non-negative, got {self.cooldown_seconds}")
        if self.min_word_count < 0:
            raise ValueError(f"min_word_count must be non-negative, got {self.min_word_count}")
        if self.quality_threshold < 0:
            raise ValueError(f"quality_threshold must be non-negative, got {self.quality_threshold}")
        if self.max_tokens_reply <= 0:
            raise ValueError(f"max_tokens_reply must be positive, got {self.max_tokens_reply}")

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> "AgentConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        return cls(**filtered)


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8765

    def __post_init__(self):
        if not self.host:
            raise ValueError("host must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")

    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> "ApiConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        return cls(**filtered)
