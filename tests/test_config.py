import pytest

from src.riff.config import AgentConfig, ApiConfig, MemoryConfig


class TestMemoryConfigDefaults:
    def test_defaults(self):
        c = MemoryConfig()
        assert c.max_context_tokens == 8000
        assert c.working_ratio == 0.5
        assert c.semantic_ratio == 0.3
        assert c.episodic_ratio == 0.2
        assert c.min_relevance_score == 0.4
        assert c.dedup_threshold == 0.90
        assert c.chars_per_token == 4
        assert c.moss_timeout_ms == 500
        assert c.ambient_top_k == 5
        assert c.deep_recall_top_k == 8


class TestMemoryConfigValidation:
    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            MemoryConfig(working_ratio=0.5, semantic_ratio=0.5, episodic_ratio=0.5)

    def test_ratios_slightly_off(self):
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            MemoryConfig(working_ratio=0.4, semantic_ratio=0.3, episodic_ratio=0.2)

    def test_negative_ratio(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            MemoryConfig(working_ratio=-0.1, semantic_ratio=0.8, episodic_ratio=0.3)

    def test_max_context_tokens_zero(self):
        with pytest.raises(ValueError, match="max_context_tokens must be positive"):
            MemoryConfig(max_context_tokens=0)

    def test_max_context_tokens_negative(self):
        with pytest.raises(ValueError, match="max_context_tokens must be positive"):
            MemoryConfig(max_context_tokens=-100)

    def test_min_relevance_score_below_zero(self):
        with pytest.raises(ValueError, match="min_relevance_score must be between 0 and 1"):
            MemoryConfig(min_relevance_score=-0.1)

    def test_min_relevance_score_above_one(self):
        with pytest.raises(ValueError, match="min_relevance_score must be between 0 and 1"):
            MemoryConfig(min_relevance_score=1.1)

    def test_dedup_threshold_above_one(self):
        with pytest.raises(ValueError, match="dedup_threshold must be between 0 and 1"):
            MemoryConfig(dedup_threshold=1.5)

    def test_dedup_threshold_below_zero(self):
        with pytest.raises(ValueError, match="dedup_threshold must be between 0 and 1"):
            MemoryConfig(dedup_threshold=-0.1)

    def test_chars_per_token_zero(self):
        with pytest.raises(ValueError, match="chars_per_token must be positive"):
            MemoryConfig(chars_per_token=0)

    def test_moss_timeout_ms_zero(self):
        with pytest.raises(ValueError, match="moss_timeout_ms must be positive"):
            MemoryConfig(moss_timeout_ms=0)

    def test_ambient_top_k_zero(self):
        with pytest.raises(ValueError, match="ambient_top_k must be positive"):
            MemoryConfig(ambient_top_k=0)

    def test_deep_recall_top_k_zero(self):
        with pytest.raises(ValueError, match="deep_recall_top_k must be positive"):
            MemoryConfig(deep_recall_top_k=0)

    def test_boundary_min_relevance_zero(self):
        c = MemoryConfig(min_relevance_score=0.0)
        assert c.min_relevance_score == 0.0

    def test_boundary_min_relevance_one(self):
        c = MemoryConfig(min_relevance_score=1.0)
        assert c.min_relevance_score == 1.0

    def test_large_max_context_tokens(self):
        c = MemoryConfig(max_context_tokens=1_000_000)
        assert c.max_context_tokens == 1_000_000


class TestMemoryConfigFromDict:
    def test_partial_override(self):
        c = MemoryConfig.from_dict({"max_context_tokens": 4000})
        assert c.max_context_tokens == 4000
        assert c.working_ratio == 0.5

    def test_full_override(self):
        c = MemoryConfig.from_dict({
            "max_context_tokens": 2000,
            "working_ratio": 0.6,
            "semantic_ratio": 0.2,
            "episodic_ratio": 0.2,
            "min_relevance_score": 0.5,
            "dedup_threshold": 0.85,
            "chars_per_token": 3,
            "moss_timeout_ms": 300,
            "ambient_top_k": 10,
            "deep_recall_top_k": 15,
        })
        assert c.max_context_tokens == 2000
        assert c.working_ratio == 0.6
        assert c.deep_recall_top_k == 15

    def test_unknown_keys_ignored(self):
        c = MemoryConfig.from_dict({"unknown_key": 999, "max_context_tokens": 5000})
        assert c.max_context_tokens == 5000

    def test_empty_dict_gives_defaults(self):
        c = MemoryConfig.from_dict({})
        assert c.max_context_tokens == 8000

    def test_invalid_values_still_raise(self):
        with pytest.raises(ValueError):
            MemoryConfig.from_dict({"max_context_tokens": -1})


class TestAgentConfigDefaults:
    def test_defaults(self):
        c = AgentConfig()
        assert c.cooldown_seconds == 8.0
        assert c.min_word_count == 3
        assert c.quality_threshold == 7.0
        assert c.max_tokens_reply == 50


class TestAgentConfigValidation:
    def test_negative_cooldown(self):
        with pytest.raises(ValueError, match="cooldown_seconds must be non-negative"):
            AgentConfig(cooldown_seconds=-1.0)

    def test_negative_min_word_count(self):
        with pytest.raises(ValueError, match="min_word_count must be non-negative"):
            AgentConfig(min_word_count=-1)

    def test_negative_quality_threshold(self):
        with pytest.raises(ValueError, match="quality_threshold must be non-negative"):
            AgentConfig(quality_threshold=-0.5)

    def test_zero_max_tokens_reply(self):
        with pytest.raises(ValueError, match="max_tokens_reply must be positive"):
            AgentConfig(max_tokens_reply=0)

    def test_negative_max_tokens_reply(self):
        with pytest.raises(ValueError, match="max_tokens_reply must be positive"):
            AgentConfig(max_tokens_reply=-10)

    def test_zero_cooldown_valid(self):
        c = AgentConfig(cooldown_seconds=0.0)
        assert c.cooldown_seconds == 0.0

    def test_zero_min_word_count_valid(self):
        c = AgentConfig(min_word_count=0)
        assert c.min_word_count == 0

    def test_zero_quality_threshold_valid(self):
        c = AgentConfig(quality_threshold=0.0)
        assert c.quality_threshold == 0.0


class TestAgentConfigFromDict:
    def test_partial_override(self):
        c = AgentConfig.from_dict({"cooldown_seconds": 5.0})
        assert c.cooldown_seconds == 5.0
        assert c.min_word_count == 3

    def test_unknown_keys_ignored(self):
        c = AgentConfig.from_dict({"bogus": True, "quality_threshold": 9.0})
        assert c.quality_threshold == 9.0

    def test_empty_dict_gives_defaults(self):
        c = AgentConfig.from_dict({})
        assert c.max_tokens_reply == 50


class TestApiConfigDefaults:
    def test_defaults(self):
        c = ApiConfig()
        assert c.host == "0.0.0.0"
        assert c.port == 8765


class TestApiConfigValidation:
    def test_empty_host(self):
        with pytest.raises(ValueError, match="host must not be empty"):
            ApiConfig(host="")

    def test_port_zero(self):
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ApiConfig(port=0)

    def test_port_negative(self):
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ApiConfig(port=-1)

    def test_port_too_high(self):
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ApiConfig(port=70000)

    def test_port_boundary_one(self):
        c = ApiConfig(port=1)
        assert c.port == 1

    def test_port_boundary_max(self):
        c = ApiConfig(port=65535)
        assert c.port == 65535


class TestApiConfigFromDict:
    def test_partial_override(self):
        c = ApiConfig.from_dict({"port": 9000})
        assert c.port == 9000
        assert c.host == "0.0.0.0"

    def test_unknown_keys_ignored(self):
        c = ApiConfig.from_dict({"nope": 1, "host": "localhost"})
        assert c.host == "localhost"

    def test_empty_dict_gives_defaults(self):
        c = ApiConfig.from_dict({})
        assert c.port == 8765
