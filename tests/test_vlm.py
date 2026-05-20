import json
from pathlib import Path
from unittest.mock import patch

import pytest

from openproctor.vlm.ollama_client import (
    AggregationStrategy,
    VLMConfig,
    OllamaVLM,
    DEFAULT_MODELS,
    DEFAULT_PROMPT,
)


class TestVLMConfig:
    def test_default_creation(self):
        cfg = VLMConfig(name="test", model="test-model")
        assert cfg.name == "test"
        assert cfg.model == "test-model"
        assert cfg.prompt == DEFAULT_PROMPT

    def test_custom_prompt(self):
        cfg = VLMConfig(name="test", model="test-model", prompt="custom prompt")
        assert cfg.prompt == "custom prompt"


class TestAggregationStrategy:
    def test_enum_values(self):
        assert AggregationStrategy.SINGLE.value == "single"
        assert AggregationStrategy.CONSENSUS.value == "consensus"
        assert AggregationStrategy.MAJORITY.value == "majority"
        assert AggregationStrategy.ANY.value == "any"


class TestResponseParsing:
    def test_parse_valid_json(self):
        response = '{"infraction": true, "reason": "user was on discord"}'
        result = OllamaVLM._parse(response)
        assert result["infraction"] is True
        assert "discord" in result["reason"]

    def test_parse_false_infraction(self):
        response = '{"infraction": false, "reason": "no cheating detected"}'
        result = OllamaVLM._parse(response)
        assert result["infraction"] is False

    def test_parse_with_markdown_fence(self):
        response = '```json\n{"infraction": true, "reason": "test"}\n```'
        result = OllamaVLM._parse(response)
        assert result["infraction"] is True

    def test_parse_invalid_json_returns_false(self):
        result = OllamaVLM._parse("this is not json at all")
        assert result["infraction"] is False
        assert "Could not parse" in result["reason"]


class TestAggregate:
    def test_single_passthrough(self):
        vlm = OllamaVLM(strategy="single")
        results = [{"infraction": True, "reason": "cheating", "model": "moondream"}]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is True
        assert agg["reason"] == "cheating"
        assert len(agg["per_model"]) == 1

    def test_majority_true(self):
        vlm = OllamaVLM(strategy="majority")
        results = [
            {"infraction": True, "reason": "a", "model": "m1"},
            {"infraction": True, "reason": "b", "model": "m2"},
            {"infraction": False, "reason": "c", "model": "m3"},
        ]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is True
        assert agg["infraction_count"] == 2
        assert agg["total_models"] == 3

    def test_majority_false(self):
        vlm = OllamaVLM(strategy="majority")
        results = [
            {"infraction": True, "reason": "a", "model": "m1"},
            {"infraction": False, "reason": "b", "model": "m2"},
            {"infraction": False, "reason": "c", "model": "m3"},
        ]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is False

    def test_consensus_true(self):
        vlm = OllamaVLM(strategy="consensus")
        results = [
            {"infraction": True, "reason": "a", "model": "m1"},
            {"infraction": True, "reason": "b", "model": "m2"},
        ]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is True

    def test_consensus_false(self):
        vlm = OllamaVLM(strategy="consensus")
        results = [
            {"infraction": True, "reason": "a", "model": "m1"},
            {"infraction": False, "reason": "b", "model": "m2"},
        ]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is False

    def test_any_true(self):
        vlm = OllamaVLM(strategy="any")
        results = [
            {"infraction": True, "reason": "a", "model": "m1"},
            {"infraction": False, "reason": "b", "model": "m2"},
        ]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is True

    def test_any_false(self):
        vlm = OllamaVLM(strategy="any")
        results = [
            {"infraction": False, "reason": "a", "model": "m1"},
            {"infraction": False, "reason": "b", "model": "m2"},
        ]
        agg = vlm._aggregate(results)
        assert agg["infraction"] is False


class TestInit:
    def test_string_model_single(self):
        vlm = OllamaVLM(model="moondream")
        assert len(vlm.models) == 1
        assert vlm.models[0].name == "moondream"
        assert vlm.strategy == AggregationStrategy.SINGLE

    def test_list_models(self):
        models = [VLMConfig(name="A", model="a"), VLMConfig(name="B", model="b")]
        vlm = OllamaVLM(model=models, strategy="majority")
        assert len(vlm.models) == 2
        assert vlm.strategy == AggregationStrategy.MAJORITY

    def test_default_models(self):
        vlm = OllamaVLM()
        assert len(vlm.models) == 1
        assert vlm.models[0].model == "moondream"


class TestAnalyseSingle:
    def test_returns_parsed_result(self, tmp_path):
        vlm = OllamaVLM(model="moondream")
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake_image_bytes")

        with patch("httpx.post") as mock_post:
            mock_response = mock_post.return_value
            mock_response.json.return_value = {"response": '{"infraction": true, "reason": "keyword detected"}'}
            mock_response.raise_for_status.return_value = None

            result = vlm.analyse(img_path)
            assert result["infraction"] is True
            assert result["reason"] == "keyword detected"
            assert result["file"] == str(img_path)

    def test_retry_on_failure(self, tmp_path):
        vlm = OllamaVLM(model="moondream", max_retries=2)
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake_image_bytes")

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("timeout")
            with pytest.raises(RuntimeError):
                vlm.analyse(img_path)
            assert mock_post.call_count == 3

    def test_file_not_found(self):
        vlm = OllamaVLM(model="moondream")
        with pytest.raises(FileNotFoundError):
            vlm.analyse(Path("/nonexistent/image.jpg"))


class TestAnalyseMultiModel:
    def test_runs_all_models(self, tmp_path):
        models = [VLMConfig(name="A", model="a"), VLMConfig(name="B", model="b")]
        vlm = OllamaVLM(model=models, strategy="consensus")
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        with patch("httpx.post") as mock_post:
            mock_response = mock_post.return_value
            mock_response.json.return_value = {"response": '{"infraction": true, "reason": "seen"}'}
            mock_response.raise_for_status.return_value = None

            result = vlm.analyse(img_path)
            assert result["infraction"] is True
            assert "per_model" in result
            assert len(result["per_model"]) == 2
            assert result["strategy"] == "consensus"
            assert result["total_models"] == 2
            assert result["infraction_count"] == 2

    def test_mixed_verdicts_majority(self, tmp_path):
        models = [VLMConfig(name="A", model="a"), VLMConfig(name="B", model="b")]
        vlm = OllamaVLM(model=models, strategy="majority")
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        responses = [
            {"response": '{"infraction": true, "reason": "yes"}'},
            {"response": '{"infraction": false, "reason": "no"}'},
        ]

        with patch("httpx.post") as mock_post:
            mock_response = mock_post.return_value
            mock_response.json.side_effect = responses
            mock_response.raise_for_status.return_value = None

            result = vlm.analyse(img_path)
            assert result["infraction"] is False
            assert result["infraction_count"] == 1

    def test_different_prompts_per_model(self, tmp_path):
        custom_prompt = "Custom prompt for model B"
        models = [
            VLMConfig(name="A", model="a"),
            VLMConfig(name="B", model="b", prompt=custom_prompt),
        ]
        vlm = OllamaVLM(model=models, strategy="any")
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        seen_prompts = []
        seen_calls = []

        def mock_post(url, json, timeout):
            seen_prompts.append(json["prompt"])
            seen_calls.append(json)
            mock_resp = type("MockResp", (), {})()
            mock_resp.json = lambda: {"response": '{"infraction": false, "reason": "ok"}'}
            mock_resp.raise_for_status = lambda: None
            return mock_resp

        with patch("httpx.post", side_effect=mock_post):
            vlm.analyse(img_path)
            assert seen_prompts[0] == DEFAULT_PROMPT
            assert seen_prompts[1] == custom_prompt


class TestAnalyseDirectory:
    def test_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        vlm = OllamaVLM(model="moondream")
        result = vlm.analyse_directory(suspects_dir=d)
        assert result == []

    def test_no_image_files(self, tmp_path):
        d = tmp_path / "no_images"
        d.mkdir()
        (d / "notes.txt").write_text("hello")
        vlm = OllamaVLM(model="moondream")
        result = vlm.analyse_directory(suspects_dir=d)
        assert result == []


class TestDEFAULT_MODELS:
    def test_all_have_names(self):
        for c in DEFAULT_MODELS:
            assert c.name
            assert c.model
            assert c.prompt == DEFAULT_PROMPT

    def test_moondream_first(self):
        assert DEFAULT_MODELS[0].name == "Moondream"
