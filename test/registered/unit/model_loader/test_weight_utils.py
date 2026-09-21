"""Unit tests for srt/model_loader/weight_utils.py shard-index consistency."""

import json
import os
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.loader import _set_kda_weight_dtype
from sglang.srt.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    maybe_add_mtp_safetensors,
    probe_kda_weight_dtype,
)
from sglang.srt.utils import runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"


def _write_index(folder, weight_map):
    with open(os.path.join(folder, INDEX_NAME), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def _touch(folder, name):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


def _write_safetensors_header(folder, name, tensors):
    path = os.path.join(folder, name)
    header = json.dumps(tensors).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
    return path


class TestFilterDuplicateSafetensorsFiles(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_shard_raises(self):
        # Index lists two shards, only one on disk (interrupted download).
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        present = _touch(self.folder, "model-00001-of-00002.safetensors")

        with self.assertRaises(RuntimeError) as cm:
            filter_duplicate_safetensors_files(
                hf_weights_files=[present],
                hf_folder=self.folder,
                index_file=INDEX_NAME,
            )
        self.assertIn("model-00002-of-00002.safetensors", str(cm.exception))

    def test_complete_checkpoint_filters_non_indexed(self):
        # All indexed shards present; a non-indexed duplicate is still filtered out.
        _write_index(
            self.folder,
            {
                "w1": "model-00001-of-00002.safetensors",
                "w2": "model-00002-of-00002.safetensors",
            },
        )
        shard1 = _touch(self.folder, "model-00001-of-00002.safetensors")
        shard2 = _touch(self.folder, "model-00002-of-00002.safetensors")
        extra = _touch(self.folder, "consolidated.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[shard1, shard2, extra],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(sorted(result), sorted([shard1, shard2]))

    def test_missing_shard_outside_allow_patterns_is_ignored(self):
        # Cosmos3-style checkpoints use one root index for multiple subfolder
        # weight sources. Loading the transformer source should not require the
        # vision encoder shard to already be present; the secondary source
        # downloads and loads it separately.
        _write_index(
            self.folder,
            {
                "llm": "transformer/diffusion_pytorch_model.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        shard = "transformer/diffusion_pytorch_model.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with self.subTest(folder=folder):
                transformer = os.path.join(folder, shard)
                result = filter_duplicate_safetensors_files(
                    hf_weights_files=[transformer],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )
                self.assertEqual(result, [transformer])

    def test_missing_shard_inside_allow_patterns_raises(self):
        _write_index(
            self.folder,
            {
                "llm1": "transformer/model-00001-of-00002.safetensors",
                "llm2": "transformer/model-00002-of-00002.safetensors",
                "vit": "vision_encoder/model.safetensors",
            },
        )
        shard = "transformer/model-00001-of-00002.safetensors"
        _touch(self.folder, shard)

        for folder in (self.folder, "s3://bucket/model"):
            with (
                self.subTest(folder=folder),
                self.assertRaisesRegex(
                    RuntimeError, r"model-00002-of-00002\.safetensors"
                ),
            ):
                filter_duplicate_safetensors_files(
                    hf_weights_files=[os.path.join(folder, shard)],
                    hf_folder=folder,
                    index_file=os.path.join(self.folder, INDEX_NAME),
                    allow_patterns=["transformer/*.safetensors"],
                )

    def test_local_index_allows_subset_of_existing_shards(self):
        _write_index(
            self.folder,
            {"w1": "first.safetensors", "w2": "second.safetensors"},
        )
        first = _touch(self.folder, "first.safetensors")
        _touch(self.folder, "second.safetensors")

        result = filter_duplicate_safetensors_files([first], self.folder, INDEX_NAME)

        self.assertEqual(result, [first])

    def test_single_file_model_no_index_returns_unchanged(self):
        # No index on disk (single-file / dummy / object-storage): early return.
        single = _touch(self.folder, "model.safetensors")

        result = filter_duplicate_safetensors_files(
            hf_weights_files=[single],
            hf_folder=self.folder,
            index_file=INDEX_NAME,
        )
        self.assertEqual(result, [single])


class TestMaybeAddMtpSafetensors(CustomTestCase):
    def test_remote_mtp_guards(self):
        folder = "s3://bucket/model"
        model = f"{folder}/model.safetensors"
        mtp = f"{folder}/mtp.safetensors"
        cases = (
            ("Glm4MoeForCausalLM", 1, [model, mtp], [model, mtp], False),
            ("Glm4MoeForCausalLM", 1, [model], [model], True),
            ("Glm4MoeForCausalLM", 0, [model], [model, mtp], False),
            ("LlamaForCausalLM", 1, [model], [model, mtp], False),
        )
        for arch, nextn, selected, available, may_list in cases:
            with (
                self.subTest(arch=arch, nextn=nextn, selected=selected),
                patch.object(
                    runai_utils, "list_safetensors", return_value=available
                ) as listing,
            ):
                config = SimpleNamespace(
                    architectures=[arch], num_nextn_predict_layers=nextn
                )
                self.assertEqual(
                    maybe_add_mtp_safetensors(selected, folder, INDEX_NAME, config),
                    selected,
                )
                if not may_list:
                    listing.assert_not_called()

    def test_local_unindexed_mtp(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _touch(folder, "model.safetensors")
            mtp = _touch(folder, "mtp.safetensors")
            config = SimpleNamespace(
                architectures=["Glm4MoeLiteForCausalLMNextN"],
                num_nextn_predict_layers=1,
            )
            with patch.object(runai_utils, "list_safetensors") as listing:
                self.assertEqual(
                    maybe_add_mtp_safetensors([model], folder, INDEX_NAME, config),
                    [model, mtp],
                )
                listing.assert_not_called()


class TestProbeKdaWeightDtype(CustomTestCase):
    def test_indexed_bfloat16_conv(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_index(
                folder,
                {"model.layers.0.self_attn.q_conv1d.weight": "shard.safetensors"},
            )
            _write_safetensors_header(
                folder,
                "shard.safetensors",
                {
                    "model.layers.0.self_attn.q_conv1d.weight": {
                        "dtype": "BF16",
                        "shape": [8, 1, 4],
                        "data_offsets": [0, 0],
                    }
                },
            )

            self.assertEqual(probe_kda_weight_dtype(folder), torch.bfloat16)

    def test_unindexed_fp32_conv_preserves_kimi_k3_dtype(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_safetensors_header(
                folder,
                "first.safetensors",
                {"model.embed_tokens.weight": {"dtype": "BF16"}},
            )
            _write_safetensors_header(
                folder,
                "second.safetensors",
                {
                    "language_model.model.layers.0.self_attn.q_conv1d.weight": {
                        "dtype": "F32"
                    }
                },
            )

            self.assertEqual(probe_kda_weight_dtype(folder), torch.float32)

    def test_unsupported_conv_dtype_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_safetensors_header(
                folder,
                "model.safetensors",
                {"model.layers.0.self_attn.q_conv1d.weight": {"dtype": "F8_E4M3"}},
            )

            self.assertIsNone(probe_kda_weight_dtype(folder))

    @patch("sglang.srt.model_loader.loader.probe_kda_weight_dtype")
    def test_model_dtype_is_fallback_when_probe_unavailable(self, probe):
        probe.return_value = None
        config = SimpleNamespace(
            hf_config=SimpleNamespace(linear_attn_config={}),
            model_path="remote/model",
            dtype=torch.bfloat16,
        )

        _set_kda_weight_dtype(config)

        self.assertEqual(config.hf_config._sglang_kda_weight_dtype, torch.bfloat16)

    @patch("sglang.srt.model_loader.loader.probe_kda_weight_dtype")
    def test_kimi_k3_keeps_fp32_fallback(self, probe):
        probe.return_value = None
        config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="kimi_k3", linear_attn_config={}),
            model_path="remote/model",
            dtype=torch.bfloat16,
        )

        _set_kda_weight_dtype(config)

        self.assertEqual(config.hf_config._sglang_kda_weight_dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
