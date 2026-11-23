import argparse
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from checkpoint import read_checkpoint, require_training_checkpoint, save_checkpoint
from config import TRAIN_CONFIG, ModelConfig
from model import GPT
from tokenizer import BYTE_TOKENIZER_PATH, PROJECT_DIR, check_tokenizer, load_tokenizer, train_tokenizer
from train import estimate_validation_loss, learning_rate_at, run


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1337)
        self.config = ModelConfig(vocab_size=1000, block_size=8, n_layer=1, n_embd=16)
        self.ids = torch.tensor([[1, 2, 3]])

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as directory:
            directory = Path(directory)
            training = replace(TRAIN_CONFIG, max_steps=4, warmup_steps=1, eval_interval=2, batch_size=2)
            options = dict(vars(training))
            control = directory / "control"
            paused = directory / "paused"
            with patch("train.MODEL_CONFIG", self.config):
                run(argparse.Namespace(run_dir=control, resume=None, tokenizer=BYTE_TOKENIZER_PATH,
                                       stop_after=None, **options))
                run(argparse.Namespace(run_dir=paused, resume=None, tokenizer=BYTE_TOKENIZER_PATH,
                                       stop_after=3, **options))
            resume_path = paused / "checkpoint-step-000003.pt"
            state = read_checkpoint(resume_path)
            require_training_checkpoint(state)
            self.assertEqual(state["completed_step"], 3)
            self.assertTrue(state["optimizer_state"]["state"])
            self.assertIsNotNone(state["best_checkpoint"])
            self.assertTrue((paused / state["best_checkpoint"]).is_file())
            self.assertEqual(state["model_config"], vars(self.config))
            self.assertEqual(state["training_config"], options)
            with self.assertRaises(FileExistsError):
                save_checkpoint(resume_path, state)
            with self.assertRaisesRegex(ValueError, "exact resume"):
                require_training_checkpoint(state["model_state"])

            saved_config_only = dict.fromkeys(options)
            run(argparse.Namespace(run_dir=None, resume=resume_path, tokenizer=None,
                                   stop_after=None, **saved_config_only))
            expected = read_checkpoint(control / "checkpoint-final.pt")
            actual = read_checkpoint(paused / "checkpoint-final.pt")
            self.assertEqual(actual["completed_step"], 4)
            self.assertEqual(actual["validation_loss"], expected["validation_loss"])
            torch.testing.assert_close(actual["model_state"], expected["model_state"], rtol=0, atol=0)
            torch.testing.assert_close(actual["optimizer_state"], expected["optimizer_state"], rtol=0, atol=0)
            torch.testing.assert_close(actual["torch_rng_state"], expected["torch_rng_state"], rtol=0, atol=0)

    def test_tokenizer_round_trip_and_byte_coverage(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR) as directory:
            path = Path(directory) / "full-byte.json"
            train_tokenizer(path)
            result = check_tokenizer(load_tokenizer(path))
            self.assertEqual(result["vocabulary_size"], 1000)
            self.assertEqual(result["unknown_token_count"], 0)
            self.assertEqual(result["missing_byte_symbols"], 0)
            self.assertTrue(result["round_trip"])
            self.assertGreater(result["token_count"], 0)

    def test_validation_is_deterministic_and_restores_mode(self):
        model = GPT(self.config)
        data = torch.arange(100)
        rng = torch.get_rng_state()
        first = estimate_validation_loss(model, data, "cpu")
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        torch.manual_seed(987)
        model.eval()
        self.assertEqual(first, estimate_validation_loss(model, data, "cpu"))
        self.assertFalse(model.training)

    def test_top_k_validates_and_supports_greedy_sampling(self):
        model = GPT(self.config).eval()
        for top_k in (0, -1, self.config.vocab_size + 1, 2.5, True):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "top_k"):
                model.generate(self.ids, max_new_tokens=1, top_k=top_k)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(ValueError, "temperature"):
                model.generate(self.ids, max_new_tokens=1, temperature=temperature)
        prediction = model(self.ids)[0][0, -1].argmax().item()
        self.assertEqual(model.generate(self.ids, max_new_tokens=1, top_k=1)[0, -1], prediction)
        torch.manual_seed(42)
        full = model.generate(self.ids, max_new_tokens=2)
        torch.manual_seed(42)
        torch.testing.assert_close(full, model.generate(self.ids, max_new_tokens=2, top_k=1000))

    def test_learning_rate_warms_up_and_decays(self):
        config = replace(TRAIN_CONFIG, max_steps=10, warmup_steps=2)
        self.assertAlmostEqual(learning_rate_at(1, config), config.learning_rate / 2)
        self.assertAlmostEqual(learning_rate_at(2, config), config.learning_rate)
        self.assertLess(learning_rate_at(5, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate_at(10, config), config.min_learning_rate)


if __name__ == "__main__":
    unittest.main()
