import json

import pytest
from efficient_vla_backend_cogact.checkpoint import SUITES, checkpoint_files


@pytest.fixture
def export(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "model.pt").touch()
    (tmp_path / "tokenizer").mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / "tokenizer" / name).write_text("{}")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "action_model_type": "DiT-B",
                "action_dim": 7,
                "future_action_window_size": 15,
                "past_action_window_size": 0,
                "vla": {"base_vlm": "prism-dinosiglip-224px+7b"},
            }
        )
    )
    stats = {
        suite + "_no_noops": {
            "action": {
                "q01": [0] * 7,
                "q99": [1] * 7,
                "mask": [True] * 6 + [False],
            }
        }
        for suite in SUITES
    }
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(stats))
    return tmp_path


def test_directory_and_explicit_checkpoint_agree(export):
    first = checkpoint_files(export)
    second = checkpoint_files(export / "checkpoints" / "model.pt")
    assert first == second
    assert len(first[3]) == 4


def test_missing_component_rejected(export):
    (export / "tokenizer" / "config.json").unlink()
    with pytest.raises(ValueError, match="complete export"):
        checkpoint_files(export)


def test_ambiguous_checkpoint_rejected(export):
    (export / "checkpoints" / "second.pt").touch()
    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_files(export)
    checkpoint_files(export / "checkpoints" / "model.pt")


def test_training_paths_are_not_followed(export):
    path = export / "config.json"
    config = json.loads(path.read_text())
    config["pretrained_checkpoint"] = "nonexistent-training-resource"
    config["offline_tokenizer_dir"] = "nonexistent-tokenizer"
    path.write_text(json.dumps(config))
    checkpoint_files(export)


def test_normalized_gripper_rejected(export):
    path = export / "dataset_statistics.json"
    stats = json.loads(path.read_text())
    stats["libero_goal_no_noops"]["action"]["mask"][-1] = True
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="gripper"):
        checkpoint_files(export)
