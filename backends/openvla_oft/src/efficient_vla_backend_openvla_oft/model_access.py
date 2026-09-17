"""OFT checkpoint identity and model geometry, never paper-specific decisions."""

from efficient_vla_benchmark.core.provenance import file_manifest


def model_manifest(root):
    names = {
        "config.json",
        "dataset_statistics.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
    }
    names.update(path.name for path in root.glob("model-*.safetensors"))
    names.update(path.name for path in root.glob("action_head--*_checkpoint.pt"))
    names.update(path.name for path in root.glob("proprio_projector--*_checkpoint.pt"))
    return file_manifest(root, names)


def decoder_layers(backend):
    return backend.model.language_model.model.layers
