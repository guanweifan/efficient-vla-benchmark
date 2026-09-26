def build(backend, config, resources):
    from ...method import TemporalVisualReuse

    vision = backend.model_access.vision
    return TemporalVisualReuse(
        backend, config, {"dino": vision.dino_featurizer, "siglip": vision.siglip_featurizer}, 1
    )
