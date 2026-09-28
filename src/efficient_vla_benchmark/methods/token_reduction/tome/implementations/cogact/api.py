def build(backend, config, resources):
    from ...method import ToMeCore

    vision = backend.model_access.vision
    return ToMeCore(
        backend, config, {"dino": vision.dino_featurizer, "siglip": vision.siglip_featurizer}
    )
