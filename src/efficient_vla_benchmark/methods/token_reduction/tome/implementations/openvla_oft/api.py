def build(backend, config, resources):
    from ...method import ToMeCore

    vision = backend.model.vision_backbone
    return ToMeCore(backend, config, {"dino": vision.featurizer, "siglip": vision.fused_featurizer})
