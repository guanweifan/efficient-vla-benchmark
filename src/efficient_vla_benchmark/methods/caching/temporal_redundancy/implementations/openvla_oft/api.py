def build(backend, config, resources):
    from ...method import TemporalVisualReuse

    vision = backend.model.vision_backbone
    if not vision.use_fused_vision_backbone or vision.get_num_images_in_input() != 2:
        raise ValueError("Expected native two-camera fused vision")
    return TemporalVisualReuse(
        backend, config, {"dino": vision.featurizer, "siglip": vision.fused_featurizer}, 2
    )
