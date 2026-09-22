"""Native OFT frame preparation and query slices for the shared SpecPrune core."""

from types import SimpleNamespace

import torch
from efficient_vla_backend_openvla_oft.decoder import DecoderAdapter, DecoderStrategy

from ...method import SpecPruneCore
from ...method import stable_patch_ids as stable_patch_ids


class SpecPrune(SpecPruneCore, DecoderStrategy):
    def __init__(self, backend, config):
        from .api import preflight

        preflight(config, None)
        decoder = backend.model.language_model.model
        if len(decoder.layers) != 32:
            raise ValueError("This SpecPrune preset requires the 32-layer OFT backbone")
        super().__init__(backend, config, views=2)
        self.adapter = DecoderAdapter(decoder, self)

    def before_query(self, prepared):
        self.prepare_frame(prepared["observation"].instruction, prepared["images"])

    def text_bounds(self, layout):
        return layout.text_start, layout.text_end

    def action_queries(self, state):
        return torch.arange(
            state.layout.action_start, state.layout.length - 1, device=state.positions.device
        )

    def frame_images(self, observation):
        return observation.image.copy(), observation.wrist_image.copy()

    def prepare_images(self, images):
        from experiments.robot.openvla_utils import prepare_images_for_vla

        return prepare_images_for_vla(list(images), SimpleNamespace(center_crop=True))
