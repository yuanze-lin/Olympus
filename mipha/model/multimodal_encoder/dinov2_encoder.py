from abc import ABC

import torch
import torch.nn as nn

from transformers import Dinov2Model, Dinov2Config, Dinov2PreTrainedModel
from mipha.model.language_model.configuration_mipha import MiphaVisionConfig


class Dinov2VisionTower(Dinov2PreTrainedModel):
    config_class = MiphaVisionConfig

    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.dinov2.embeddings.patch_embedding

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.config.mm_vision_select_layer]
        if self.config.mm_vision_select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.config.mm_vision_select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.config.mm_vision_select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.dinov2(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.dinov2(images.to(device=self.device, dtype=self.dtype),
                                             output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return list(self.dinov2.parameters())[0].dtype

    @property
    def device(self):
        return list(self.dinov2.parameters())[0].device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
