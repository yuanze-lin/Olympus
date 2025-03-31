import os
from typing import Union
from transformers import PretrainedConfig, PhiConfig, Dinov2Config, GemmaConfig, GPTNeoXConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import get_aligned_output_features_output_indices
from .phi3.modeling_phi3 import Phi3Config
from .phi1_5.modeling_phi import PhiConfig as Phi1_5Config

logger = logging.get_logger(__name__)


class MiphaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        mm_vision_select_feature (`str`, *optional*, defaults to `"patch"`):
            The feature to select from the vision encoder output. Can be one of `"patch"` or `"cls_patch"`.
        mm_vision_select_layer (`int`, *optional*, defaults to `-2`):
            The layer to select from the vision encoder output.
        vision_model_name_or_path (`str`, *optional*, defaults to `"clip"`):
            The vision model name or path to instantiate a vision encoder from.
    ```"""

    model_type = "mipha_vision_model"

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=512,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=32,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            mm_vision_select_feature="patch",
            mm_vision_select_layer=-2,
            vision_model_name_or_path="clip",
            # mlp_ratio=4,
            # hidden_dropout_prob=0.0,
            # attention_probs_dropout_prob=0.0,
            # qkv_bias=True,
            # layerscale_value=1.0,
            # drop_path_rate=0.0,
            # use_swiglu_ffn=False,
            # out_features=None,
            # out_indices=None,
            # apply_layernorm=True,
            # reshape_hidden_states=True,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.vision_model_name_or_path = vision_model_name_or_path
        # self.mlp_ratio = mlp_ratio
        # self.hidden_dropout_prob = hidden_dropout_prob
        # self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # self.qkv_bias = qkv_bias
        # self.layerscale_value = layerscale_value
        # self.drop_path_rate = drop_path_rate
        # self.use_swiglu_ffn = use_swiglu_ffn
        # self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        # self._out_features, self._out_indices = get_aligned_output_features_output_indices(
        #     out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        # )
        # self.apply_layernorm = apply_layernorm
        # self.reshape_hidden_states = reshape_hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if (config_dict.get("model_type") == "mipha_phi" or config_dict.get("model_type") == "mipha_gemma"
                or config_dict.get("model_type") == "mipha_phi3"):
            config_dict = config_dict["vision_config"]["vision_tower"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ProjectorConfig(PretrainedConfig):
    model_type = "mipha_projector"

    def __init__(
            self,
            mm_projector_type="linear",
            mm_hidden_size=768,
            hidden_size=2560,
            **kwargs
    ):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "mipha_phi" or config_dict.get("model_type") == "mipha_gemma" or \
                config_dict.get("model_type") == "mipha_phi3":
            config_dict = config_dict["vision_config"]["mm_projector"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


DEFAULT_VISUAL_CONFIG = {
    "vision_tower": MiphaVisionConfig().to_dict(),
    "mm_projector": ProjectorConfig().to_dict()
}


class MiphaPhiConfig(PhiConfig):
    model_type = "mipha_phi"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        super().__init__(**kwargs)

class MiphaPhi15Config(Phi1_5Config):
    model_type = "mipha_phi"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        super().__init__(**kwargs)


class MiphaGemmaConfig(GemmaConfig):
    model_type = "mipha_gemma"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        super().__init__(**kwargs)


class MiphaPhi3Config(Phi3Config):
    model_type = "mipha_phi3"

    def __init__(self, vision_config=None, **kwargs):
        if vision_config is None:
            self.vision_config = DEFAULT_VISUAL_CONFIG
        else:
            self.vision_config = vision_config

        super().__init__(**kwargs)


if __name__ == "__main__":
    print(MiphaVisionConfig())
