import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, \
    CLIPImageProcessor, SiglipImageProcessor, BitImageProcessor
import torch
from mipha.model import *
from mipha.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda",
                          device="cuda"):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    print("load model from model_path: ", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if "phi3" in model_name.lower() or "phi-3" in model_name.lower():
        config = MiphaPhi3Config.from_pretrained(model_path, trust_remote_code=True)
        model = MiphaPhi3ForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_safetensors=True,
            **kwargs).to("cuda")
    elif "phi2" or "phi-2" or "olympus" or "mipha" in model_name.lower():
        config = MiphaPhiConfig.from_pretrained(model_path, trust_remote_code=True)
        model = MiphaPhiForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_safetensors=True,
            **kwargs).to("cuda")
    elif "phi1_5" in model_name.lower():
        config = MiphaPhi15Config.from_pretrained(model_path, trust_remote_code=True)
        model = MiphaPhi15ForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_safetensors=True,
            **kwargs).to("cuda")
    elif "gemma" in model_name.lower():
        config = MiphaGemmaConfig.from_pretrained(model_path, trust_remote_code=True)
        model = MiphaGemmaForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_safetensors=True,
            **kwargs).to("cuda")
    else:
        raise ValueError(f"Unknown model name: {model_name}. Your model name should containe one of ['phi-2', 'phi2',"
                         f"'phi3', 'phi-3', 'gemma', phi1_5']")

    if "clip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
    elif "siglip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = SiglipImageProcessor.from_pretrained(model_path)
    elif "dinov2" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        image_processor = BitImageProcessor.from_pretrained(model_path)
    else:
        return NotImplementedError

    if 'phi' or "gemma" or "Olympus" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

        # TODO: the tokenizer length of phi-2 is 50295, but the output class of lm_head is 51200
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    model.to(device="cuda")
    print(kwargs)
    # print(model)
    return tokenizer, model, image_processor, context_len
