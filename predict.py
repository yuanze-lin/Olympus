import argparse
import re
from collections import defaultdict
import pdb
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append(os.getcwd())
from mipha.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mipha.conversation import conv_templates, SeparatorStyle
from mipha.model.builder import load_pretrained_model
from mipha.utils import disable_torch_init
from mipha.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math

def predict(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    cur_prompt = args.prompt
    images = None

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], cur_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.eos_token_id,  # Pad token
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print('\nInput Prompt: ', args.prompt)
    print('\nOutput: ', outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--prompt", type=str, default="Could you generate a photo of a cat?")
    args = parser.parse_args()

    predict(args)
