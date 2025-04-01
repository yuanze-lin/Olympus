import argparse
import os
import sys
import torch
import json
from tqdm import tqdm
from PIL import Image

from mipha.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from mipha.conversation import conv_templates, SeparatorStyle
from mipha.model.builder import load_pretrained_model
from mipha.utils import disable_torch_init
from mipha.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    process_images,
)

def predict(args):
    # Setup model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Prepare prompt and conversation
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], args.prompt)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()

    # Setup stopping criteria
    stop_token = conv.sep2
    stopping_criteria = [
        KeywordsStoppingCriteria([stop_token], tokenizer, input_ids)
    ]

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=None,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    # Post-process output
    input_token_len = input_ids.shape[1]
    if (input_ids != output_ids[:, :input_token_len]).sum().item() > 0:
        print("[Warning] Some output_ids differ from input_ids.")

    decoded_output = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0].strip()

    if decoded_output.endswith(stop_token):
        decoded_output = decoded_output[:-len(stop_token)].strip()

    # Display results
    print("\nInput Prompt:\n", args.prompt)
    print("\nGenerated Output:\n", decoded_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m", help="Path to the pretrained model.")
    parser.add_argument("--model-base", type=str, default=None, help="Optional model base if using delta weights.")
    parser.add_argument("--conv-mode", type=str, default="llava_v0", help="Conversation mode to use.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--prompt", type=str, default="Could you generate a photo of a cat?", help="Input prompt.")
    
    args = parser.parse_args()
    predict(args)
