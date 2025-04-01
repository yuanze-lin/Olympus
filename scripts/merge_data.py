import json
import os

# Load LLaVA instruction data
llava_data_path = 'jsons/llava_v1_5_mix665k.json'
with open(llava_data_path, 'r') as f:
    llava_data = json.load(f)

# Load OlympusInstruct data
olympus_instruct_path = 'jsons/OlympusInstruct.json'
with open(olympus_instruct_path, 'r') as f:
    olympus_data = json.load(f)

# Merge both datasets
merged_data = llava_data + olympus_data

# Save the final merged training data
output_path = 'jsons/Olympus.json'
with open(output_path, 'w') as f:
    json.dump(merged_data, f, indent=2)
f.close()

print(f"Merged dataset saved to {output_path} with {len(merged_data)} total entries.")
