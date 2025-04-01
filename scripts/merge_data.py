import json
import os

# Load LLaVA instruction data
x = json.load(open('jsons/llava_v1_5_mix665k.json'))

# Load OlympusInstruct
y = json.load(open('jsons/OlympusInstruct.json'))
x.extend(y)

# Save final training data
final_json = 'jsons/Olympus.json'
with open(final_json, 'w') as fp:
    json.dump(x, fp, indent=2)
fp.close()
