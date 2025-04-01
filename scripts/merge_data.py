import json
import os

# Remove elements from OlympusBench
def remove_ele_from_eval(name, json_list):
    new_list = []

    for i in range(len(json_list)):
        index = name.split('.json')[0]+str(i)
        if index not in eval_list:
            new_list.append(json_list[i])
            
    return new_list
    
# Load llava_data
x = json.load(open('jsons/llava_v1_5_mix665k.json'))

json_path = 'jsons'
print(len(x))
count = 0

task_count = 0

# Load OlympusBench
eval_json = json.load(open('jsons/eval.json'))
eval_list = []
for eval_item in eval_json:
    for key, value in eval_item.items():
        eval_list.append(value[2])

    
files = os.listdir(json_path)
# Merge different tasks
for json_file in files:
    if not json_file.endswith('.json') or json_file == 'eval.json':
        continue
    json_path2 = os.path.join(json_path, json_file)
    m = json.load(open(json_path2))
    new_m = remove_ele_from_eval(json_file, m)
    print('\n')
    print(json_file, len(m), len(m)- len(new_m), len(new_m))
    x.extend(new_m)
    count += len(new_m)
    task_count += 1
print(count)
print(task_count)

print(len(x))

# Save final training data
final_json = 'jsons/Olympus.json'
with open(final_json, 'w') as fp:
    json.dump(x, fp, indent=2)
fp.close()

