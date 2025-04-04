import json
import os

def remove_items_from_eval(task_name, data_list, eval_keys):
    """Remove entries from data_list if their index is in eval_keys."""
    filtered_list = []
    for i in range(len(data_list)):
        index = task_name.replace('.json', '') + str(i)
        if index not in eval_keys:
            filtered_list.append(data_list[i])
    return filtered_list

# Load base dataset
base_data_path = 'jsons/llava_v1_5_mix665k.json'
base_data = json.load(open(base_data_path))
print(f"Initial base data size: {len(base_data)}")

# Load evaluation exclusion list
eval_file_path = 'jsons/eval.json'
eval_data = json.load(open(eval_file_path))
eval_keys = [item[key][2] for item in eval_data for key in item]

# Initialize counters
total_added = 0
task_count = 0

# Specify the required task JSON file(s) to be used
used_task_files = ['']  # Add the path(s) to your task JSON file(s) here

# Process and merge other JSON files
for filename in used_task_files:
    if not filename.endswith('.json') or filename == 'eval.json':
        continue

    file_path = os.path.join(json_dir, filename)
    task_data = json.load(open(file_path))
    cleaned_data = remove_items_from_eval(filename, task_data, eval_keys)

    print(f"\nProcessing {filename}:")
    print(f" - Original entries: {len(task_data)}")
    print(f" - Removed: {len(task_data) - len(cleaned_data)}")
    print(f" - Remaining: {len(cleaned_data)}")

    base_data.extend(cleaned_data)
    total_added += len(cleaned_data)
    task_count += 1

print(f"\nTotal new entries added: {total_added}")
print(f"Merged task count: {task_count}")
print(f"Final dataset size: {len(base_data)}")

# Save the merged dataset
output_path = 'jsons/Olympus_specified_tasks.json'
with open(output_path, 'w') as f:
    json.dump(base_data, f, indent=2)
f.close()
