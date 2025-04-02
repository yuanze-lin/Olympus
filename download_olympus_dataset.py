import os
from huggingface_hub import snapshot_download

# Define the repository ID and type
repo_id = "Yuanze/Olympus"
repo_type = "dataset"

# Define the target directories
train_data_dir = "train_data"
jsons_dir = "jsons"

# Ensure the target directories exist
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(jsons_dir, exist_ok=True)

# Download 'Olympus.json' directly into the 'train_data' directory
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=train_data_dir,
    allow_patterns="Olympus.json",
    local_dir_use_symlinks=False
)

# Download all other JSON files into the 'jsons' directory
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=jsons_dir,
    allow_patterns="*.json",
    ignore_patterns="Olympus.json",
    local_dir_use_symlinks=False
)