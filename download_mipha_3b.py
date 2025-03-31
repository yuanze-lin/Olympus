from huggingface_hub import snapshot_download

# Specify model repo and local directory
model_repo = "zhumj34/Mipha-3B"
local_dir = "./ckpts/Mipha-3B"

# Download model snapshot
snapshot_download(
    repo_id=model_repo,
    local_dir=local_dir,
    local_dir_use_symlinks=False  # Optional: avoid symlinks, get actual files
)

print(f"Model downloaded to: {local_dir}")

