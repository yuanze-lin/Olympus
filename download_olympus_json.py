import gdown

# Google Drive shareable link
url = 'https://drive.google.com/file/d/1CMLZLa6hkVN2K1ebCcJEOaFGc2cLeLQ7/view?usp=sharing'

# Convert to direct download link
file_id = url.split("/d/")[1].split("/")[0]
direct_url = f"https://drive.google.com/uc?id={file_id}"

# Output filename (optional)
output = "train_data/Olympus.json"

# Download the file
gdown.download(direct_url, output, quiet=False)

