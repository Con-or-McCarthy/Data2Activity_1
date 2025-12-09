#!/bin/bash

ORIGINAL_DATA_DIR="./data/NFI_FARED/original"  

echo "-------------------------------------------------------------------------"
echo "This script will download and extract the original NFI_FARED dataset."
echo "The data will be saved in the directory: $ORIGINAL_DATA_DIR"
echo "-------------------------------------------------------------------------"
echo ""

die () {
    echo >&2 "$@"
    exit 1
}

download_and_extract_dataset () {
    # $1: HuggingFace repo id (e.g. NetherlandsForensicInstitute/NFI_FARED_Digital_Traces)
    # $2: dataset folder name to create under $ORIGINAL_DATA_DIR
    repo_id="$1"
    dataset_name="$2"
    local_dir="$ORIGINAL_DATA_DIR"

    mkdir -p "$local_dir"
    echo "Downloading .pkl files from HuggingFace repo '$repo_id' into $local_dir ..."

    python - <<PY
import os
from huggingface_hub import list_repo_files, hf_hub_download
repo_id = "${repo_id}"
repo_type = "dataset"
local_dir = "${local_dir}"
files_on_hf = [f for f in list_repo_files(repo_id, repo_type=repo_type) if f.endswith(".pkl")]
files_in_local = os.listdir(local_dir)
print("Found files on HuggingFace:", files_on_hf)
for f in files_on_hf:
    if f in files_in_local:
        print(f"{f} already exists in {local_dir}, skipping...")
        continue
    p = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=f, local_dir=local_dir)
    print("Saved:", p)
print(f"\nDownloaded NFI_FARED to {local_dir}.")
PY
    return 0
}

echo "Downloading NFI_FARED via huggingface_hub..."
download_and_extract_dataset "NetherlandsForensicInstitute/NFI_FARED_Digital_Traces" "NFI_FARED"
echo "-------------------------------------------------------------------------"