#!/bin/bash

# Create weights folder if it doesn't exist
mkdir -p weights
cd weights

echo "Starting download of model weights..."

# === FastSAM ===
echo "Downloading FastSAM checkpoint..."
gdown --id 10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV -O fastsam.pth

# === EdgeSAM ===
echo "Downloading EdgeSAM checkpoint..."
wget -O edge_sam.pth https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth

# === SAM ===
echo "Downloading Segment Anything (SAM) checkpoint..."
wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# === MobileSAM ===
echo "Downloading MobileSAM checkpoint..."
wget -O mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# === SAM-HQ ===
echo "Downloading SAM-HQ checkpoint..."
gdown --id 11yExZLOve38kRZPfRx_MRxfIAKmfMY47 -O sam_hq.pth

echo "All model weights downloaded successfully into the weights/ directory."