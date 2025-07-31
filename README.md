# Squeeze Every Bit of Insight: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields

## Abstract

 Object Detection in deep learning typically requires large manually labeled datasets and significant computational resources for model training, making it costly and resource-intensive. To address these challenges, we propose a novel framework featuring a two-stage pipeline that eliminates the need for additional training. Our framework leverages the Segment Anything Model (SAM) as an object proposal generator combined with few-shot models to construct an efficient object detector. We introduce the use of the Mahalanobis distance with support and context prototypes, which significantly improves performance compared to traditional Euclidean-based distance metrics. The proposed pipeline was validated through a custom pineapple detection application, demonstrating its effectiveness in real-world scenarios. Furthermore, we show that our approach, utilizing only a few labeled samples, can outperform state-of-the-art few-shot models without additional training. Finally, we evaluated several SAM variants for the object proposal network and found that FastSAM achieves the highest mean average precision (mAP) for drone imagery collected from pineapple fields, outperforming other Segment Anything Model variants.

Getting Started
---------------





## 🧠 Model Weights Downloader

This script `download_weights.sh` will download pretrained weights for the following models and store them in a `weights/` directory: **FastSAM**, **EdgeSAM**, **SAM (Segment Anything Model)**, **MobileSAM**, **SAM-HQ**.

### 📦 Prerequisites

Make sure you have the following tools installed:

```bash
pip install gdown
sudo apt install wget -y
```

> `gdown` is required for downloading from Google Drive.

### 🚀 Usage
1. **Make it executable**

   ```bash
   chmod +x download_weights.sh
   ```

2. **Run the script**

   ```bash
   ./download_weights.sh
   ```

### 📁 Output

The following files will be created inside the `weights/` directory:
 `fastsam.pth`, `edge_sam.pth`, `sam_vit_h_4b8939.pth`, `mobile_sam.pt`, `sam_hq.pth`

Citation
---------------
If you find **Squeeze Every Bit of Insight: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields** useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{squeeze_bit_insight,
    title={Squeeze Every Bit of Insight: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields},
    year={2025}
}  
```

Related object detection on pineapple crops on images framework work:
```
@inproceedings{simple_object_detection_framework_without_training,
  author={Xie-Li, Danny and Fallas-Moya, Fabian and Calderon-Ramirez, Saul},
  booktitle={2024 IEEE 6th International Conference on BioInspired Processing (BIP)}, 
  title={Simple Object Detection Framework without Training}, 
  year={2024},
  pages={1-6},
  doi={10.1109/BIP63158.2024.10885396}}
```

Acknowledgement
---------------
This work was supported by We thank the National Center for High Technology of Costa Rica for access to the Kabré Cluster, the University of Costa Rica (project C4612), and ITCR's Postgraduate Office for supporting this publication.