
---

<p align="center">
  <img src="assets/simple_object_detection_framework.jpg?raw=true" width="99.1%" />
</p>

# **Squeeze Every Bit of Insight**

## *Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields*

---

## 📝 Abstract
 Object Detection in deep learning typically requires large manually labeled datasets and significant computational resources for model training, making it costly and resource-intensive. To address these challenges, we propose a novel framework featuring a two-stage pipeline that eliminates the need for additional training. Our framework leverages the **Segment Anything Model (SAM)** as an object proposal generator combined with few-shot models to construct an efficient object detector. 
 
 We introduce the use of the Mahalanobis distance with support and context prototypes, which significantly improves performance compared to traditional Euclidean-based distance metrics. The proposed pipeline was validated through a custom pineapple detection application, demonstrating its effectiveness in real-world scenarios. Furthermore, we show that our approach, utilizing only a few labeled samples, can outperform state-of-the-art few-shot models without additional training. Finally, we evaluated several SAM variants for the object proposal network and found that **FastSAM** achieves the highest mean average precision (mAP) for drone imagery collected from pineapple fields, outperforming other Segment Anything Model variants.


---

## ⚙️ Installation

### ✅ Requirements

* Python ≥ 3.8
* Virtual environment (recommended)
* Dependencies listed in `requirements.txt`

### 🔧 Setup Instructions

1. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Clone the repository**

```bash
git clone https://github.com/imagine-laboratory/squeeze_every_bit.git
cd squeeze_every_bit
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

* PyTorch with torchvision and torchaudio
* Image/video libraries: OpenCV, PyYAML, etc.
* Utilities: `timm`, `scikit-learn`, `psutil`
* Few-shot segmentation models: SAM, FastSAM, MobileSAM, EdgeSAM, etc.

---

### ⚠️ Notes

* Use Python 3.8 or higher.
* For GPU acceleration, install the appropriate [PyTorch with CUDA](https://pytorch.org/get-started/locally/).
* If you encounter issues with `git+https` dependencies, make sure Git is installed:

```bash
sudo apt install git  # For Debian/Ubuntu
```


## 🗂️ Dataset Format (COCO Style)

The dataset follows the **COCO format**, commonly used for object detection tasks.

### 📁 Structure

```
dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   ├── image1.jpg
│   └── ...
├── val/
│   ├── image101.jpg
│   └── ...
```


## 🗂️ COCO Dataset – Main Sections

1. **`images`** — Image metadata
2. **`annotations`** — Object instance
3. **`categories`** — Class definitions

### 📄 Full Minimal Example

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 10,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "pineapple",
      "supercategory": "fruit"
    }
  ]
}
```

### ✅ Notes

* Bounding boxes use `[x, y, width, height]` format.


---

## 🚀 Getting Started

### 🧠 Download Pretrained Model Weights

To get started quickly, use the provided script to download pretrained weights.

#### 📦 Requirements

```bash
pip install gdown
sudo apt install wget -y
```

> `gdown` is used for downloading files from Google Drive.

#### ▶️ Usage

1. Make the script executable:

```bash
chmod +x download_weights.sh
```

2. Run the script:

```bash
./download_weights.sh
```

#### 📁 Weights Output

The following files will be saved in the `weights/` directory:
 `fastsam.pth`, `edge_sam.pth`, `sam_vit_h_4b8939.pth`, `mobile_sam.pt`, `sam_hq.pth`

---

## 📖 Citation

If you use this work in your research, please star ⭐ the repository and cite:

```
@inproceedings{squeeze_bit_insight,
    title={Squeeze Every Bit of Insight: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields},
    year={2025}
}
```

For related work on training-free object detection for pineapple crops:

```
@inproceedings{simple_object_detection_framework_without_training,
  author={Xie-Li, Danny and Fallas-Moya, Fabian and Calderon-Ramirez, Saul},
  booktitle={2024 IEEE 6th International Conference on BioInspired Processing (BIP)}, 
  title={Simple Object Detection Framework without Training}, 
  year={2024},
  pages={1-6},
  doi={10.1109/BIP63158.2024.10885396}
}
```

---

## Acknowledgements
We thank the National Center for High Technology of Costa Rica for access to the Kabré Cluster, the University of Costa Rica (project C4612), and ITCR's Postgraduate Office for supporting this publication.
