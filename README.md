<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Add this line to set text color to blue */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->

<h1 align="center">
  <span style="color: teal; font-family: Consolas;">Squeeze Every Bit of Insight</span>: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields
</h1>




<div align="center">
  <a href="https://scholar.google.com/citations?user=j02Fj8EAAAAJ&hl=en" target="_blank">Fabian&nbsp;Fallas-Moya</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=vipkAKEAAAAJ&hl=es" target="_blank">Danny&nbsp;Xie-Li</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=O3hhdFQAAAAJ&hl=en" target="_blank">Saul&nbsp;Calderón-Ramírez</a><sup>2</sup>
  <br>
  <sup>1</sup> Atlantic Campus, Universidad de Costa Rica, Cartago, Costa Rica &emsp; <br>
  <sup>2</sup> Computer Science Department, Instituto Tecnologico de Costa Rica, Cartago, Costa Rica &emsp;
</div>

</div>

---

<p align="center">
  <img src="assets/simple_object_detection_framework.jpg?raw=true" width="99.1%" />
</p>


---

## 📝 Abstract
 Object Detection in deep learning typically requires large manually labeled datasets and significant computational resources for model training, making it costly and resource-intensive. To address these challenges, we propose a novel framework featuring a two-stage pipeline that eliminates the need for additional training. Our framework leverages the **Segment Anything Model (SAM)** as an object proposal generator combined with few-shot models to construct an efficient object detector. 
 
 We introduce the use of the Mahalanobis distance with support and context prototypes, which significantly improves performance compared to traditional Euclidean-based distance metrics. The proposed pipeline was validated through a custom pineapple detection application, demonstrating its effectiveness in real-world scenarios. Furthermore, we show that our approach, utilizing only a few labeled samples, can outperform state-of-the-art few-shot models without additional training. Finally, we evaluated several SAM variants for the object proposal network and found that **FastSAM** achieves the highest mean average precision (mAP) for drone imagery collected from pineapple fields, outperforming other Segment Anything Model variants.


---

## ⚙️ Installation

### ✅ Requirements

* Python == 3.8
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
cd squeeze_every_bit/src
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

* Use Python 3.8.
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
 `FastSAM-x.pt`, `edge_sam.pth`, `sam_vit_h_4b8939.pth`, `mobile_sam.pt`, `sam_hq.pth`

---

## Running `methods.py`

`methods.py` is a Python script designed to evaluate few shot models with various configurable options including TIMM models, Segment Anything Models variants, dimensionality reduction.

### Usage

Run the script via the command line with optional arguments to customize evaluation:

```bash
python methods.py [OPTIONS]
```

### Available Command-Line Arguments

| Argument                   | Type    | Default             | Description                                                                 |
| -------------------------- | ------- | ------------------- | --------------------------------------------------------------------------- |
| `--root`                   | `str`   | `.`                 | Root directory path.                                                        |
| `--num-classes`            | `int`   | `1`                 | Number of output classes.                                                   |
| `--use-sam-embeddings`     | `int`   | `0`                 | Use SAM embeddings (0 = False, 1 = True).                                   |
| `--timm-model`             | `str`   | `""`                | Name of TIMM model architecture to use, check on [TIMM Collections](https://huggingface.co/timm/collections).|
| `--ood-labeled-samples`    | `int`   | `1`                 | Number of labeled out-of-distribution samples.                              |
| `--ood-unlabeled-samples`  | `int`   | `10`                | Number of unlabeled out-of-distribution samples.                            |
| `--ood-histogram-bins`     | `int`   | `15`                | Number of bins for OOD histogram.                                           |
| `--dataset`                | `str`   | `"coco17"`          | Dataset name to use.                                                        |
| `--batch-size`             | `int`   | `4`                 | Batch size for training.                                                    |
| `--batch-size-val`         | `int`   | `64`                | Batch size for validation.                                                  |
| `--img-resolution`         | `int`   | `512`               | Input image resolution.                                                     |
| `--new-sample-size`        | `int`   | `224`               | Size of new samples after augmentation.                                     |
| `--batch-size-labeled`     | `int`   | `1`                 | Batch size for labeled data (in semi-supervised learning).                  |
| `--batch-size-unlabeled`   | `int`   | `4`                 | Batch size for unlabeled data (in semi-supervised learning).                |
| `--method`                 | `str`   | `"None"`            | Few shot method to use: `'samAlone'`, `'fewshot1'`, `'fewshot2'`, `'fewshotOOD'`, `'fewshotRelationalNetwork'`, `'fewshotMatching'`, `'fewshotBDCSPN'`, `'fewshotMahalanobis'`, `'ss'`                                           |
| `--numa`                   | `int`   | `None`              | NUMA node to use (for CPU affinity).                                        |
| `--output-folder`          | `str`   | `None`              | Folder to save outputs or checkpoints.                                      |
| `--run-name`               | `str`   | `None`              | Name of the run/experiment.                                                 |
| `--seed`                   | `int`   | `None`              | Random seed for reproducibility.                                            |
| `--sam-model`              | `str`   | `None`              | SAM Model weights size to use, available only for `"sam"` when using `--sam-proposal` model `"b"` or `"h"`. |
| `--device`                 | `str`   | `"cuda"`            | Device to run on (`"cuda"` or `"cpu"`).                                     |
| `--sam-proposal`           | `str`   | `"sam"`             | SAM proposal type: `"sam"`, `"edgsam"`, `"mobilesam"`, `"samhq"`, or `"fastsam"`. |
| `--dim-red`                | `str`   | `"svd"`             | Dimensionality reduction method (`"svd"`).                                  |
| `--n-components`           | `int`   | `10`                | Number of components for dimensionality reduction.                          |
| `--beta`                   | `int`   | `1`                 | Beta parameter (context-dependent).                                         |
| `--mahalanobis`            | `str`   | `"normal"`          | Mahalanobis distance variant.                                               |
| `--batch-size-validation`  | `int`   | `4`                 | Batch size during validation.                                               |
| `--ood-validation-samples` | `int`   | `10`                | Number of OOD validation samples.                                           |
| `--mahalanobis-lambda`     | `float` | `-1.0`              | Lambda parameter for Mahalanobis metric.                                    |

---
### Fewshot Models Available

#### 🧠 Few-Shot Model Options

The code can be executed with one of the following few-shot model options passed as a parameter (e.g., `--method fewshot1`). Each corresponds to a different few-shot strategy:

| Method Name                | Description                                                                                                                                                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `samAlone`                 | **SAM Alone**. This method performs object segmentation using one of several Segment Anything Models (SAM-H, HQ-SAM-H, MobileSAM, EdgeSAM, SlimSAM-50, or FastSAM), treating all region proposals as independent object hypotheses without class-level adaptation. |
| `fewshot1` `fewshot2`      | **Euclidean Prototype**. Implements the [prototypical network](https://arxiv.org/abs/1703.05175) method, where class prototypes are computed as the mean of embedded support samples, and classification is done using Euclidean distance to these prototypes, for one class `fewshot1` and for two classes `fewshot2`.                    |
| `ss`                       | **Selective Search (SS)**. A classical, non-deep learning region proposal method that uses hierarchical segmentation based on pixel intensity to generate object candidates. Useful as a baseline to compare against deep learning models.                         |
| `fewshotOOD`               |  **Density Prototype**. This model extends the prototypical approach by estimating class prototypes using [density functions](https://ieeexplore.ieee.org/document/10459891), providing robustness for out-of-distribution (OOD) detection.                         |
| `fewshotBDCSPN`            | **BD-CSPN**. Based on [Liu et al.](https://arxiv.org/abs/1911.10713), this approach modifies the prototypical method by dynamically refining class centroids.                                                                                  |
| `fewshotMahalanobis`       | **Mahalanobis Distance Prototype**. This variation replaces Euclidean distance with Mahalanobis distance to account for feature covariance, enabling more adaptive decision boundaries.                                                                            |

### Example command

```bash
python methods.py \
--root ./pineapples_5m --num-classes 1  --use-sam-embeddings 0 --timm-model "resnet50" \
--dataset "coco17" --batch-size 16 --img-resolution 224 --device "cuda" \
--run-name "experiment_01" --sam-proposal "fastsam" --method "samAlone"
```
---
## 📊 Results
The table below summarizes the average mAP (↑) obtained by different few-shot learning methods combined with multiple object proposal models.  
Results are reported as **mean ± standard deviation**.

- **Bold values** indicate the best-performing configuration for each few-shot method.
- The **overall best performance** is achieved by **Mahalanobis (λ = 1.0)** with **FastSAM**.

| Few-shot method | # Labeled images | Feature extractor | Object proposal model | Avg. mAP ↑ |
|-----------------|------------------|-------------------|-----------------------|------------|
| SAM Alone | – | – | SAM-H | 0.232 ± 0.007 |
| SAM Alone | – | – | HQ-SAM-H | 0.219 ± 0.007 |
| SAM Alone | – | – | EdgeSAM | 0.077 ± 0.006 |
| SAM Alone | – | – | MobileSAM | 0.205 ± 0.007 |
| SAM Alone | – | – | SlimSAM-50 | 0.145 ± 0.007 |
| SAM Alone | – | – | FastSAM | **0.378 ± 0.006** |
| Euclidean prototype | 6 | EfficientNet-L2 | SAM-H | 0.335 ± 0.019 |
| Euclidean prototype | 1 | ViT-CLIP | HQ-SAM-H | 0.287 ± 0.019 |
| Euclidean prototype | 1 | ViT-CLIP | EdgeSAM | 0.112 ± 0.009 |
| Euclidean prototype | 5 | ViT-CLIP | MobileSAM | 0.266 ± 0.005 |
| Euclidean prototype | 6 | EfficientNet-L2 | SlimSAM-50 | 0.194 ± 0.016 |
| Euclidean prototype | 4 | Xcit Nano | FastSAM | **0.393 ± 0.002** |
| Density prototype | 9 | EfficientNet-L2 | SAM-H | 0.317 ± 0.014 |
| Density prototype | 10 | Xcit Nano | HQ-SAM-H | 0.270 ± 0.019 |
| Density prototype | 10 | Xcit Nano | EdgeSAM | 0.102 ± 0.011 |
| Density prototype | 10 | Xcit Nano | MobileSAM | 0.230 ± 0.012 |
| Density prototype | 9 | EfficientNet-L2 | SlimSAM-50 | 0.185 ± 0.013 |
| Density prototype | 8 | ViT-CLIP | FastSAM | **0.354 ± 0.049** |
| BD-CSPN | 2 | Swin V2 | SAM-H | 0.264 ± 0.014 |
| BD-CSPN | 2 | Swin V2 | HQ-SAM-H | 0.234 ± 0.010 |
| BD-CSPN | 2 | Swin V2 | EdgeSAM | 0.080 ± 0.007 |
| BD-CSPN | 2 | Swin V2 | MobileSAM | 0.222 ± 0.010 |
| BD-CSPN | 2 | Swin V2 | SlimSAM-50 | 0.163 ± 0.010 |
| BD-CSPN | 2 | Swin V2 | FastSAM | **0.379 ± 0.005** |
| Mahalanobis λ=0.5 (Ours) | 3 | ViT-CLIP | SAM-H | 0.354 ± 0.031 |
| Mahalanobis λ=0.5 (Ours) | 1 | ViT-CLIP | HQ-SAM-H | 0.305 ± 0.044 |
| Mahalanobis λ=0.5 (Ours) | 1 | ViT-CLIP | EdgeSAM | 0.109 ± 0.011 |
| Mahalanobis λ=0.5 (Ours) | 3 | ViT-CLIP | MobileSAM | 0.270 ± 0.010 |
| Mahalanobis λ=0.5 (Ours) | 1 | Swin V2 | SlimSAM-50 | 0.210 ± 0.025 |
| Mahalanobis λ=0.5 (Ours) | 4 | Xcit Nano | FastSAM | **0.402 ± 0.004** |
| **Mahalanobis λ=1.0 (Ours)** | 3 | ViT-CLIP | SAM-H | **0.389 ± 0.015** |
| **Mahalanobis λ=1.0 (Ours)** | 3 | ViT-CLIP | HQ-SAM-H | **0.346 ± 0.013** |
| **Mahalanobis λ=1.0 (Ours)** | 3 | ViT-CLIP | EdgeSAM | **0.122 ± 0.008** |
| **Mahalanobis λ=1.0 (Ours)** | 3 | ViT-CLIP | MobileSAM | **0.284 ± 0.009** |
| **Mahalanobis λ=1.0 (Ours)** | 3 | ViT-CLIP | SlimSAM-50 | **0.243 ± 0.010** |
| **Mahalanobis λ=1.0 (Ours)** | 8 | Xcit Nano | FastSAM | **0.406 ± 0.003** |

---


## 📖 Citation

If you find this repository useful, please star ⭐ the repository and cite:

<a href="10.1109/CLEI67442.2025.11420556">Squeeze Every Bit of Insight: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields</a> [<b>bib</b>]

```bibtex
@InProceedings{11420556,
  author={Fallas-Moya, Fabián and Xie-Li, Danny and Calderón-Ramírez, Saúl},
  booktitle={2025 LI Latin American Computer Conference (CLEI)}, 
  title={Squeeze Every Bit of Insight: Leveraging Few-shot Models with a Compact Support Set for Domain Transfer in Object Detection from Pineapple Fields}, 
  year={2025},
  pages={1-10},
  keywords={Training;Pipelines;Prototypes;Estimation;Crops;Object detection;Transformers;Few shot learning;Load modeling;Drones;few-shot;object detection;agriculture;mahalanobis},
  doi={10.1109/CLEI67442.2025.11420556}}
```

For related work on training-free object detection for agriculture applications:


<a href="https://ieeexplore.ieee.org/abstract/document/10885396">Simple Object Detection Framework without Training</a> [<b>bib</b>]

```bibtex
@InProceedings{simple_object_detection_framework_without_training,
  author={Xie-Li, Danny and Fallas-Moya, Fabian and Calderon-Ramirez, Saul},
  booktitle={2024 IEEE 6th International Conference on BioInspired Processing (BIP)}, 
  title={Simple Object Detection Framework without Training}, 
  year={2024},
  pages={1-6},
  doi={10.1109/BIP63158.2024.10885396}
}
```


<a href="https://ieeexplore.ieee.org/document/10459891">Object detection in pineapple fields drone imagery using few shot learning and the segment anything model</a> [<b>bib</b>]
```bibtex
@InProceedings{fallas2023object,
  title={Object detection in pineapple fields drone imagery using few shot learning and the segment anything model},
  author={Fallas-Moya, Fabian and Calderon-Ramirez, Saul and Sadovnik, Amir and Qi, Hairong},
  booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)},
  pages={1635--1642},
  year={2023},
  organization={IEEE}
}
```

---

## Acknowledgements
This research was partially supported by computational resources provided through a machine allocation on the Kabré supercomputer at the Costa Rica National High Technology Center. Additional support was received from the University of Costa Rica (project C4612) and the Postgraduate Office of the Instituto Tecnológico de Costa Rica, which facilitated this publication.
