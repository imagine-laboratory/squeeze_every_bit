import numpy as np
import torch
import torchvision

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2:
    def __init__(self, args) -> None:
        if args.sam_model == 'b':
            self.checkpoint = "/home/danny.xie/data/dxie/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif args.sam_model == 'l':
            self.checkpoint = "/home/danny.xie/data/dxie/sam2/checkpoints/sam2.1_hiera_large.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif args.sam_model == 's':
            self.checkpoint = "/home/danny.xie/data/dxie/sam2/checkpoints/sam2.1_hiera_small.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif args.sam_model == 't':
            self.checkpoint = "/home/danny.xie/data/dxie/sam2/checkpoints/sam2.1_hiera_tiny.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        
        self.model = build_sam2(self.model_cfg, 
                          self.checkpoint, 
                          device="cuda", 
                          apply_postprocessing=False)
    

    def load_simple_mask(self):
        """
        Configure and initialize the automatic mask generator.

        This uses default and custom parameters for improved segmentation of
        small objects, and filters out small mask regions.
        """

        self.mask_generator = SAM2AutomaticMaskGenerator(
                                model=self.model,
                                points_per_side=32,
                                min_mask_region_area=100,
                                use_m2m=True,
                                output_mode='coco_rle')
    

    def get_unlabeled_samples(self, batch, idx, transform, use_sam_embeddings):
        """
        Generate region proposals (unlabeled samples) from a batch using SAM.

        Args:
            batch (tuple): Tuple of batched tensors (images, labels, etc.).
            idx (int): Index of the image to process from the batch.
            transform: Transformation utility with preprocess functions.
            use_sam_embeddings (bool): Whether to use SAM-specific preprocessing.

        Returns:
            imgs (List[Tensor]): Preprocessed image crops.
            box_coords (List[List[float]]): Corresponding bounding boxes in xywh.
            scores (List[float]): IoU prediction confidence scores.
        """
        imgs, box_coords, scores = [], [], []

        # Extract and convert image
        img = batch[0][idx].cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray(img)

        # Generate masks from image
        masks = self.mask_generator.generate(img)

        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), in_fmt='xywh', out_fmt='xyxy'
            )
            crop = img_pil.crop(np.array(xyxy))

            # Apply appropriate preprocessing
            if use_sam_embeddings:
                sample = transform.preprocess_sam_embed(crop)
            else:
                sample = transform.preprocess_timm_embed(crop)

            imgs.append(sample)
            box_coords.append(xywh)
            scores.append(float(ann['predicted_iou']))

        return imgs, box_coords, scores
