from segment_anything_hq import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

import numpy as np
import torch
import torchvision

class SAMHq:
    """
    High-quality implementation of the Segment Anything Model (SAM-HQ).
    
    Based on:
    - GitHub: https://github.com/SysCV/sam-hq
    - Paper: https://arxiv.org/abs/2306.01567
    """

    def __init__(self, args) -> None:
        """
        Initialize the SAM-HQ model based on the model type specified in args.

        Args:
            args: An object with attributes:
                - sam_model: One of ['b', 'l', 'h', 'tiny'], determines the SAM variant.
                - device: The device to load the model on (e.g., 'cuda' or 'cpu').
        """
        # Set checkpoint path and model type
        if args.sam_model == 'b':
            self.checkpoint = "weights/sam_hq_vit_b.pth"
            self.model_type = "vit_b"
        elif args.sam_model == 'l':
            self.checkpoint = "weights/sam_hq_vit_l.pth"
            self.model_type = "vit_l"
        elif args.sam_model == 'h':
            self.checkpoint = "weights/sam_hq_vit_h.pth"
            self.model_type = "vit_h"
        elif args.sam_model == 'tiny':
            self.checkpoint = "weights/sam_hq_vit_tiny.pth"
            self.model_type = "vit_tiny"

        # Load model and set to eval mode
        self.use_sam_embeddings = False
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(args.device)
        self.model.eval()
        self.mask_generator = None

        # Get feature embedding size using a dummy image
        dummy_img = Image.new(mode="RGB", size=(200, 200))
        predictor = SamPredictor(self.model)
        predictor.set_image(np.array(dummy_img))
        self.features_size = predictor.features.shape[1]  # Usually 256
        predictor.reset_image()

    def load_simple_mask(self):
        """
        Load the SAM automatic mask generator with predefined configuration.
        
        Recommended for generating segmentation masks automatically. This method configures
        the mask generator with specific parameters for good-quality proposals.
        """
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            min_mask_region_area=100,  # Filters small objects
            output_mode="coco_rle",
        )

    def get_unlabeled_samples(self, batch, idx, transform, use_sam_embeddings):
        """
        Extracts unlabeled samples (cropped object proposals) from a batch using SAM-generated masks.

        Args:
            batch (Tuple[Tensor, Any]): Batch of data; batch[0] should be the images tensor.
            idx (int): Index of the sample in the batch to process.
            transform: Transform object with preprocess methods for Timm and SAM.
            use_sam_embeddings (bool): Whether to use SAM embedding preprocessing.

        Returns:
            imgs (List[Tensor]): List of cropped and preprocessed samples.
            box_coords (List[List[float]]): List of bounding boxes [x, y, w, h].
            scores (List[float]): List of predicted IoU scores for each proposal.
        """
        imgs, box_coords, scores = [], [], []

        # Get image and convert to PIL
        img = batch[0][idx].cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray(img)

        # Generate segmentation proposals
        masks = self.mask_generator.generate(img)

        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(torch.tensor(xywh), in_fmt='xywh', out_fmt='xyxy')
            crop = img_pil.crop(np.array(xyxy))

            # Apply transform based on config
            if use_sam_embeddings:
                sample = transform.preprocess_sam_embed(crop)
            else:
                sample = transform.preprocess_timm_embed(crop)

            imgs.append(sample)
            box_coords.append(xywh)
            scores.append(float(ann['predicted_iou']))

        return imgs, box_coords, scores

    def get_embeddings(self, img):
        """
        Compute average pooled feature embeddings from a given image.

        Args:
            img (np.ndarray): Image as a NumPy array.

        Returns:
            torch.Tensor: Pooled feature embeddings of shape (1, features_dim).
        """
        self.mask_generator_embeddings.predictor.set_image(img)
        embeddings = self.mask_generator_embeddings.predictor.features

        with torch.no_grad():
            avg_pooled = torch.nn.AdaptiveAvgPool2d((1, 1))(embeddings).view(embeddings.size(0), -1)

        self.mask_generator_embeddings.predictor.reset_image()
        return avg_pooled

    def get_features(self, img):
        """
        Retrieve full feature map embeddings from a given image.

        Args:
            img (np.ndarray): Image as a NumPy array.

        Returns:
            torch.Tensor: Feature map embeddings of shape (B, C, H, W).
        """
        if self.use_sam_embeddings:
            self.mask_generator_embeddings.predictor.set_image(img)
            return self.mask_generator_embeddings.predictor.features