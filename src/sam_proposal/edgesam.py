from edge_sam import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import torch
import torchvision

class EdgeSAM:
    """
    EdgeSAM wrapper class for handling SAM-based mask generation and feature extraction.
    
    Reference:
        - EdgeSAM: Prompt-In-the-Loop Distillation for On-Device Deployment of SAM
        - https://arxiv.org/abs/2312.06660
        - https://github.com/chongzhou96/EdgeSAM
    """

    def __init__(self, args) -> None:
        """
        Initialize the EdgeSAM object with model and configuration.

        Args:
            args: An object containing the configuration (must include `device` attribute).
        """
        self.checkpoint = "weights/edge_sam.pth"
        self.model_type = "edge_sam"
        self.use_sam_embeddings = False

        # Load model and move to device
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(args.device)
        self.model.eval()
        self.mask_generator = None

        # Initialize predictor to estimate embedding size
        dummy_image = Image.new(mode="RGB", size=(200, 200))
        predictor = SamPredictor(self.model)
        predictor.set_image(np.array(dummy_image))
        self.features_size = predictor.features.shape[1]
        predictor.reset_image()

    def load_simple_mask(self):
        """
        Initialize a basic SAM automatic mask generator with default parameters.

        You may adjust the parameters like `points_per_side`, `stability_score_thresh`, 
        and `min_mask_region_area` for finer control over mask generation.
        """
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            stability_score_thresh=0.1,
            min_mask_region_area=100,
            output_mode="coco_rle",
        )

    def get_unlabeled_samples(self, batch, idx, transform, use_sam_embeddings):
        """
        Extract and preprocess unlabeled object regions (crops) from the image using SAM masks.

        Args:
            batch (tuple): A tuple where the first element is a tensor of images (B x C x H x W).
            idx (int): Index of the image in the batch to process.
            transform (object): Object that provides `preprocess_sam_embed` and `preprocess_timm_embed`.
            use_sam_embeddings (bool): Flag indicating which preprocessing method to use.

        Returns:
            imgs (list): List of transformed image crops.
            box_coords (list): List of bounding boxes (xywh format).
            scores (list): List of predicted IOU scores for each crop.
        """
        imgs, box_coords, scores = [], [], []

        # Convert tensor to PIL image
        img = batch[0][idx].cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray(img)

        # Generate segmentation masks
        masks = self.mask_generator.generate(img)

        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), in_fmt='xywh', out_fmt='xyxy'
            )

            # Crop the image
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

    def get_embeddings(self, img):
        raise NotImplementedError("get_embeddings is not yet implemented.")

    def get_features(self, img):
        raise NotImplementedError("SAM embeddings are not enabled. Set `use_sam_embeddings = True`.")