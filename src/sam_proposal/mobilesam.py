from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

import numpy as np
import torch
import torchvision

class MobileSAM:
    """
    MobileSAM Wrapper

    This module defines a class to work with the MobileSAM (Segment Anything Model)
    architecture for efficient image segmentation and mask generation, particularly
    optimized for mobile and lightweight applications.

    References:
    - https://github.com/ChaoningZhang/MobileSAM
    - https://arxiv.org/pdf/2306.14289
    """

    def __init__(self, args) -> None:
        """
        Initializes MobileSAM with a lightweight SAM model for object proposals and 
        optionally loads a heavier SAM model for embedding extraction.

        Parameters:
        - args: Argument namespace containing the following attributes:
            - device: device string, e.g. 'cuda' or 'cpu'
            - use_sam_embeddings (int): 1 to use additional SAM model for embeddings
            - sam_model (str): 'b' for vit_b or 'h' for vit_h SAM variant
        """
        # MobileSAM lightweight configuration
        self.checkpoint = "weights/mobile_sam.pt"
        self.model_type = "vit_t"
        self.use_sam_embeddings = False
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(args.device)
        self.mask_generator = None
        self.model.eval()

        # Determine embedding feature size
        dummy_image = Image.new(mode="RGB", size=(200, 200))
        predictor = SamPredictor(self.model)
        predictor.set_image(np.array(dummy_image))
        self.features_size = predictor.features.shape[1]
        predictor.reset_image()

        # Load additional full SAM model for embeddings if needed
        if args.use_sam_embeddings == 1:
            raise RuntimeError("No valid SAM config found for embeddings.")

    def load_simple_mask(self):
        """
        Initializes the automatic mask generator for the lightweight model and optionally 
        the embedding model if enabled. Uses default configuration with customizable 
        parameters for dense sampling and post-processing.
        """
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            min_mask_region_area=100,
            output_mode="coco_rle",
        )

    def get_unlabeled_samples(self, batch, idx, transform, use_sam_embeddings):
        """
        Extracts cropped region proposals and their embeddings from a batch image.

        Parameters:
        - batch: tuple or list of (images, labels, etc.)
        - idx: index of the image in the batch to process
        - transform: object with transform.preprocess_sam_embed() or preprocess_timm_embed()
        - use_sam_embeddings: whether to use SAM-based embedding preprocessing

        Returns:
        - imgs: list of processed image tensors
        - box_coords: list of bounding boxes in xywh format
        - scores: list of predicted IOU scores for each mask
        """
        imgs = []
        box_coords = []
        scores = []

        img = batch[0][idx].cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray(img)

        masks = self.mask_generator.generate(img)

        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), in_fmt='xywh', out_fmt='xyxy'
            )
            crop = img_pil.crop(np.array(xyxy))

            sample = (
                transform.preprocess_sam_embed(crop)
                if use_sam_embeddings else
                transform.preprocess_timm_embed(crop)
            )

            imgs.append(sample)
            box_coords.append(xywh)
            scores.append(float(ann['predicted_iou']))

        return imgs, box_coords, scores

    def get_embeddings(self, img):
        pass

    def get_features(self, img):
        pass