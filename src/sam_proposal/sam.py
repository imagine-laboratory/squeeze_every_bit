import numpy as np
import torch
import torchvision
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.predictor import SamPredictor

class SAM:
    """
    Segment Anything Module (SAM)

    Wrapper around Meta's Segment Anything model for generating masks,
    extracting region proposals, and computing image embeddings.

    References:
    - https://ai.meta.com/research/publications/segment-anything/
    - https://github.com/facebookresearch/segment-anything
    """

    def __init__(self, args) -> None:
        """
        Initialize SAM model with the specified model type and device.

        Args:
            args: Namespace or object with the following attributes:
                - sam_model (str): 'b' for ViT-B or 'h' for ViT-H.
                - device (str): Device to load the model onto ('cpu' or 'cuda').

        Raises:
            RuntimeError: If no valid SAM model type is specified.
        """
        if args.sam_model == 'b':
            self.checkpoint = "weights/sam_vit_b_01ec64.pth"
            self.model_type = "vit_b"
        elif args.sam_model == 'h':
            self.checkpoint = "weights/sam_vit_h_4b8939.pth"
            self.model_type = "vit_h"
        else:
            raise RuntimeError("No valid SAM model config found (use 'b' or 'h')")

        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(args.device)
        self.mask_generator = None

        # Estimate feature size using a dummy image
        dummy_img = Image.new(mode="RGB", size=(200, 200))
        predictor = SamPredictor(self.model)
        predictor.set_image(np.array(dummy_img))

        # Feature map shape is [1, 256, H, W]; we keep 256 as feature size
        self.features_size = predictor.features.shape[1]
        predictor.reset_image()

    def load_simple_mask(self):
        """
        Configure and initialize the automatic mask generator.

        This uses default and custom parameters for improved segmentation of
        small objects, and filters out small mask regions.
        """
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            min_mask_region_area=100,  # Post-processing filter (requires OpenCV)
            output_mode="coco_rle",
        )

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

    def get_embeddings(self, img):
        """
        Get average pooled embeddings from the SAM model.

        Args:
            img (np.ndarray): Input image as NumPy array.

        Returns:
            torch.Tensor: Pooled embedding vector (1D).
        """
        self.mask_generator.predictor.set_image(img)
        embeddings = self.mask_generator.predictor.features

        with torch.no_grad():
            pooled = torch.nn.AdaptiveAvgPool2d((1, 1))(embeddings).view(embeddings.size(0), -1)

        self.mask_generator.predictor.reset_image()
        return pooled

    def get_features(self, img):
        """
        Get feature map embeddings from the SAM model.

        Args:
            img (np.ndarray): Input image as NumPy array.

        Returns:
            torch.Tensor: Feature map (C x H x W).
        """
        self.mask_generator.predictor.set_image(img)
        return self.mask_generator.predictor.features