from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from fastsam import FastSAM

import torch
import torchvision


class FASTSAM:
    """
    FASTSAM integrates FastSAM for efficient object proposal generation
    and optionally supports SAM (Segment Anything Model) embeddings.

    FastSAM is used to generate segmentation masks and bounding boxes,
    and optionally, SAM can extract deep embeddings from these object proposals.

    References:
    - FastSAM: https://arxiv.org/pdf/2306.12156
    - GitHub: https://github.com/CASIA-IVA-Lab/FastSAM
    """

    def __init__(self, args) -> None:
        """
        Initialize the FASTSAM pipeline.

        Parameters:
        - args: An object containing the following attributes:
            - device (str): Device for running inference (e.g., "cuda" or "cpu").
            - use_sam_embeddings (int): Flag (1 or 0) to indicate whether to use SAM embeddings.
            - sam_model (str): Type of SAM model ('b' for vit_b, 'h' for vit_h).
        """
        self.device = args.device
        self.use_sam_embeddings = False  # default

        # Load FastSAM model for fast segmentation and object proposals
        self.checkpoint = '/home/rtxmsi1/Downloads/squeeze_every_bit/src/weights/FastSAM-x.pt'
        self.model = FastSAM(self.checkpoint)

        # Load SAM model if required
        if args.use_sam_embeddings == 1:
            raise RuntimeError("SAM embedding extraction is not implemented in this class.")

    def get_unlabeled_samples(self, batch, idx, transform, use_sam_embeddings):
        """
        Extract unlabeled object proposals and corresponding image crops from a batch.

        Parameters:
        - batch (tuple): A batch containing images (batch[0]) as tensors.
        - idx (int): Index of the image in the batch to extract proposals from.
        - transform (object): Transformation class with methods:
            - `preprocess_sam_embed` for SAM embeddings
            - `preprocess_timm_embed` for standard feature extraction
        - use_sam_embeddings (bool): Whether to use SAM preprocessing.

        Returns:
        - imgs (list): List of transformed image crops.
        - box_coords (list): Corresponding bounding boxes in [x_center, y_center, width, height] format.
        - scores (list): Confidence scores of the detected regions.
        """
        imgs = []
        box_coords = []
        scores = []

        # Convert the selected image tensor to a PIL image
        img = batch[0][idx].cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray(img)

        # Run FastSAM to obtain bounding box proposals
        everything_results = self.model(
            img_pil,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.1,  # Minimum confidence for detections
            iou=0.1    # Minimum IoU for NMS
        )

        # Extract bounding boxes in the format (x1, y1, x2, y2)
        results = everything_results[0].boxes

        for bbox, score in zip(results.data, results.conf):
            # Convert bounding box to xywh format (for consistency with other models)
            xyxy = bbox.cpu().numpy()[:4]
            xywh = torchvision.ops.box_convert(
                torch.tensor(xyxy), in_fmt='xyxy', out_fmt='xywh'
            )

            # Crop the region from the original image
            crop = img_pil.crop(xyxy)

            # Preprocess using the selected embedding mode
            if use_sam_embeddings:
                sample = transform.preprocess_sam_embed(crop)
            else:
                sample = transform.preprocess_timm_embed(crop)

            imgs.append(sample)
            box_coords.append([round(v) for v in xywh.tolist()])
            scores.append(float(score.item()))

        return imgs, box_coords, scores

    def load_simple_mask(self):
        """
        Placeholder for loading a simple binary mask or mask generator.

        To be implemented in the future.
        """
        pass
