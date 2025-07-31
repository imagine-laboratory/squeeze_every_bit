from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from fastsam import FastSAM

import torch
import torchvision

class FASTSAM:
    """
    FASTSAM integrates FastSAM for efficient mask generation and optionally SAM
    (Segment Anything Model) for extracting embeddings from object proposals.

    References:
    - FastSAM: https://arxiv.org/pdf/2306.12156
    - GitHub: https://github.com/CASIA-IVA-Lab/FastSAM
    """

    def __init__(self, args) -> None:
        """
        Initialize the FASTSAM pipeline.

        Parameters:
        - args: An object containing the following attributes:
            - device: Device for running inference (e.g., "cuda" or "cpu").
            - use_sam_embeddings: Flag (1 or 0) to indicate whether to use SAM embeddings.
            - sam_model: Type of SAM model ('b' for vit_b, 'h' for vit_h).
        """
        self.device = args.device
        self.use_sam_embeddings = False

        # Load FastSAM model for fast segmentation and object proposal
        self.checkpoint = 'weights/FastSAM.pt'
        self.model = FastSAM(self.checkpoint)

        # Load SAM model for embedding extraction if enabled
        if args.use_sam_embeddings == 1:
            raise RuntimeError("No valid SAM model config found.")

    def load_simple_mask(self):
        pass