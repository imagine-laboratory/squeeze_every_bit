from platform import processor
import numpy as np
import torch
import torchvision
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3:

    def __init__(self, args) -> None:
        bpe_path = "/home/danny.xie/data/dxie/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path)

        self.processor = Sam3Processor(model, confidence_threshold=0.5)
        dummy_img = Image.new(mode="RGB", size=(200, 200))
        


    def get_unlabeled_samples(self, batch, idx, transform, use_sam_embeddings, concept="pineapple"):

        imgs, box_coords, scores = [], [], []

        # Extract and convert image
        img = batch[0][idx].cpu().numpy().transpose(1, 2, 0)
        img_pil = Image.fromarray(img)

        self.inference_state = self.processor.set_image(img_pil)
        self.processor.reset_all_prompts(self.inference_state)
        inference_state = self.processor.set_text_prompt(
            state=self.inference_state, prompt=concept
        )

        nb_objects = len(inference_state["scores"])
        print(f"found {nb_objects} object(s)")

        for i in range(nb_objects):
            # ---- box (tensor -> list)
            box = inference_state["boxes"][i].detach().cpu()  # xyxy
            crop = img_pil.crop(box.numpy())

            # ---- preprocessing
            if use_sam_embeddings:
                sample = transform.preprocess_sam_embed(crop)
            else:
                sample = transform.preprocess_timm_embed(crop)

            # ---- convert box format and make JSON-safe
            xywh = torchvision.ops.box_convert(
                box.unsqueeze(0), in_fmt="xyxy", out_fmt="xywh"
            ).squeeze(0)

            box_coords.append(xywh.tolist())

            # ---- per-object score (scalar)
            score = inference_state["scores"][i].item()
            scores.append(float(score))

            imgs.append(sample)

        return imgs, box_coords, scores
