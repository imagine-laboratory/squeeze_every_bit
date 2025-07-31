import torch
import torch.nn as nn
import numpy as np

from typing import List
from torch import Tensor
from .fewshot_model import FewShot
from .fewshot_utils import compute_prototypes

class BDCSPN(FewShot):

    """
    Jinlu Liu, Liang Song, Yongqiang Qin
    "Prototype Rectification for Few-Shot Learning" (ECCV 2020)
    https://arxiv.org/abs/1911.10713

    Rectify prototypes with label propagation and feature shifting.
    Classify queries based on their cosine distance to prototypes.
    This is a transductive method.
    """
    def __init__(
        self,
        *args,
        is_single_class, use_sam_embeddings, device="cpu",
        **kwargs,
    ):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: dimension of the feature vectors extracted by the backbone.
            support_encoder: module encoding support features. If none is specific, we use
                the default encoder from the original paper.
            query_encoder: module encoding query features. If none is specific, we use
                the default encoder from the original paper.
        """
        super().__init__(*args, **kwargs)

        self.num_samples = None
        self.is_single_class = is_single_class
        self.use_sam_embeddings = use_sam_embeddings
        self.device = device

    def get_embeddings_timm(self, img):
        """
        Returns the embeddings from the backbone which is a timm model.
        """
        with torch.no_grad():
            x = self.backbone.forward(img.unsqueeze(dim=0).to(self.device))
        return x
    
    def get_embeddings_sam(self, img):
        """
        Returns the embeddings from the backbone which SAM.
        """
        with torch.no_grad():
            x = self.backbone.get_embeddings(img)
        return x

    def process_support_set(
        self,
        support_images: List,
        support_labels: List = None,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract features from the support set with full context embedding.
        Store contextualized feature vectors, as well as support labels in the one hot format.

        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        support_features = []
        self.num_samples = len(support_images)

        #---------------------------------------
        # split the ids
        if support_labels is None:
            y_labels = np.zeros(len(support_images))
        else:
            y_labels = np.array(support_labels)
        support_labels = torch.Tensor(y_labels)

        # get feature maps from the images
        for img in support_images:
            if self.use_sam_embeddings:
                t_temp = self.get_embeddings_sam(img)
            else:
                t_temp = self.get_embeddings_timm(img)
            support_features.append(t_temp.squeeze().cpu())

        # get prototypes and save them into cuda memory
        self.support_features = torch.stack(support_features).to(self.device)
        support_labels = torch.Tensor(support_labels)
        prototypes = compute_prototypes(self.support_features, support_labels)
        self.prototypes = prototypes.to(self.device)
        self.support_labels = support_labels

    def rectify_prototypes(self, query_features: Tensor):
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels.long(), n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift

        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()

        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )

        support_logits = support_logits.to(self.device)
        one_hot_support_labels = support_logits.to(self.device)
        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [n_query, n_classes]

        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Update prototypes using query images, then classify query images based
        on their cosine distance to updated prototypes.
        """
        if self.use_sam_embeddings:
            z_query = self.get_embeddings_sam(query_images)
        else:
            z_query = self.get_embeddings_timm(query_images)
        self.rectify_prototypes(
            query_features=z_query,
        )
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(z_query)
        )

    @staticmethod
    def is_transductive() -> bool:
        return True