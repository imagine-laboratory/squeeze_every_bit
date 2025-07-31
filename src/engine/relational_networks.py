from typing import List, Optional
from torch import Tensor, nn
from torchvision import transforms

from .fewshot_model import FewShot
from .fewshot_utils import compute_prototypes
from .predesigned_modules import default_relation_module

import torch
import numpy as np

class RelationNetworks(FewShot):
    """
    Sung, Flood, Yongxin YRelationNetworksang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales.
    "Learning to compare: Relation network for few-shot learning." (2018)
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

    In the Relation Networks algorithm, we first extract feature maps for both support and query
    images. Then we compute the mean of support features for each class (called prototypes).
    To predict the label of a query image, its feature map is concatenated with each class prototype
    and fed into a relation module, i.e. a CNN that outputs a relation score. Finally, the
    classification vector of the query is its relation score to each class prototype.

    Note that for most other few-shot algorithms we talk about feature vectors, because for each
    input image, the backbone outputs a 1-dim feature vector. Here we talk about feature maps,
    because for each input image, the backbone outputs a "feature map" of shape
    (n_channels, width, height). This raises different constraints on the architecture of the
    backbone: while other algorithms require a "flatten" operation in the backbone, here "flatten"
    operations are forbidden.

    Relation Networks use Mean Square Error. This is unusual because this is a classification
    problem. The authors justify this choice by the fact that the output of the model is a relation
    score, which makes it a regression problem. See the article for more details.
    """

    def __init__(
        self,
        *args,
        is_single_class, use_sam_embeddings,
        device: str = "cpu",
        feature_dimension: int,
        relation_module: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Build Relation Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: first dimension of the feature maps extracted by the backbone.
            relation_module: module that will take the concatenation of a query features vector
                and a prototype to output a relation score. If none is specific, we use the default
                relation module from the original paper.
        """
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension
        self.device = device
        self.mean = None
        self.std = None
        self.num_samples = None
        self.is_single_class = is_single_class
        self.use_sam_embeddings = use_sam_embeddings
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.backbone.input_size, self.backbone.input_size)),
            transforms.ToTensor()
        ])

        # Here we build the relation module that will output the relation score for each
        # (query, prototype) pair. See the function docstring for more details.
        self.relation_module = (
            relation_module
            if relation_module
            else default_relation_module(self.feature_dimension)
        )

    def get_embeddings_timm(self, img):
        """
        Returns the embeddings from the backbone which is a timm model.
        """
        with torch.no_grad():
            x =  self.backbone.forward_features(img.unsqueeze(dim=0).to(self.device))
        return x

    def get_embeddings_sam(self, img):
        """
        Returns the embeddings from the backbone which SAM.
        """
        with torch.no_grad():
            x = self.backbone.get_features(img)
        return x

    def process_support_set(
        self,
        support_images: List,
        support_labels: List = None,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature maps from the support set and store class prototypes.
        """
        support_features = []
        self.num_samples = len(support_images)

        #---------------------------------------
        # split the ids
        if support_labels is None:
            y_labels = np.zeros(len(support_images))
        else:
            y_labels = np.array(support_labels)

        #---------------------------------------
        # get feature maps from the images
        for img in support_images:
            # Resize to a normalized size to have normalized feature maps
            img = self.preprocess(img)
            if self.use_sam_embeddings:
                t_temp = self.get_embeddings_sam(img)
            else:
                t_temp = self.get_embeddings_timm(img)
            support_features.append(t_temp.squeeze().cpu())

        # get prototypes and save them into cuda memory
        support_features = torch.stack(support_features)
        if self.is_single_class:
            print("Not implemented!")
        else:
            y_labels = torch.Tensor(y_labels)
            prototypes = compute_prototypes(support_features, y_labels)
        self.prototypes = prototypes.to(self.device)

    def forward(self, query_image: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict the label of a query image by concatenating its feature map with each class
        prototype and feeding the result into a relation module, i.e. a CNN that outputs a relation
        score. Finally, the classification vector of the query is its relation score to each class
        prototype.
        """
        # Resize to a normalized size to have normalized feature maps
        query_image = self.preprocess(query_image)
        if self.use_sam_embeddings:
            z_query = self.get_embeddings_sam(query_image)
        else:
            z_query = self.get_embeddings_timm(query_image)

        # For each pair (query, prototype), we compute the concatenation of their feature maps
        # Given that query_features is of shape (n_queries, n_channels, width, height), the
        # constructed tensor is of shape (n_queries * n_prototypes, 2 * n_channels, width, height)
        # (2 * n_channels because prototypes and queries are concatenated)
        print("z_query: ", z_query.shape)
        query_prototype_feature_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(dim=0).expand(
                    z_query.shape[0], -1, -1, -1, -1
                ),
                z_query.unsqueeze(dim=1).expand(
                    -1, self.prototypes.shape[0], -1, -1, -1
                ),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *z_query.shape[2:])

        # Each pair (query, prototype) is assigned a relation scores in [0,1]. Then we reshape the
        # tensor so that relation_scores is of shape (n_queries, n_prototypes).
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(
            -1, self.prototypes.shape[0]
        )

        return self.softmax_if_specified(relation_scores)

    def _validate_features_shape(self, features):
        if len(features.shape) != 4:
            raise ValueError(
                "Illegal backbone for Relation Networks. "
                "Expected output for an image is a 3-dim  tensor of shape (n_channels, width, height)."
            )
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False