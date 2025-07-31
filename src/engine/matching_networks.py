"""
See original implementation at
https://github.com/facebookresearch/low-shot-shrink-hallucinate
"""
from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn

from .predesigned_modules import (
    default_matching_networks_query_encoder,
    default_matching_networks_support_encoder,
)
from .fewshot_model import FewShot

class MatchingNetworks(FewShot):
    """
    Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
    "Matching networks for one shot learning." (2016)
    https://arxiv.org/pdf/1606.04080.pdf

    Matching networks extract feature vectors for both support and query images. Then they refine
    these feature by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.

    Be careful: while some methods use Cross Entropy Loss for episodic training, Matching Networks
    output log-probabilities, so you'll want to use Negative Log Likelihood Loss.
    """
    def __init__(
        self,
        is_single_class,
        use_sam_embeddings,
        feature_dimension: int,
        support_encoder: Optional[nn.Module] = None,
        query_encoder: Optional[nn.Module] = None,
        device="cpu",
        *args,
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

        self.feature_dimension = feature_dimension
        self.device = device
        print(self.device)

        # These modules refine support and query feature vectors
        # using information from the whole support set
        self.support_features_encoder = (
            support_encoder
            if support_encoder
            else default_matching_networks_support_encoder(self.feature_dimension)
        ).to(self.device)

        self.query_features_encoding_cell = (
            query_encoder
            if query_encoder
            else default_matching_networks_query_encoder(self.feature_dimension)
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1).to(self.device)

        self.contextualized_support_features = torch.tensor(()).to(self.device)
        self.one_hot_support_labels = torch.tensor(()).to(self.device)
        self.is_single_class = is_single_class
        self.use_sam_embeddings = use_sam_embeddings        

    def get_embeddings_timm(self, img):
        """
        Returns the embeddings from the backbone which is a timm model.
        """
        with torch.no_grad():
            x = self.backbone.forward(img.unsqueeze(dim=0).to(self.device))
        return x.to(self.device)
    
    def get_embeddings_sam(self, img):
        """
        Returns the embeddings from the backbone which SAM.
        """
        with torch.no_grad():
            x = self.backbone.get_embeddings(img)
        return x.to(self.device)
    
    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
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

        #---------------------------------------
        # split the ids
        if support_labels is None:
            y_labels = np.zeros(len(support_images))
        else:
            y_labels = np.array(support_labels) 
        
        # get feature maps from the images
        for img in support_images:
            if self.use_sam_embeddings:
                t_temp = self.get_embeddings_sam(img)
            else:
                t_temp = self.get_embeddings_timm(img)
            support_features.append(t_temp.squeeze().cpu())

        support_features = torch.stack(support_features).to(self.device)
        if self.is_single_class:
            raise NotImplementedError("Not implemented for single class.")
        else:
            self.contextualized_support_features = self.encode_support_features(
                support_features
            ).to(self.device)
            self.one_hot_support_labels = nn.functional.one_hot(torch.tensor(y_labels, dtype=torch.long)).float().to(self.device)

    def forward(self, query_image: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict query labels based on their cosine similarity to support set features.
        Classification scores are log-probabilities.

        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """

        # Refine query features using the context of the whole support set        
        if self.use_sam_embeddings:
            z_query = self.get_embeddings_sam(query_image)
        else:
            z_query = self.get_embeddings_timm(query_image)

        contextualized_query_features = self.encode_query_features(z_query).to(self.device)

        # Compute the matrix of cosine similarities between all query images
        # and normalized support images
        # Following the original implementation, we don't normalize query features to keep
        # "sharp" vectors after softmax (if normalized, all values tend to be the same)
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                nn.functional.normalize(self.contextualized_support_features).T
            )
        )

        # Compute query log probabilities based on cosine similarity to support instances
        # and support labels
        self.one_hot_support_labels = self.one_hot_support_labels.to(self.device)  # Move the tensor to the CUDA device

        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def encode_support_features(
        self,
        support_features: Tensor,
    ) -> Tensor:
        """
        Refine support set features by putting them in the context of the whole support set,
        using a bidirectional LSTM.
        Args:
            support_features: output of the backbone of shape (n_support, feature_dimension)

        Returns:
            contextualised support features, with the same shape as input features
        """
        # Since the LSTM is bidirectional, hidden_state is of the shape
        # [number_of_support_images, 2 * feature_dimension]
        support_features = support_features.to(self.device)  # Move the tensor to the CUDA device
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[
            0
        ].squeeze(0).to(self.device)

        # Following the paper, contextualized features are computed by adding original features, and
        # hidden state of both directions of the bidirectional LSTM.
        contextualized_support_features = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        ).to(self.device)

        return contextualized_support_features

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        """
        Refine query set features by putting them in the context of the whole support set,
        using attention over support set features.
        Args:
            query_features: output of the backbone of shape (n_query, feature_dimension)

        Returns:
            contextualized query features, with the same shape as input features
        """
        hidden_state = query_features.to(self.device)
        cell_state = torch.zeros_like(query_features).to(self.device)

        # We do as many iterations through the LSTM cell as there are query instances
        # Check out the paper for more details about this!
        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            ).to(self.device)
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1).to(self.device)

            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state.to(self.device) + query_features

        return hidden_state

    def _validate_features_shape(self, features: Tensor):
        self._raise_error_if_features_are_multi_dimensional(features)
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False
