from torch import nn

def default_matching_networks_support_encoder(feature_dimension: int) -> nn.Module:
    return nn.LSTM(
        input_size=feature_dimension,
        hidden_size=feature_dimension,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )

def default_matching_networks_query_encoder(feature_dimension: int) -> nn.Module:
    return nn.LSTMCell(feature_dimension * 2, feature_dimension)

def default_relation_module(
    feature_dimension: int, inner_channels: int = 8
) -> nn.Module:
    """
    Build the relation module that takes as input the concatenation of two feature maps, from
    Sung et al. : "Learning to compare: Relation network for few-shot learning." (2018)
    In order to make the network robust to any change in the dimensions of the input images,
    we made some changes to the architecture defined in the original implementation
    from Sung et al.(typically the use of adaptive pooling).
    Args:
        feature_dimension: the dimension of the feature space i.e. size of a feature vector
        inner_channels: number of hidden channels between the linear layers of  the relation module
    Returns:
        the constructed relation module
    """
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(
                feature_dimension * 2,
                feature_dimension,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((5, 5)),
        ),
        nn.Sequential(
            nn.Conv2d(
                feature_dimension,
                feature_dimension,
                kernel_size=3,
                padding=0,
            ),
            nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        ),
        nn.Flatten(),
        nn.Linear(feature_dimension, inner_channels),
        nn.ReLU(),
        nn.Linear(inner_channels, 1),
        nn.Sigmoid(),
    )