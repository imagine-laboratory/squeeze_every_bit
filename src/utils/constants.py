class AugMethod:
    """
    Defines the available data augmentation methods used in the pipeline.
    
    Attributes:
        NO_AUGMENTATION (str): No augmentation is applied to the data.
        RAND_AUGMENT (str): Random augmentation strategy is used.
    """
    NO_AUGMENTATION = 'no_augmentation'
    RAND_AUGMENT = 'rand_augmentation'


class MainMethod:
    """
    Enumerates the main experimental strategies or learning paradigms used.

    Attributes:
        ALONE (str): Baseline method using only SAM without any additional learning.
        FEWSHOT_1_CLASS (str): Few-shot learning with one class.
        FEWSHOT_2_CLASSES (str): Few-shot learning with two classes.
        FEWSHOT_OOD (str): Few-shot learning for out-of-distribution detection.
        FEWSHOT_2_CLASSES_RELATIONAL_NETWORK (str): Few-shot learning using a relational network approach.
        FEWSHOT_2_CLASSES_MATCHING (str): Few-shot learning using a matching network approach.
        FEWSHOT_2_CLASSES_BDCSPN (str): Few-shot learning using BDCSPN method.
        FEWSHOT_MAHALANOBIS (str): Few-shot learning using Mahalanobis distance.
        SELECTIVE_SEARCH (str): Selective Search as a region proposal or filtering method.
    """
    ALONE = 'samAlone'
    FEWSHOT_1_CLASS = 'fewshot1'
    FEWSHOT_2_CLASSES = 'fewshot2'
    FEWSHOT_OOD = 'fewshotOOD'
    FEWSHOT_2_CLASSES_RELATIONAL_NETWORK = 'fewshotRelationalNetwork'
    FEWSHOT_2_CLASSES_MATCHING = 'fewshotMatching'
    FEWSHOT_2_CLASSES_BDCSPN = 'fewshotBDCSPN'
    FEWSHOT_MAHALANOBIS = 'fewshotMahalanobis'
    SELECTIVE_SEARCH = 'ss'


class SamMethod:
    """
    Specifies the variants of SAM (Segment Anything Model) used.

    Attributes:
        SAM (str): The original Segment Anything Model.
        MOBILE_SAM (str): Lightweight/mobile version of SAM.
        FAST_SAM (str): Optimized version of SAM for faster inference.
        EDGE_SAM (str): Edge-compatible version of SAM.
        SAM_HQ (str): High-quality version of SAM with refined segmentation.
        SLIM_SAM (str): Slim sam version.
    """
    SAM = 'sam'
    MOBILE_SAM = 'mobilesam'
    FAST_SAM = 'fastsam'
    EDGE_SAM = 'edgsam'
    SAM_HQ = 'samhq'
    SLIM_SAM = 'slimsam'


class DimensionalityReductionMethod:
    """
    Lists dimensionality reduction techniques.

    Attributes:
        SVD (str): Singular Value Decomposition method for reducing feature dimensions.
    """
    SVD = 'svd'
