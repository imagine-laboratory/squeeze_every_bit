from torch import nn 
import torch
import timm
import copy
import torch.nn as nn

class Timm_head_names:
    RESNET10 = "resnet10t.c3_in1k"
    RESNET18 = 'resnet18'
    RESNETV2_50 = 'resnetv2_50'
    EFFICIENT = "tf_efficientnet_l2.ns_jft_in1k_475"
    RESNETRS_420 = "resnetrs420"

    SWINV2_BASE_WINDOW8_256 = 'swinv2_base_window8_256.ms_in1k'
    SWIN_LARGE = "swin_large_patch4_window12_384.ms_in22k_ft_in1k"
    ViT = "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
    COATNET = "coatnet_3_rw_224.sw_in12k"
    VIT_GIGANTIC = "vit_gigantic_patch14_clip_224.laion2b"
    MAXVIT = "maxvit_xlarge_tf_512.in21k_ft_in1k"

    # Small embeddings arquitectures
    XCIT_NANO = "xcit_nano_12_p8_224.fb_dist_in1k"
    COAT = "coatnext_nano_rw_224.sw_in1k"
    OPENAI_CLIP_MODEL = "vit_base_patch16_clip_224.openai_ft_in1k"
    EVA_MODEL = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    
    # Densenet
    DENSENET = "densenet121.ra_in1k"
    WIDERESNET = "wide_resnet50_2.racm_in1k"

class Identity(nn.Module):
    """ Identity to remove one layer """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class MyFeatureExtractor(nn.Module):

    def __init__(self, model_name, pretrained, num_c, use_fc=False, freeze_all=False):
        super(MyFeatureExtractor,self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.backbone.eval()
        temp_input_size = 32
        self.is_transformer = False
        self.input_size = 0

        # different types of head
        if model_name == Timm_head_names.RESNET10 or model_name == Timm_head_names.RESNET18 or model_name == Timm_head_names.RESNETRS_420 or model_name == Timm_head_names.WIDERESNET:
            # get rid of fc
            self.backbone.fc = Identity()

        elif model_name == Timm_head_names.RESNETV2_50:
            # get rid of head.fc
            self.backbone.head.fc = Identity()
        elif model_name == Timm_head_names.EFFICIENT or model_name == Timm_head_names.DENSENET or model_name == Timm_head_names.DENSENET:
            self.backbone.classifier = Identity()

        elif model_name == Timm_head_names.SWINV2_BASE_WINDOW8_256 or \
            model_name == Timm_head_names.SWIN_LARGE or model_name == Timm_head_names.COAT:
            # get rid of head.fc
            self.backbone.head.fc = Identity()
            self.backbone.head.flatten = Identity()

            # get input size - it comes in the file name.
            model_name=model_name.split('.')[0]
            temp_input_size = int(model_name.split('_')[-1])
            self.is_transformer = True

        elif model_name == Timm_head_names.ViT or \
            model_name == Timm_head_names.VIT_GIGANTIC or model_name == Timm_head_names.OPENAI_CLIP_MODEL or model_name == Timm_head_names.EVA_MODEL:
            # get rid of head
            self.backbone.head = Identity()

            model_name=model_name.split('.')[0]
            temp_input_size = int(model_name.split('_')[-1])
            self.is_transformer = True

        elif model_name == Timm_head_names.COATNET or \
             model_name == Timm_head_names.MAXVIT:
            # get rid of head
            self.backbone.head.fc = Identity()

            model_name=model_name.split('.')[0]
            temp_input_size = int(model_name.split('_')[-1])
            self.is_transformer = True

        elif model_name == Timm_head_names.XCIT_NANO:
            self.backbone.head = Identity()

            model_name=model_name.split('.')[0]
            temp_input_size = int(model_name.split('_')[-1])
            self.is_transformer = True

        if freeze_all:
            for param in self.backbone.parameters():
                param.requires_grad=False 

        # get the proper size of feature maps
        self.input_size = temp_input_size
        features_size = self.compute_backbone_output_shape(temp_input_size)
        if len(features_size) != 1:
            raise ValueError(
                "Illegal backbone for Prototypical Networks. "
                "Expected output for an image is a 1-dim tensor."
            )

        # final fc layer
        self.fc_final = nn.Linear(in_features=features_size[0], out_features=num_c)
        self.use_fc=use_fc
        self.features_size = features_size[0]

    def compute_backbone_output_shape(self, input_size=32):
        """ Compute the dimension of the feature space defined by a feature extractor.
        Params
        :backbone (nn.Module) -> feature extractor
        Returns
        :shape (int) -> shape of the feature vector computed by the feature extractor for an instance
        """
        input_images = torch.ones((4, 3, input_size, input_size))
        # Use a copy of the backbone on CPU, to avoid device conflict
        output = copy.deepcopy(self.backbone).cpu()(input_images)

        return tuple(output.shape[1:])

    def forward(self, x):
        """ Conditional forwarding (useful when we like the final feature maps) """
        x = self.backbone(x)
        if self.use_fc:
            x = self.fc_final(x)
        return x
    
    def forward_features(self, x):
        """ Conditional forwarding (useful when we like the final feature maps) """
        x = self.backbone.forward_features(x)
        if self.use_fc:
            x = self.fc_final(x)
        return x
    #----------------------
    @property
    def use_fc(self):
        return self._use_fc 

    @use_fc.setter
    def use_fc(self,value):
        self._use_fc=value
    #----------------------