import os
import json
import torch
import scipy
# from fastai.vision import *
# from fastai.callbacks import CSVLogger
from numbers import Integral
import torch
import numpy as np
from tqdm import tqdm
from data import Transform_To_Models
from metrics.iou import intersection
torch.set_printoptions(threshold=10_000)
from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


class OOD_filter_neg_likelihood:
    def __init__(self, 
        timm_model=None, 
        timm_pretrained=True, 
        num_classes=1,
        sam_model=None,
        use_sam_embeddings=False,
        device="cpu"):
        """ OOD filter constructor
        Params
        :timm_model (str) -> model name from timm library
        :timm_pretrained (bool) -> whether to load a pretrained model or not
        :num_classes (int) -> number of classes in the ground truth
        """
        self.device=device
        self.num_classes = num_classes
        if not use_sam_embeddings:
            # create a model for feature extraction
            feature_extractor = MyFeatureExtractor(
                timm_model, timm_pretrained, num_classes
            ).to(self.device)
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = sam_model
        self.sam_model = sam_model

        # create the default transformation
        if use_sam_embeddings:
            trans_norm = Transform_To_Models()
        else:
            if feature_extractor.is_transformer:
                trans_norm = Transform_To_Models(
                        size=feature_extractor.input_size, 
                        force_resize=True, keep_aspect_ratio=False
                    )
            else:
                trans_norm = Transform_To_Models(
                        size=33, force_resize=False, keep_aspect_ratio=True
                    )
        self.trans_norm = trans_norm
        self.use_sam_embeddings = use_sam_embeddings
        self.power_transf = PowerTransformer(method='yeo-johnson', standardize=True)


        # self.mean = 0
        # self.std = 0

    def run_filter(self, 
        labeled_loader, 
        unlabeled_loader,  
        dir_filtered_root = None, 
        ood_thresh = 0.0, 
        ood_hist_bins = 15, val=False):
        """
        Params
        :labeled_loader: path for the first data bunch, labeled data
        :unlabeled_loader: unlabeled data
        :dir_filtered_root: path for the filtered data to be stored
        :ood_thresh: ood threshold to apply
        :path_reports_ood: path for the ood filtering reports

        Return
        :NULL -> the output of this method is a json file save in a directory.
        """      
        # 1. Get feature maps from the labeled set
        labeled_imgs = []
        labeled_labels = []
        for batch in labeled_loader:
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from bounding boxes)
                imgs_f, labels_f = get_foreground(
                    batch, idx, self.trans_norm,
                    self.use_sam_embeddings
                )

                labeled_imgs += imgs_f
                labeled_labels += labels_f
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]

        # get all features maps using: the extractor + the imgs
        feature_maps_list = self.get_all_features(labeled_imgs)

        #----------------------------------------------------------------
        #y_dumpy = np.zeros(len(feature_maps_list))
        #imgs_1, imgs_2, _, _ = train_test_split(
        #    feature_maps_list, y_dumpy, 
        #    train_size = 0.6,
        #    shuffle=True # shuffle the data before splitting
        #)
        #----------------------------------------------------------------

        # 2. Calculating the histograms from the labeled data
        # go over each dimension and get values from the whole dataset
        #----------------------------------------------------------------
        (
            histograms_labeled,    # e.g. 512 x 15
            buckets_labeled        # e.g. 512 x 15
        ) = self.calculate_hist_dataset(feature_maps_list, ood_hist_bins) #
        #) = self.calculate_hist_dataset(imgs_1, ood_hist_bins) #


        # get likelihoods
        likelihoods = self.get_likelihoods(
            histograms_labeled,    # e.g. 512 x 15
            buckets_labeled,       # e.g. 512 x 15
            feature_maps_list
            #imgs_2
        )

        # transfor the distribution to a gaussian one
        likelihoods_data = likelihoods.reshape((len(likelihoods),1))
        _ = self.power_transf.fit_transform(likelihoods_data)
        # self.mean = scipy.mean(likelihoods.numpy())
        # self.std = scipy.std(likelihoods.numpy())
        #----------------------------------------------------------------

        # go through each batch unlabeled
        likelihoods_all = 0
        # epsilon to avoid Inf results in logarithm
        eps = 0.0000000001
        # keep track of the img id for every sample created by sam
        imgs_ids = []
        imgs_box_coords = []
        imgs_scores = []

        # 3. Get batch of unlabeled // Evaluating the likelihood of unlabeled data
        for (batch_num, batch) in tqdm(
            enumerate(unlabeled_loader), total= len(unlabeled_loader)
        ):
            unlabeled_imgs = []

            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from sam)
                imgs_s, box_coords, scores = self.sam_model.get_unlabeled_samples(
                    batch, idx, self.trans_norm, self.use_sam_embeddings
                )
                unlabeled_imgs += imgs_s

                # accumulate SAM info (inferences)
                imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
                imgs_box_coords += box_coords
                imgs_scores += scores

            # get all features maps using: the extractor + the imgs
            featuremaps_list = self.get_all_features(unlabeled_imgs)
            featuremaps = torch.stack(featuremaps_list) # e.g. [387 x 512]

            # init buffer with dims
            samples_num = featuremaps.shape[0]
            likelihoods_buffer = torch.zeros(
                samples_num, self.feature_extractor.features_size
            )

            # go  through each dimension, and calculate the likelihood for the whole unlabeled dataset
            for dimension in range(self.feature_extractor.features_size):
                # calculate the histogram for the given feature, in the labeled dataset
                hist_dim = histograms_labeled[dimension, :] #e.g. [512 x 15] -> [15]
                bucks_dim = buckets_labeled[dimension, :]   #e.g. [512 x 15] -> [15]

                # take only the values of the current feature for all the observations
                vals_dim = featuremaps[:, dimension] # e.g. [387 x 512] -> [387]

                # fetch the bucket indices for all the observations, for the current feature
                min_buckets = self.find_closest_bucket_all_obs( # e.g. [387]
                    vals_dim, bucks_dim
                )

                # evaluate likelihood for the specific
                likelihoods_ = self.get_prob_values_all_obs( # e.g. [387]
                    min_buckets, hist_dim
                )
                # squeeze to eliminate an useless dimension
                likelihoods_buffer[:, dimension] = likelihoods_.squeeze() # e.g. [387 x 512]

            # calculate the log of the sum of the likelihoods for all the dimensions, obtaining a score per observation
            #THE LOWER THE BETTER
            likelihoods_batch = -1 * torch.sum(torch.log(likelihoods_buffer + eps), 1) # e.g. [387]
            
            # accumulate
            if (batch_num == 0):
                likelihoods_all = likelihoods_batch
            else:
                likelihoods_all = torch.cat((likelihoods_all, likelihoods_batch), 0)

        # transform data 
        scores = []
        for j in range(0, likelihoods_all.shape[0]):
            scores += [likelihoods_all[j].item()]
        scores = np.array(scores).reshape((len(scores),1))
        scores_t = self.power_transf.transform(scores)

        # store std for 1 and for 2 and 3
        for idx_ in range(1,4):
            idx_float = float(idx_)
            # limit = self.mean + (idx_float * self.std)
            limit = idx_float

            # accumulate results
            results = []
            for index, score in enumerate(scores_t):
                if(score.item() <= limit):
                    image_result = {
                        'image_id': imgs_ids[index],
                        'category_id': 1, # fix this
                        'score': imgs_scores[index],
                        'bbox': imgs_box_coords[index],
                    }
                    results.append(image_result)

            if len(results) > 0:
                # write output
                if val:
                    results_file = f"{dir_filtered_root}/bbox_results_val_std{idx_}.json"
                else:
                    results_file = f"{dir_filtered_root}/bbox_results_std{idx_}.json"
                if os.path.isfile(results_file):
                    os.remove(results_file)
                json.dump(results, open(results_file, 'w'), indent=4)

    def get_likelihoods(self, 
        histograms,    # e.g. 512 x 15
        buckets,       # e.g. 512 x 15
        imgs_2
    ):
        """ 
        Select the best threshold.
        """
        # epsilon to avoid Inf results in logarithm
        eps = 0.0000000001
        featuremaps = torch.stack(imgs_2) # e.g. [387 x 512]
        # init buffer with dims
        num_samples = featuremaps.shape[0]
        likelihoods_dims = torch.zeros(
            num_samples, self.feature_extractor.features_size
        )
        # go  through each dimension, and calculate the likelihood for all samples
        for dimension in range(self.feature_extractor.features_size):
            # calculate the histogram for the given feature, in the labeled dataset
            hist_dim = histograms[dimension, :] #e.g. [512 x 15] -> [15]
            bucks_dim = buckets[dimension, :]   #e.g. [512 x 15] -> [15]

            # take only the values of the current feature for all the observations
            vals_dim = featuremaps[:, dimension] # e.g. [387 x 512] -> [387]

            # fetch the bucket indices for all the observations, for the current feature
            min_buckets = self.find_closest_bucket_all_obs( # e.g. [387]
                vals_dim, bucks_dim
            )

            # evaluate likelihood for the specific
            likelihoods_all = self.get_prob_values_all_obs( # e.g. [387]
                min_buckets, hist_dim
            )

            # store the results with probabilities
            likelihoods_dims[:, dimension] = likelihoods_all.squeeze() # e.g. [387 x 512]

        # calculate the log of the sum of the likelihoods for all the dimensions, obtaining a score per observation
        #THE LOWER THE BETTER
        likelihoods_log = -1 * torch.sum(torch.log(likelihoods_dims + eps), dim=1) # e.g. [387]

        return likelihoods_log
        # # once we got all the batches of the unlabeled data...
        # #----------------------------------------
        # num_bins = 30
        #  # calculate the histogram of the likelihoods
        # (histogram_likelihoods, buckets_likelihoods) = np.histogram(
        #     likelihoods_log.numpy(), bins=num_bins, 
        #     range=None, weights=None, density=None
        # )
        # #----------------------------------------
        # print()

# 
    def get_prob_values_all_obs(self, min_buckets, hist_norm):
        """ Evaluate the histogram values according to the buckets mapped previously
        Params
        :min_buckets: selected buckets according to the feature values. Eg. [404]
        :hist_norm: normalized histogram
        
        Return
        :the likelihood values, according to the histogram evaluated
        """
        # put in a matrix of one column
        min_buckets = min_buckets.unsqueeze(dim=0).transpose(0, 1) # [30] -> [30 x 1]
        # repeat the histograms to perform substraction
        repeated_hist = hist_norm.repeat(1, min_buckets.shape[0]) # e.g. [1 x 6060]
        repeated_hist = repeated_hist.view(-1, hist_norm.shape[0]) # e.g. [1 x 6060] -> [404 x 15]
        # evaluate likelihood for all observations
        likelihoods = repeated_hist.gather(1, min_buckets) # e.g. [404 x 1]
        return likelihoods
            
    def find_closest_bucket_all_obs(self, features, buckets):
        """ Finds the closest bucket position, according to a set of values (from features) received
        Params
        :features (tensor) -> values of features received, to map to the buckets. E.g. [387]
        :buckets (tensor) -> buckets of the previously calculated histogram. E.g. [15]
        
        Return 
        :the list of bucket numbers closest to the buckets received
        """
        # create repeated map to do a matrix substraction, 
        # unsqueezeing and transposing the feature values for all the observations
        features = features.unsqueeze(dim=0).transpose(0, 1) # e.g. [387] -> [387 x 1]
        # rep mat
        expanded_features = features.repeat(1, buckets.shape[0]) # e.g. [387 x 1] -> [387 x 15]
        expanded_features = expanded_features.view(-1, buckets.shape[0]) # e.g. [387 x 1] -> [387 x 15]
        # do substraction
        res = torch.abs(expanded_features - buckets) # [387 x 15]
        # find the closest bin per observation (one observation per row)
        min_buckets = torch.argmin(res, 1) # [387]
        return min_buckets


    def calculate_hist_dataset(self, feature_maps_list, num_bins=15):
        """ 
        For every dimension, create a histogram (with its bucket list).
        
        Params
        :feature_maps (List<tensor>) -> all feature maps in the labeled dataset.
        :num_bins (int) -> num of bins in the histogram.
        Return 
        :histogram
        """
        dimensions = self.feature_extractor.features_size
        # to store results
        histograms_all_features_labeled = torch.zeros((dimensions, num_bins))   # 512 x 15
        buckets_all_features_labeled = torch.zeros((dimensions, num_bins))      # 512 x 15

        # create a single tensor
        feature_maps = torch.stack(feature_maps_list) # 44 x 512
        for dimension in range(0, dimensions):
            # # get a single dimension from the whole batch
            dim_data = feature_maps[:, dimension].numpy() # 44
            
            # calculate a histogram from that dim
            (hist1, bucks1) = np.histogram(
                dim_data, bins=num_bins, 
                range=None, weights=None, density=False
            )
            # manual normalization, np doesnt work
            hist1 = hist1 / hist1.sum()
            # instead of bin edges, get bin mean
            bucks1 = np.convolve(bucks1, [0.5, 0.5], mode='valid')
            hist1 = torch.tensor(np.array(hist1))
            bucks1 = torch.tensor(bucks1)
            # accumulate
            histograms_all_features_labeled[dimension, :] = hist1
            buckets_all_features_labeled[dimension, :] = bucks1
        return (histograms_all_features_labeled, buckets_all_features_labeled) 
        

    def get_threshold(self, scores, percent_to_filter):
        """ Get the threshold according to the list of observations and the percent of data to filter
        Params
        :percent_to_filter (float) -> value from 0 to 1
        Return
        :the threshold
        """
        new_scores_no_validation = scores.copy()
        #percent_to_filter is from  0 to 1
        new_scores_no_validation.sort()
        num_to_filter = int(percent_to_filter * len(new_scores_no_validation))
        threshold = new_scores_no_validation[num_to_filter]
        return threshold
        
    def get_all_features(self, images):
        """
        Extract feature vectors from the images.
        
        Params
        :images (List<tensor>) images to be used to extract features
        """
        features = []
        # get feature maps from the images
        if self.use_sam_embeddings:
            with torch.no_grad():
                for img in images:
                    t_temp = self.feature_extractor.get_embeddings(img)
                    features.append(t_temp.squeeze().cpu())
        else:
            with torch.no_grad():
                for img in images:
                    t_temp = self.feature_extractor(img.unsqueeze(dim=0).to(self.device))
                    features.append(t_temp.squeeze().cpu())
        return features


    def sanity_check_bootstrapping(self, 
        labeled_loader, 
        dir_filtered_root = None, 
        ood_hist_bins = 15):

        """
        Experiment to make sure that the confidence interval from the samples 
        contains the confidence interval of the true population.

        Params
        :labeled_loader: path for the first data bunch, labeled data
        :unlabeled_loader: unlabeled data
        :dir_filtered_root: path for the filtered data to be stored
        :ood_thresh: ood threshold to apply
        :path_reports_ood: path for the ood filtering reports

        Return
        :NULL -> the output of this method is a json file save in a directory.
        """      
        import random
        
        import scipy

        # 1. Get feature maps from the labeled set
        imgs_sets = []
        for batch in labeled_loader:
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from bounding boxes)
                imgs_f, _ = get_foreground(
                    batch, idx, self.trans_norm,
                    self.use_sam_embeddings
                )
                imgs_sets.append(imgs_f)

        # get all features maps using: the extractor + the imgs
        experiment_runs = 35
        l_size = len(imgs_sets)
        my_list = list(range(0,l_size))
        num_k = 6
        
        # run experiments
        for i in range(experiment_runs):
            idx_winners = random.sample(my_list, k=num_k)
            # idx_rest = [i for i in my_list if i is not idx_winner[0]]

            labeled_set, unlabeled_set = [], []
            for k,v in enumerate(imgs_sets):
                # print(k)
                if k in idx_winners:
                    labeled_set += v
                else:
                    unlabeled_set += v

            # labeled_set = [v for k,v in enumerate(feature_maps_list) if k in idx_winner]
            # unlabeled_set = [v for k,v in enumerate(feature_maps_list) if k in idx_rest]

            # if this is correct I need to unravel the lists into a single one
            labeled_set = self.get_all_features(labeled_set)
            unlabeled_set = self.get_all_features(unlabeled_set)

            y_dumpy = np.zeros(len(labeled_set))
            imgs_1, imgs_2, _, _ = train_test_split(
                labeled_set, y_dumpy, 
                train_size = 0.5,
                shuffle=True # shuffle the data before splitting
            )

            # Calculating the histograms from the labeled data
            # go over each dimension and get values from the whole dataset
            (
                histograms_labeled,    # e.g. 512 x 15
                buckets_labeled        # e.g. 512 x 15
            ) = self.calculate_hist_dataset(imgs_1, ood_hist_bins)


            #-------------------------- using labeled -----------------------------
            # get likelihoods
            likelihoods = self.get_likelihoods(
                histograms_labeled,    # e.g. 512 x 15
                buckets_labeled,       # e.g. 512 x 15
                imgs_2
            )

            # transfor the distribution to a gaussian one
            likelihoods_data = likelihoods.reshape((len(likelihoods),1))
            # power transform the raw data
            power = PowerTransformer(method='yeo-johnson', standardize=True)
            data_trans = power.fit_transform(likelihoods_data)
            bootstrap_ci = scipy.stats.bootstrap(
                (data_trans,), np.mean, confidence_level=0.999, method='BCa'
            )
            low, high = bootstrap_ci.confidence_interval
            low = low[0]
            high = high[0]
            #----------------------------------------------------------------------


            #-------------------------- using unlabeled -----------------------------
            # get likelihoods
            likelihoods_u = self.get_likelihoods(
                histograms_labeled,    # e.g. 512 x 15
                buckets_labeled,       # e.g. 512 x 15
                unlabeled_set
            )

            # transfor the distribution to a gaussian one
            likelihoods_data_u = likelihoods_u.reshape((len(likelihoods_u),1))
            # power transform the raw data
            data_trans_u = power.transform(likelihoods_data_u)
            bootstrap_ci_u = scipy.stats.bootstrap(
                (data_trans_u,), np.mean, confidence_level=0.999, method='BCa'
            )
            low_u, high_u = bootstrap_ci_u.confidence_interval
            low_u = low_u[0]
            high_u = high_u[0]
            #----------------------------------------------------------------------

            file_name = f"{dir_filtered_root}/bootstrap_imgs{num_k}.txt"
            with open(file_name, mode='+a') as file_:

                intersection = 0
                is_min_sample = "sample"
                min_low, min_high, max_low, max_high = 0, 0, 0, 0
                if low < low_u:
                    min_low = low
                    min_high = high
                    max_low = low_u
                    max_high = high_u
                else:
                    is_min_sample = "population"
                    min_low = low_u
                    min_high = high_u
                    max_low = low
                    max_high = high
                if min_high > max_low:
                    intersection = 1

                res = f"{intersection}: Min({is_min_sample}). Values: <{low},{high}> | <{low_u},{high_u}>. Labeled data size: {len(labeled_set)}"
                file_.write(f"{res}\n")



        
