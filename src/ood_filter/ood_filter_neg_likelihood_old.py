import os
import os
import torch
import torchvision.models as models
# from fastai.vision import *
# from fastai.callbacks import CSVLogger
from numbers import Integral
import torch
import logging
import sys
from torchvision.utils import save_image
import numpy as np
# import pandas as pd
import scipy
from PIL import Image
import torchvision.models.vgg as models2
import torchvision.models as models3
import random

from scipy.stats import entropy
from scipy.spatial import distance
# from utilities.InBreastDataset import InBreastDataset
import matplotlib
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision
from scipy.stats import mannwhitneyu
#from shutil import copyfile
import shutil
from .utils_ood import get_foreground, get_all_features
from data import Transform_Normalization
torch.set_printoptions(threshold=10_000)
from engine.my_backbones import MyBackbone

BATCH_SIZE = 4


class OOD_filter_neg_likelihood:
    def __init__(self):
        """
        OOD filter constructor
        :param model_name: name of the model to get the feature space from, pretrained with imagenet, wideresnet and densenet have been tested so far
        """
        #list of scores, filepaths and labels of unlabeled data processed
        self.scores = []
        self.file_paths = []
        self.labels = []
        self.file_names = []
        #was copied, was test?
        self.info = []


    def find_closest_bucket_all_obs(self, vals_feature_all_obs, buckets):
        """
        Finds the closest bucket position, according to a set of values (from features) received
        :param vals_feature_all_obs: values of features received, to map to the buckets
        :param buckets: buckets of the previously calculated histogram
        :return: returns the list of bucket numbers closest to the buckets received
        """
        # create repeated map to do a matrix substraction, unsqueezeing and transposing the feature values for all the observations
        vals_feature_all_obs = vals_feature_all_obs.unsqueeze(dim=0).transpose(0, 1)
        # rep mat
        repeated_vals_dim_obs = vals_feature_all_obs.repeat(1, buckets.shape[0])
        repeated_vals_dim_obs = repeated_vals_dim_obs.view(-1, buckets.shape[0])
        # do substraction
        substracted_all_obs = torch.abs(repeated_vals_dim_obs - buckets)
        # find the closest bin per observation (one observation per row)
        min_buckets_all_obs = torch.argmin(substracted_all_obs, 1)
        return min_buckets_all_obs



    def get_prob_values_all_obs(self, min_buckets_all_obs, histogram_norm):
        """
        Evaluate the histogram values according to the buckets mapped previously
        :param min_buckets_all_obs: selected buckets according to the feature values
        :param histogram_norm: normalized histogram
        :return: returns the likelihood values, according to the histogram evaluated
        """
        # put in a matrix of one column
        min_buckets_all_obs = min_buckets_all_obs.unsqueeze(dim=0).transpose(0, 1)
        # repeat the histograms to perform substraction
        repeated_histograms = histogram_norm.repeat(1, min_buckets_all_obs.shape[0])
        repeated_histograms = repeated_histograms.view(-1, histogram_norm.shape[0])
        # evaluate likelihood for all observations
        likelihoods_all_obs = repeated_histograms.gather(1, min_buckets_all_obs)
        return likelihoods_all_obs

    def extract_features(self, feature_extractor, batch_tensors1):
        """
        Extract features from a tensor bunch
        :param feature_extractor:
        :param batch_tensors1:
        :return:
        """
        features_bunch1 = feature_extractor(batch_tensors1)
        if (self.model_name != "wideresnet"):

            # pool of non-square window
            #print("features_bunch1 shape ", features_bunch1.shape)
            avg_layer = nn.AvgPool2d((features_bunch1.shape[2], features_bunch1.shape[3]), stride=(1, 1))
            #averaging the features to lower the dimensionality in case is not wide resnet
            features_bunch1 = avg_layer(features_bunch1)
            features_bunch1 = features_bunch1.view(-1, features_bunch1.shape[1] * features_bunch1.shape[2] *
                                                   features_bunch1.shape[3])
        return features_bunch1

    def calculate_hist_dataset(self, tensorbunch1, feature_extractor, dimensions, batch_size=5, num_bins=15, plot=False):
        """
        Calculate feature histogram for the dataset
        param tensorbunch1: tensor dataset to calculate the histogram from
        param feature_extractor: feature extractor to use
        param dimension: dimension to calculate histogram from
        param batch_size: batch to calculate the histogram from
        param num_bins: number of bins of the histogram
        param plot: store pdf plot?
        return histogram
        """
        print("Number of dimensions ", dimensions)
        histograms_all_features_labeled = torch.zeros((dimensions, num_bins), device="cuda:0")
        buckets_all_features_labeled = torch.zeros((dimensions, num_bins), device="cuda:0")
        # random pick of batch observations
        total_number_obs_1 = tensorbunch1.shape[0]
        # print("total number of obs ", total_number_obs_1)
        number_batches = total_number_obs_1 // batch_size
        batch_tensors1 = tensorbunch1[0: batch_size, :, :, :]
        # get the  features from the selected batch
        features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
        # for each dimension, calculate its histogram
        for dimension in range(0, dimensions):
            # get the values of a specific dimension
            values_dimension_bunch_all_batches = features_bunch1[:, dimension].cpu().detach().numpy()
            # Go through each batch...
            for current_batch_num in range(1, number_batches):
                # create the batch of tensors to get its features
                batch_tensors1 = tensorbunch1[(current_batch_num) * batch_size: (current_batch_num + 1) * batch_size, :, :,:]
                # get the  features from the selected batch
                features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
                # get the values of a specific dimension
                values_dimension_bunch1 = features_bunch1[:, dimension].cpu().detach().numpy()
                values_dimension_bunch_all_batches = np.concatenate((values_dimension_bunch_all_batches, values_dimension_bunch1), 0)

            # calculate the histograms
            (hist1, bucks1) = np.histogram(values_dimension_bunch_all_batches, bins=num_bins, range=None, normed=None,
                                           weights=None,
                                           density=False)
            # manual normalization, np doesnt work
            hist1 = hist1 / hist1.sum()
            # instead of bin edges, get bin mean
            bucks1 = np.convolve(bucks1, [0.5, 0.5], mode='valid')
            # normalize the histograms and move it to the gpu
            hist1 = torch.tensor(np.array(hist1), device="cuda:0")
            bucks1 = torch.tensor(bucks1, device="cuda:0")
            histograms_all_features_labeled[dimension, :] = hist1
            buckets_all_features_labeled[dimension, :] = bucks1
        return (histograms_all_features_labeled, buckets_all_features_labeled)


    def plot_histogram(self, bins1, y1, plot_name="histogram.pgf"):
        """
        Histogram plotter in latex, saves the plot to latex
        :param bins1:
        :param y1:
        :param bins2:
        :param y2:
        :param plot_name:
        :param title_plot:
        :return:
        """

        # matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 20
        })
        print("hist1 ")
        print(y1.shape)
        print("buckets ")
        print(bins1[0:-1].shape)
        fig, ax = plt.subplots()
        ax.plot(bins1[0:-1], y1 / y1.sum(), '--')

        ax.set_xlabel('Feature values')
        ax.set_ylabel('Probability density')
        # ax.set_title(title_plot)

        # Tweak spacing to prevent clipping of ylabel

        fig.tight_layout()
        print("Saving fig with name ")
        print(plot_name)
        plt.savefig(plot_name, dpi=400)



    def databunch_to_tensor(self, databunch1):
        """
        Convert the databunch to tensor set
        :param databunch1: databunch to convert
        :return: converted tensor
        """
        # tensor of tensor
        tensor_bunch = torch.zeros(len(databunch1.train_ds), databunch1.train_ds[0][0].shape[0],
                                   databunch1.train_ds[0][0].shape[1], databunch1.train_ds[0][0].shape[2], device="cuda:0")
        for i in range(0, len(databunch1.train_ds)):
            # print(databunch1.train_ds[i][0].data.shape)
            tensor_bunch[i, :, :, :] = databunch1.train_ds[i][0].data.to(device="cuda:0")

        return tensor_bunch


    def get_feature_extractor(self, model):
        """
        Gets the feature extractor from a given model
        :param model: model to use as feature extractor
        :return: returns the feature extractor which can be used later
        """
        path = untar_data(URLs.MNIST_SAMPLE)
        data = ImageDataBunch.from_folder(path)
        # save learner to reload it as a pytorch model
        learner = Learner(data, model, metrics=[accuracy])
        learner.export('/media/Data/saul/Code_Projects/OOD4SSDL/utilities/model/final_model_.pk')
        torch_dict = torch.load('/media/Data/saul/Code_Projects/OOD4SSDL/utilities/model/final_model_.pk')
        # get the model
        model_loaded = torch_dict["model"]
        # put it on gpu!
        model_loaded = model_loaded.to(device="cuda:0")
        model_loaded.eval()
        # usually the last set of layers act as classifier, therefore we discard it
        if(self.model_name == "wideresnet"):
            feature_extractor = model_loaded.features[:-1]
            print("Using wideresenet")

        if(self.model_name == "densenet"):
            #print(model_loaded.features)
            feature_extractor = model_loaded.features[:-2]
            print("Using densenet")
        if (self.model_name == "alexnet"):
            # print(model_loaded.features)
            feature_extractor = model_loaded.features[:-1]
            print("Using alexnet")
        return feature_extractor



    def run_filter(self, path_bunch1, path_bunch2, ood_perc=100, num_unlabeled=3000, name_ood_dataset="SVHN", num_batches=10, size_image=120, batch_size_p=BATCH_SIZE, dir_filtered_root = "/media/Data/saul/Datasets/Covid19/Dataset/OOD_COVID19/OOD_FILTERED/batch_", ood_thresh = 0.8, path_reports_ood = "/reports_ood/" ):
        """
        :param path_bunch1: path for the first data bunch, labeled data
        :param path_bunch2: unlabeled data
        :param ood_perc: percentage of data ood in the unlabeled dataset
        :param num_unlabeled: number of unlabeled observations in the unlabeled dataset
        :param name_ood_dataset: name of the unlabeled dataset
        :param num_batches: Number of batches of the unlabeled dataset to filter
        :param size_image: input image dimensions for the feature extractor
        :param batch_size_p: batch size
        :param dir_filtered_root: path for the filtered data to be stored
        :param ood_thresh: ood threshold to apply
        :param path_reports_ood: path for the ood filtering reports
        :return:
        """
        batch_size_unlabeled = 2
        batch_size_labeled = 10
        global key
        key = "pdf"
        print("Filtering OOD data for dataset at: ", path_bunch2)
        print("OOD threshold ", ood_thresh)
        for num_batch_data in range(0, num_batches):

            # 1. extract features from images

            # load pre-trained model, CORRECTION
            #model = models.alexnet(pretrained=True)
            if(self.model_name == "wideresnet"):
                print("Using wideresnet")
                model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
            if (self.model_name == "densenet"):
                print("Using densenet")
                model = models.densenet121(num_classes=10)
            if (self.model_name == "alexnet"):
                print("Using alexnet")
                model = models.alexnet(num_classes=10)
            # number of histogram bins
            num_bins = 15
            print("Processing batch of labeled and unlabeled data: ", num_batch_data)
            # paths of data for all batches
            #DEBUG INCLUDE TRAIN
            path_labeled = path_bunch1 + "/batch_" + str(num_batch_data) + "/train/"
            path_unlabeled = path_bunch2 + str(num_batch_data) + "/batch_" + str(num_batch_data) + "_num_unlabeled_" + str(
                num_unlabeled) + "_ood_perc_" + str(ood_perc)
            print("path labeled ", path_labeled)
            print("path unlabeled ", path_unlabeled)
            # get the dataset readers
            #  S_l
            databunch_labeled = (ImageList.from_folder(path_labeled)
                                 .split_none()
                                 .label_from_folder()
                                 .transform(size=size_image)
                                 .databunch())
            # S_u
            databunch_unlabeled = (ImageList.from_folder(path_unlabeled)
                                   .split_none()
                                   .label_from_folder()
                                   .transform(size=size_image)
                                   .databunch())
            # get tensor bunches
            tensorbunch_labeled = self.databunch_to_tensor(databunch_labeled)
            tensorbunch_unlabeled = self.databunch_to_tensor(databunch_unlabeled)
            num_obs_unlabeled = tensorbunch_unlabeled.shape[0]
            num_obs_labeled = tensorbunch_labeled.shape[0]
            print("Number of unlabeled observations in batch: ", num_obs_unlabeled)
            print("Number of  labeled observations in batch: ", num_obs_labeled)
            # calculate the number of batches
            num_batches_unlabeled = num_obs_unlabeled // batch_size_unlabeled
            print("Number of unlabeled data batches to process: ", num_batches_unlabeled)
            # DO THIS FOR ALL THE BATCHES
            # get number of features
            batch_tensors1 = tensorbunch_labeled[0:batch_size_p, :, :, :]
            feature_extractor = self.get_feature_extractor(model)
            features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
            num_features = features_bunch1.shape[1]
            # epsilon to avoid Inf results in logarithm
            eps = 0.0000000001
            # likelihoods for all the observations to calculate
            likelihoods_all_obs_unlabeled = torch.zeros(num_obs_unlabeled)
            indices_all_obs_unlabeled = torch.arange(0, num_obs_unlabeled)
            # go through each batch unlabeled
            likelihoods_final_all_obs = 0

            # 2. ------------------------------------------------------------
            print("Calculating the histograms from the labeled data...")
            (histograms_all_features_labeled, buckets_all_features_labeled) = self.calculate_hist_dataset(tensorbunch_labeled,
                                                                                                     feature_extractor,
                                                                                                     num_features,
                                                                                                     batch_size=batch_size_labeled,
                                                                                                     num_bins=num_bins,
                                                                                                     plot=False)
            print("Evaluating likelihood of unlabeled data...")
            for current_batch_num_unlabeled in range(0, num_batches_unlabeled):
                # print("Calculating pdf distance for for feature space of dimensions: ", num_features)
                # get the features for the current unlabeled batch, CORRECTION Batch_number=current_batch_num_unlabeled
                values_features_bunch_unlabeled, batch_indices_unlabeled = self.get_batch_features(tensorbunch_unlabeled,
                                                                                              batch_size_unlabeled=batch_size_unlabeled,
                                                                                              batch_number=current_batch_num_unlabeled,
                                                                                              feature_extractor=feature_extractor)
                num_obs_unlabeled_batch = values_features_bunch_unlabeled.shape[0]
                # init buffer with dims
                likelihoods_all_obs_all_dims = torch.zeros(num_obs_unlabeled_batch, num_features)
                # go  through each dimension, and calculate the likelihood for the whole unlabeled dataset
                for dimension in range(0, num_features):
                    # calculate the histogram for the given feature, in the labeled dataset
                    hist_dim_obs_batch_labeled = histograms_all_features_labeled[dimension, :]
                    bucks_dim_obs_batch_labeled = buckets_all_features_labeled[dimension, :]
                    # take only the values of the current feature for all the observations
                    vals_feature_all_obs_unlabeled = values_features_bunch_unlabeled[:, dimension]
                    # fetch the bucket indices for all the observations, for the current feature
                    min_buckets_all_obs_unlabeled = self.find_closest_bucket_all_obs(vals_feature_all_obs_unlabeled,
                                                                                bucks_dim_obs_batch_labeled)

                    # evaluate likelihood for the specific
                    likelihoods_all_obs_dim_unlabeled = self.get_prob_values_all_obs(min_buckets_all_obs_unlabeled, hist_dim_obs_batch_labeled)
                    # squeeze to eliminate an useless dimension
                    likelihoods_all_obs_all_dims[:, dimension] = likelihoods_all_obs_dim_unlabeled.squeeze()
                # calculate the log of the sum of the likelihoods for all the dimensions, obtaining a score per observation
                #THE LOWER THE BETTER
                likelihoods_all_obs_batch = -1 * torch.sum(torch.log(likelihoods_all_obs_all_dims + eps), 1)
                # store the likelihood for all the observations
                if (current_batch_num_unlabeled == 0):
                    likelihoods_final_all_obs = likelihoods_all_obs_batch
                else:
                    likelihoods_final_all_obs = torch.cat((likelihoods_final_all_obs, likelihoods_all_obs_batch), 0)

            # once we got all the batches of the unlabeled data...
            num_bins = 30
            # calculate the histogram of the likelihoods
            (histogram_likelihoods, buckets_likelihoods) = np.histogram(likelihoods_final_all_obs.numpy(), bins=num_bins,
                                                                        range=None, weights=None,
                                                                        density=None)
            # store per file scores
            file_names = []
            file_paths = []
            scores = []
            labels = []
            #create final summary
            for j in range(0, likelihoods_final_all_obs.shape[0]):
                file_name = os.path.splitext(os.path.basename(databunch_unlabeled.items[j]))[0]
                file_paths += [databunch_unlabeled.items[j]]
                file_names += [file_name]
                scores += [likelihoods_final_all_obs[j].item()]
                labels += [databunch_unlabeled.y[j]]
            #store filtering information
            self.scores = scores
            self.file_paths = file_paths
            self.labels = labels
            self.file_names = file_names

            # copy filtered information to the given folder
            dir_filtered = dir_filtered_root + "/batch_" + str(num_batch_data) + "/batch_" + str(num_batch_data) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc) + "/"
            self.copy_filtered_observations(dir_root=dir_filtered, percent_to_filter=ood_thresh)


            print("file_names lenght ", len(file_names))
            print("scores ", len(scores))
            dict_csv = {'File_names': file_names,
                        'Scores': scores, "Info":self.info}
            dataframe = pd.DataFrame(dict_csv, columns=['File_names', 'Scores', 'Info'])
            try:
                os.makedirs(path_reports_ood)
            except:
                a = 0
            dataframe.to_csv(path_reports_ood + 'scores_files_batch' + str(num_batch_data) + '.csv', index=False, header=True)
            print(dataframe)


    def run_filter_fabian(self, 
        labeled_loader, 
        unlabeled_loader,  
        dir_filtered_root = ".", 
        path_reports_ood = "output", 
        ood_thresh = 0.8, 
        ood_hist_bins = 15,
        num_classes=1,
        timm_model=None,
        timm_pretrained=True):
        """
        Params
        :labeled_loader: path for the first data bunch, labeled data
        :unlabeled_loader: unlabeled data
        :dir_filtered_root: path for the filtered data to be stored
        :ood_thresh: ood threshold to apply
        :path_reports_ood: path for the ood filtering reports

        Return
        :return:
        """
        # epsilon to avoid Inf results in logarithm
        eps = 0.0000000001
        labeled_imgs = []
        labeled_labels = []
        # transformations
        transf = Transform_Normalization(
            size=33, force_resize=False, keep_aspect_ratio=True
        )
        
        # ITERATE: BATCH
        for batch in labeled_loader:
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in list(range(batch[1]['img_idx'].numel())):
                # get foreground samples (from bounding boxes)
                imgs_f, labels_f = get_foreground(batch, idx, transf)

                labeled_imgs += imgs_f
                labeled_labels += labels_f

        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]

        # create a model for feature extraction
        feature_extractor = MyBackbone(
            timm_model, timm_pretrained, num_classes
        ).to('cuda')

        # get all features maps using: the extractor + the imgs
        feature_maps = get_all_features(labeled_imgs, feature_extractor)

        # go over each dimension and get values from the whole dataset
        print()


    def copy_filtered_observations(self, dir_root, percent_to_filter):
        """
        Copy filtered observations applying the thresholds
        :param dir_root: directory where to copy the filtered data
        :param   percent_to_filter: percent of observations to keep
        :return:
        """

        thresh = self.get_threshold(percent_to_filter)
        print("Threshold ", thresh)
        print("Percent to threshold: ", percent_to_filter)
        num_selected = 0
        #store info about the observation filtering
        self.info = [""] * len(self.scores)
        #only filter training
        for i in range(0, len(self.scores)):
            #print("self.file_paths[i]")
            #print(self.file_paths[i])
            #print("Path ", self.file_paths[i])
            #print("Current score ", self.scores[i], " of observation ", i, " condition ", "test" not in str(self.file_paths[i]))
            if(self.scores[i] <= thresh and "test" not in str(self.file_paths[i])):
                num_selected += 1
                rand_class = random.randint(0, self.num_classes - 1)
                path_dest = dir_root + "/train/" + str(rand_class) + "/"
                path_origin = self.file_paths[i]

                try:
                    os.makedirs(path_dest)
                except:
                    a = 0
                file_name = os.path.basename(self.file_paths[i])
                #print("File to copy", path_origin)
                #print("Path to copy", path_dest + file_name)
                shutil.copyfile(path_origin, path_dest + file_name)
                self.info[i] =  "Copied, is training observation lower than thresh " + str(thresh)

            if("test" in str(self.file_paths[i])):
                path_dest = dir_root + "/test/" + str(self.labels[i]) + "/"
                path_origin = self.file_paths[i]
                try:
                    os.makedirs(path_dest)
                except:
                    a = 0
                file_name = os.path.basename(self.file_paths[i])
                #print("File to copy", path_origin)
                #print("Path to copy", path_dest + file_name)
                shutil.copyfile(path_origin, path_dest + file_name)
                self.info[i] = "Copied, is test observation"
        print("Number of unlabeled observations preserved: ", num_selected)

    def get_threshold(self, percent_to_filter):
        """
        Get the threshold according to the list of observations and the percent of data to filter
        :param percent_to_filter: value from 0 to 1
        :return: the threshold
        """
        new_scores_no_validation = []
        for i in range(0, len(self.scores)):
            #bug fixed!!
            if("test" not in str(self.file_paths[i])):
                new_scores_no_validation += [self.scores[i]]

        #percent_to_filter is from  0 to 1
        new_scores_no_validation.sort()
        num_to_filter = int(percent_to_filter * len(new_scores_no_validation))
        threshold = new_scores_no_validation[num_to_filter]
        return threshold



    def get_batch_features(self, tensorbunch_unlabeled, batch_size_unlabeled, batch_number, feature_extractor):
        """
        Get the batcch of features using a specific feature extractor
        :param tensorbunch_unlabeled: tensorbunch to evaluate using the feature extractor
        :param batch_size_unlabeled: batch size to use during evaluation
        :param batch_number: batch number to evaluate
        :param feature_extractor: feature extractor to use
        :return: features extracted
        """
        total_number_obs_1 = tensorbunch_unlabeled.shape[0]
        # create the batch of tensors to get its features
        batch_tensors1 = tensorbunch_unlabeled[
                         batch_number * batch_size_unlabeled:(batch_number + 1) * batch_size_unlabeled, :, :, :]
        # batch indices for accountability
        batch_indices = torch.arange(batch_number * batch_size_unlabeled, (batch_number + 1) * batch_size_unlabeled)
        # print("batch tensors ", batch_tensors1.shape)
        # get the  features from the selected batch
        features_bunch1 = self.extract_features(feature_extractor, batch_tensors1)
        # get the values of a specific dimension
        # values_dimension_bunch1 = features_bunch1[:, :].cpu().detach().numpy()
        values_dimension_bunch1 = features_bunch1[:, :]
        return values_dimension_bunch1, batch_indices


def run_tests_pdf():
    ood_filter_neg_likelihood = OOD_filter_neg_likelihood(model_name = "wideresnet")

    """
    :param distance: distance_str
    :return:
    """
    #S_l is the IID data for indiana i.e image_67.jpg
    #"/media/Data/saul/Datasets/Covid19/Dataset/batches_labeled_undersampled_in_dist_BINARY_INDIANA_30_val_40_labels"
    #S_u is contaminated dataset
    #/media/Data/saul/Datasets/Covid19/Dataset/OOD_COVID19/OOD_CR_25

    ood_filter_neg_likelihood.run_filter(
        path_bunch1="/media/Data/saul/Datasets/Covid19/Dataset/batches_labeled_undersampled_in_dist_BINARY_INDIANA_30_val_40_labels",
        path_bunch2="/media/Data/saul/Datasets/Covid19/Dataset/OOD_COVID19/OOD_CR_50/batch_", ood_perc=50,
        num_unlabeled=90, name_ood_dataset="SVHN", num_batches=10, size_image=100, batch_size_p=BATCH_SIZE, dir_filtered_root = "/media/Data/saul/Datasets/Covid19/Dataset/OOD_COVID19/OOD_FILTERED_CR_50_THRESH_70_ALL/", ood_thresh = 0.7, path_reports_ood = "/media/Data/saul/Code_Projects/OOD4SSDL/utilities/reports_ood/")







#simple debugging test with MNIST data
#run_tests_pdf()