import os
import cv2
import json
import random
import torch
import numpy as np

from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground, Transform_To_Models, get_background
from tqdm import tqdm
from numpy import linalg as la
from sklearn.decomposition import TruncatedSVD, PCA
from utils.constants import DimensionalityReductionMethod

class MahalanobisFilter:

    def __init__(self,
                timm_model=None,
                timm_pretrained=True,
                num_classes=1,
                sam_model=None,
                use_sam_embeddings=False,
                is_single_class=True,
                device="cpu", 
                dim_red=None, 
                n_components=10):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        self.mean = None
        self.inv_cov = None
        self.device = device
        self.num_samples = None
        self.num_classes = num_classes
        self.timm_model = timm_model
        self.sam_model = sam_model
        self.is_single_class = is_single_class

        # Dimensionality reduction
        self.dim_red = dim_red
        self.n_components = n_components

        if not use_sam_embeddings:
            # create a model for feature extraction
            feature_extractor = MyFeatureExtractor(
                timm_model, timm_pretrained, num_classes #128, use_fc=True
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

    def is_positive_semidefinite(self, covariance_matrix):
        # We say that A is (positive) semidefinite, and write A >= 0. 
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        positive_semidefinite = all(eigenvalues >= 0)
        return positive_semidefinite
    
    def is_positive_definite(self, matrix):
        # We say that A is (positive) definite, and write A > 0 

        ## Cholesky factorization is a decomposition of a positive-definite matrix into the product of 
        ## a lower triangular matrix and its conjugate transpose. 
        try:
            np.linalg.cholesky(matrix)
            return True  # Cholesky decomposition succeeded, matrix is positive definite
        except np.linalg.LinAlgError:
            return False  # Cholesky decomposition failed, matrix is not positive definite

    def estimate_covariance(self, examples, rowvar=False, inplace=False):
        """
        From Improve Few Shot Classification
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    def fit_normal(self, support_set):
        ## Estimate the mean and covariance matrix for mahalanobis distance
        ## We obtain the pseudoinverse of the covariance matrix because of singular matrix.
        self.mean = torch.mean(support_set, axis=0)
        covariance_matrix = self.estimate_covariance(support_set)
        self.inv_cov = torch.pinverse(covariance_matrix)
        if self.is_positive_definite(self.inv_cov):
            print("The matrix is positive definite.")
        elif self.is_positive_semidefinite(self.inv_cov):
            print("The matrix is positive semi-definite.")
        else:
            print("The matrix is neither positive definite nor positive semi-definite.")

    def fit_regularization(self, support_set, beta=1, context_features=None, lambda_mahalanobis=-1.0):
        # Based on the paper https://github.com/plai-group/simple-cnaps
        ## Lambda is the influence of the covariance matrix of the support set and context features.
        ## Beta is the influence of identical matrix. 
        self.mean = torch.mean(support_set, axis=0)
        
        covariance_matrix = self.estimate_covariance(support_set) #self.estimate_covariance(examples)
        if context_features != None:
            context_covariance_matrix = self.estimate_covariance(context_features) #self.estimate_covariance(context_features)

        if lambda_mahalanobis==-1.0:
            print(lambda_mahalanobis)
            lambda_k_tau = (support_set.size(0) / (support_set.size(0) + 1))
        else:
            lambda_k_tau = lambda_mahalanobis

        self.inv_cov = torch.inverse((lambda_k_tau * covariance_matrix) + ((1 - lambda_k_tau) * context_covariance_matrix) \
                    + (beta * torch.eye(support_set.size(1), support_set.size(1))))

        if self.is_positive_definite(self.inv_cov):
            print("The matrix is positive definite.")
        elif self.is_positive_semidefinite(self.inv_cov):
            print("The matrix is positive semi-definite.")
        else:
            print("The matrix is neither positive definite nor positive semi-definite.")

        
    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        x_mu = mean - values  # batch x features
        dist = torch.diagonal(torch.mm(torch.mm(x_mu, inv_covariance), x_mu.T))
        return dist.sqrt()

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        mean_value = torch.mean(distances).item()
        std_deviation = torch.std(distances).item()
        return distances

    def predict_for_class(self, features: torch.Tensor, cls: int) -> torch.Tensor:
        """Distancia Mahalanobis usando los parámetros de la clase `cls`."""
        saved_mean    = self.mean
        saved_inv_cov = self.inv_cov
        self.mean     = self.class_means[cls]
        self.inv_cov  = self.class_inv_covs[cls]
        dist = self.predict(features)
        self.mean     = saved_mean
        self.inv_cov  = saved_inv_cov
        return dist
    
    def fit_svd(self, x, n_components=10):
        self.svd = TruncatedSVD(n_components=n_components)
        self.svd.fit(x)

    def fit_pca(self, x, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(x)
        
    def run_filter(self,
        labeled_loader,
        unlabeled_loader, validation_loader,
        dir_filtered_root = None, get_background_samples=True,
        num_classes:float=0, 
        mahalanobis_method="regularization", beta=1, seed=10, lambda_mahalanobis=-1.0):

        # 1. Get feature maps from the labeled set
        labeled_imgs = []
        labeled_labels = []
        
        back_imgs_context = []
        back_label_context = []

        for (batch_num, batch) in tqdm(
                    enumerate(labeled_loader), total= len(labeled_loader), desc="Extract images"
            ):            
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
                
                if get_background_samples and self.is_single_class:
                    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

                    imgs_b, labels_b = get_background(
                        batch, idx, self.trans_norm, ss, 
                        num_classes, self.use_sam_embeddings)
                    back_imgs_context += imgs_b
                    back_label_context += labels_b

        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]

        all_labeled_features = self.get_all_features(labeled_imgs)

        if self.is_single_class:
            # Rama original intacta
            if len(back_imgs_context) > 512:
                back_imgs_context = random.Random(seed).sample(back_imgs_context, 512)
            all_background_features = self.get_all_features(back_imgs_context)
            all_context_features = all_labeled_features + random.Random(seed).sample(
                all_background_features, len(all_labeled_features)
            )
            all_features = all_labeled_features + all_background_features
        else:
            # Multiclase: contexto = foreground global (todas las clases juntas)
            all_background_features = []
            all_context_features = all_labeled_features
            all_features = all_labeled_features

        #----------------------------------------------------------------
        if self.is_single_class:
            labels = np.zeros(len(all_labeled_features))
        else:
            labels = np.array(labels)

        # 2. Calculating the mean prototype for the labeled data
        #----------------------------------------------------------------
        all_labeled_features = torch.stack(all_labeled_features)
        all_context_features = torch.stack(all_context_features)        
        all_features = torch.stack(all_features)        

        dim_original = all_features.shape[1]

        # 3. Calculating the sigma (covariance matrix), the distances 
        # with respect of the support features and get the threshold
        #----------------------------------------------------------------
        if self.is_single_class:
            
            # Dimensionality reduction usign SVD for all the features including foreground and background
            if self.dim_red == DimensionalityReductionMethod.SVD:
                self.fit_svd(all_features.detach().numpy(), n_components=self.n_components)
                all_labeled_features = torch.Tensor(self.svd.transform(all_labeled_features.detach().numpy()))
                all_context_features = torch.Tensor(self.svd.transform(all_context_features.detach().numpy()))
            elif self.dim_red == DimensionalityReductionMethod.PCA:
                self.fit_pca(all_features.detach().numpy(), n_components=self.n_components)
                all_labeled_features = torch.Tensor(self.pca.transform(all_labeled_features.detach().numpy()))
                all_context_features = torch.Tensor(self.pca.transform(all_context_features.detach().numpy()))
            
            # Estimate covariance and mean for mahalanobis 
            if mahalanobis_method == "regularization":
                self.fit_regularization(all_labeled_features, beta=beta, context_features=all_context_features, lambda_mahalanobis=lambda_mahalanobis)
            else:
                self.fit_normal(all_labeled_features)

            # Calculate the distances of the support 
            distances = self.predict(all_labeled_features)

            # Calculate threshold using IQR 
            Q1 = np.percentile(distances.numpy(), 25)
            Q3 = np.percentile(distances.numpy(), 75)
            IQR = Q3 - Q1
            threshold = 1.5 * IQR #1.2 * IQR 
            self.threshold = Q3 + threshold

        else:
            # Reducción de dimensionalidad compartida
            if self.dim_red == DimensionalityReductionMethod.SVD:
                self.fit_svd(all_features.detach().numpy(), n_components=self.n_components)
                all_labeled_features = torch.Tensor(self.svd.transform(all_labeled_features.detach().numpy()))
                all_context_features  = torch.Tensor(self.svd.transform(all_context_features.detach().numpy()))
            elif self.dim_red == DimensionalityReductionMethod.PCA:
                self.fit_pca(all_features.detach().numpy(), n_components=self.n_components)
                all_labeled_features = torch.Tensor(self.pca.transform(all_labeled_features.detach().numpy()))
                all_context_features  = torch.Tensor(self.pca.transform(all_context_features.detach().numpy()))

            self.threshold = float('inf')
            self.class_means      = {}
            self.class_inv_covs   = {}
            self.class_thresholds = {}

            unique_classes = np.unique(labels)
            print(f"DEBUG run_filter: labels únicos = {unique_classes}")
            for cls in unique_classes:
                n = int((labels == cls).sum())
                print(f"DEBUG run_filter: clase {cls} → {n} samples")

            # ── NUEVO: construir mapeo cls (0-indexed) → category_id COCO (1-indexed) ──
            # labeled_labels tiene los category_id originales (1-7)
            # labels tiene esos mismos valores menos 1 (0-6)
            self.cls_to_category_id = {}
            for orig_label, shifted_label in zip(labeled_labels, labels):
                cat_id = int(orig_label)    # category_id COCO original (1-7)
                cls_key = int(shifted_label) # clave 0-indexed usada en class_means
                self.cls_to_category_id[cls_key] = cat_id
            print(f"DEBUG mapeo cls→category_id: {self.cls_to_category_id}")

            for cls in unique_classes:
                cls_mask     = labels == cls
                cls_features = all_labeled_features[cls_mask]

                # Mínimo 2 samples para estimar covarianza
                if cls_features.shape[0] < 2:
                    print(f"ADVERTENCIA: clase {cls} tiene {cls_features.shape[0]} sample — se omite.")
                    continue

                if mahalanobis_method == "regularization":
                    self.fit_regularization(
                        cls_features, beta=beta,
                        context_features=all_context_features,
                        lambda_mahalanobis=lambda_mahalanobis
                    )
                else:
                    self.fit_normal(cls_features)

                # Verificar covarianza válida
                if torch.isnan(self.inv_cov).any() or torch.isinf(self.inv_cov).any():
                    print(f"ADVERTENCIA: clase {cls} → inv_cov inválida, se omite.")
                    continue

                self.class_means[cls]    = self.mean.clone()
                self.class_inv_covs[cls] = self.inv_cov.clone()

                distances_cls = self.predict(cls_features)
                distances_np  = distances_cls.numpy()


                # Estas líneas deben estar FUERA del if, al mismo nivel
                Q1       = np.percentile(distances_np, 25)
                Q3       = np.percentile(distances_np, 75)
                IQR      = Q3 - Q1
                max_dist = np.max(distances_np)
                self.class_thresholds[cls] = Q3 + 1.5 * IQR
                print(f"DEBUG clase {cls}: Q3={Q3:.3f}, max={max_dist:.3f}, threshold={self.class_thresholds[cls]:.3f}")    

            last_cls     = list(self.class_means.keys())[-1]
            self.mean    = self.class_means[last_cls]
            self.inv_cov = self.class_inv_covs[last_cls]
            # Para stats_count usamos el umbral medio entre clases como referencia
            distances = torch.cat([
                self.predict_for_class(all_labeled_features[labels == cls], cls)
                for cls in self.class_means
            ])

        stats_count = {
            "lambda_support_set": float(lambda_mahalanobis), 
            "labeled": int(all_labeled_features.shape[0]), 
            "dimension": int(dim_original),
            "reduced_dimension": int(all_labeled_features.shape[1]), 
            "all": int(all_features.shape[0]), 
            "context": int(all_context_features.shape[0]),
            "threshold": float(self.threshold),
            "max": float(np.max(distances.numpy())),
            "positive_definite": bool(self.is_positive_definite(self.inv_cov)),
            "semi_positive_definite": bool(self.is_positive_semidefinite(self.inv_cov))}
        
        self.save_stats(dir_filtered_root, stats_count)

        self.evaluate(unlabeled_loader, dir_filtered_root, "bbox_results")
        self.evaluate(validation_loader, dir_filtered_root, "bbox_results_val")

    def evaluate(self, dataloader, dir_filtered_root, result_name):
        # go through each batch unlabeled
        distances_all = 0

        # keep track of the img id for every sample created by sam
        imgs_ids = []
        imgs_box_coords = []
        imgs_scores = []

        # 3. Get batch of unlabeled // Evaluating the likelihood of unlabeled data
        for (batch_num, batch) in tqdm(
            enumerate(dataloader), total= len(dataloader), desc="Iterate dataloader"
        ):
            unlabeled_imgs = []
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in tqdm(list(range(batch[1]['img_idx'].numel())), desc="Iterate images"):
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
            if len(featuremaps_list) == 0:
                print(f"DEBUG: batch {batch_num} sin propuestas SAM, se omite.")
                continue
            featuremaps = torch.stack(featuremaps_list) # e.g. [387 x 512]

            # Reduce dimensionality
            if self.dim_red == DimensionalityReductionMethod.SVD:
                featuremaps = torch.Tensor(self.svd.transform(featuremaps.detach().numpy()))
            elif self.dim_red == DimensionalityReductionMethod.PCA:
                featuremaps = torch.Tensor(self.pca.transform(featuremaps.detach().numpy()))

            # init buffer with distances
            if self.is_single_class:
                # --- Rama original ---
                distances = self.predict(featuremaps)
                support_set_distances = distances

                # accumulate
                if (batch_num == 0):
                    distances_all = support_set_distances
                else:
                    distances_all = torch.cat((distances_all, support_set_distances), 0)

            else:
                batch_min_distances    = []
                batch_assigned_classes = []

                for feat in featuremaps:
                    best_cls  = None
                    best_dist = float('inf')

                    for cls in self.class_means:
                        self.mean    = self.class_means[cls]
                        self.inv_cov = self.class_inv_covs[cls]
                        dist = self.predict(feat.unsqueeze(0)).item()

                        # Ignorar distancias inválidas o explosivas
                        if np.isnan(dist) or np.isinf(dist) or dist > 1e5:
                            continue

                        if dist < best_dist:
                            best_dist = dist
                            best_cls  = cls

                    batch_min_distances.append(best_dist if best_cls is not None else float('inf'))
                    batch_assigned_classes.append(best_cls)

                support_set_distances = torch.tensor(batch_min_distances)

                if batch_num == 0:
                    distances_all        = support_set_distances
                    assigned_classes_all = batch_assigned_classes
                else:
                    distances_all        = torch.cat((distances_all, support_set_distances), 0)
                    assigned_classes_all += batch_assigned_classes

        # transform data 
        scores = []
        for j in range(0, distances_all.shape[0]):
            scores += [distances_all[j].item()]
        scores = np.array(scores).reshape((len(scores),1))

        limit = self.threshold 
        # accumulate results
        results = []
        count = 0
        print("Scores: ", len(scores))
        print(f"DEBUG: Threshold actual: {limit}")
        if len(scores) > 0:
            print(f"DEBUG: Score min: {np.min(scores)}, Max: {np.max(scores)}, Mean: {np.mean(scores)}")
        else:
            print("DEBUG: ¡OJO! No se generaron scores. Revisa el dataloader o el modelo SAM.")

        if not self.is_single_class:
            print(f"DEBUG thresholds por clase: {self.class_thresholds}")
        for index, score in enumerate(scores):
            if self.is_single_class:
                limit    = self.threshold
                cat_id   = 1
                accepted = score.item() <= limit
            else:
                cls    = assigned_classes_all[index]
                if cls is None:
                    continue
                limit  = self.class_thresholds[cls]
                # Usar mapeo original en lugar de cls+1
                cat_id = self.cls_to_category_id[cls]
                accepted = score.item() <= limit

            if accepted:
                image_result = {
                    'image_id':    imgs_ids[index],
                    'category_id': cat_id,
                    'score':       imgs_scores[index],
                    'bbox':        imgs_box_coords[index],
                }
                results.append(image_result)
                count += 1

        # Debug de distribución de clases asignadas
        if not self.is_single_class and len(results) > 0:
            from collections import Counter
            cat_dist = Counter(r['category_id'] for r in results)
            print(f"DEBUG evaluate: distribución category_id en results = {dict(cat_dist)}")

        print(f"Count: {count}")

        results_file = f"{dir_filtered_root}/{result_name}.json"

        # Eliminar archivo anterior si existe
        try:
            if os.path.isfile(results_file):
                os.remove(results_file)
        except OSError as e:
            print(f"ADVERTENCIA: no se pudo eliminar archivo anterior ({e})")

        # Escribir con reintentos para tolerar NFS busy
        max_attempts = 5
        written = False
        for attempt in range(max_attempts):
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"DEBUG evaluate: archivo escrito con {len(results)} resultados → {results_file}")
                written = True
                break
            except OSError as e:
                print(f"ADVERTENCIA escritura intento {attempt+1}/{max_attempts}: {e}")
                if attempt < max_attempts - 1:
                    import time
                    time.sleep(1)

        if not written:
            # Último recurso: path alternativo
            fallback = f"{dir_filtered_root}/{result_name}_fallback.json"
            with open(fallback, 'w') as f:
                json.dump(results, f, indent=4)
            import shutil
            shutil.copy2(fallback, results_file)
            print(f"DEBUG evaluate: archivo escrito via fallback → {results_file}")

        if len(results) == 0:
            print("ADVERTENCIA: bbox_results vacío. El archivo existe pero no tiene predicciones.")
            
    def save_stats(self, dir_filtered_root, stats):
        file_name_stats = f"{dir_filtered_root}/stats.json"
        with open(file_name_stats, 'w') as file:
            file.write(json.dumps(stats))

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