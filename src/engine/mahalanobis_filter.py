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
        self.mean = None #Debe ser arreglo
        self.inv_cov = None #Debe ser arreglo
        self.device = device #Debe ser arreglo
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
        covariance_matrix = self.estimate_covariance(support_set)
        inv_cov = torch.pinverse(covariance_matrix)
        if self.is_positive_definite(inv_cov):
            print("The matrix is positive definite.")
        elif self.is_positive_semidefinite(inv_cov):
            print("The matrix is positive semi-definite.")
        else:
            print("The matrix is neither positive definite nor positive semi-definite.")
        return inv_cov

    def fit_regularization(self, support_set, beta=1, context_features=None, lambda_mahalanobis=-1.0): 
        """
        Estima la media y la covarianza inversa regularizada para el support set.
        Basado en Simple CNAPS (https://github.com/plai-group/simple-cnaps).

        La covarianza regularizada combina tres términos:
            Σ_reg = λ * Σ_support + (1-λ) * Σ_context + β * I
        donde:
            Σ_support : covarianza del support set (la clase de interés)
            Σ_context : covarianza del contexto (otras clases o background)
            β * I     : término de regularización para garantizar invertibilidad
            λ         : peso del support set. Si -1.0, se calcula como
                        n/(n+1) donde n = número de samples del support set.

        Params:
            support_set       (Tensor): features del support set. Shape: [N, D].
            beta              (float) : peso de la matriz identidad.
            context_features  (Tensor): en single-class = foreground + background.
                                        en multiclase   = features de todas las otras clases.
                                        Si es None, se usa solo la covarianza del support set
                                        con peso completo (lambda=1).
            lambda_mahalanobis(float) : peso del support set. -1.0 = automático.

        Returns:
            None. Actualiza self.mean y self.inv_cov.
        """
        
        covariance_matrix     = self.estimate_covariance(support_set)
        if context_features != None:
            context_covariance_matrix = self.estimate_covariance(context_features)


        # Lambda automático: más samples → más peso al support set
        if lambda_mahalanobis == -1.0:
            lambda_k_tau = support_set.size(0) / (support_set.size(0) + 1)
        else:
            lambda_k_tau = lambda_mahalanobis
            
        inv_cov = (
                (lambda_k_tau       * covariance_matrix)  +
                ((1 - lambda_k_tau) * context_covariance_matrix) +
                (beta               * torch.eye(support_set.size(1), support_set.size(1)))
            ) # Calcular la inversa en general (como danny lo hizo) y a su vez como está el paper

        if self.is_positive_definite(inv_cov):
            print("The matrix is positive definite.")
        elif self.is_positive_semidefinite(inv_cov):
            print("The matrix is positive semi-definite.")
        else:
            print("The matrix is neither positive definite nor positive semi-definite.")


        return inv_cov

        
    @staticmethod
    def mahalanobis_distance( # Recibe vectores sobre cada clase y comparando con el value de entrada
        values: torch.Tensor,
        mean: torch.Tensor,
        inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula la distancia Mahalanobis entre un batch de vectores y una distribución.
        Implementación vectorizada por batches.

        Referencia: https://github.com/ORippler/gaussian-ad-mvtec

        La fórmula es: d(x) = sqrt( (μ-x)^T * Σ^{-1} * (μ-x) )

        Params:
            values        (Tensor): batch de feature vectors. Shape: [N, D].
            mean          (Tensor): media de la distribución. Shape: [D] o [1, D].
            inv_covariance(Tensor): covarianza inversa. Shape: [D, D].

        Returns:
            dist (Tensor): distancias Mahalanobis. Shape: [N].
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        # x_mu = μ - x  →  Shape: [N, D]
        x_mu = mean - values
        # (x_mu @ Σ^{-1}) @ x_mu^T  →  Shape: [N, N]
        # La diagonal contiene la distancia al cuadrado de cada sample
        dist = torch.diagonal(torch.mm(torch.mm(x_mu, inv_covariance), x_mu.T))
        return dist.sqrt()

    def predict(self, embeddings, cls_mean, cls_inv_cov):
        distances = self.mahalanobis_distance(embeddings, cls_mean, cls_inv_cov)
        mean_value = torch.mean(distances).item()
        std_deviation = torch.std(distances).item()
        #print(f"Flow8: distancias Mahalanobis del support set, mean, std para la clase {self.crr_cls_print}")
        #print(distances)
        #print("Mean predicted:", mean_value)
        #print("Standard Deviation predicted:", std_deviation)
        return distances
    
    def fit_svd(self, x, n_components=10):
        self.svd = TruncatedSVD(n_components=n_components)
        self.svd.fit(x)

    def fit_pca(self, x, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(x)
        
    def run_filter(self,
        labeled_loader,
        unlabeled_loader,
        validation_loader,
        dir_filtered_root=None,
        get_background_samples=True,
        num_classes: float=0,
        mahalanobis_method="regularization",
        beta=1,
        seed=10,
        lambda_mahalanobis=-1.0):
        """
        Método principal del filtro Mahalanobis. Entrena la distribución gaussiana
        sobre el conjunto labeled y luego evalúa sobre test y validación.

        Params:
            labeled_loader   (DataLoader): loader con imágenes etiquetadas (support set).
            unlabeled_loader (DataLoader): loader con imágenes de test a filtrar.
            validation_loader(DataLoader): loader con imágenes de validación.
            dir_filtered_root     (str)  : carpeta de salida para resultados.
            get_background_samples(bool) : si True, extrae muestras de background
                                           (solo en modo single-class).
            num_classes           (float): número de clases (usado en get_background).
            mahalanobis_method    (str)  : método de estimación de covarianza.
                                           'regularization' o 'normal'.
            beta                  (float): peso de la matriz identidad en regularización.
            seed                  (int)  : semilla para reproducibilidad.
            lambda_mahalanobis    (float): peso del support set en la covarianza
                                           regularizada. -1.0 = calculado automático.

        Returns:
            None. Escribe bbox_results.json y bbox_results_val.json en dir_filtered_root.
        """

        # ─────────────────────────────────────────────────────────────────
        # PASO 1: Extraer imágenes y labels del conjunto labeled
        # Itera sobre el labeled_loader para obtener:
        #   - labeled_imgs  : crops de foreground (bounding boxes del GT)
        #   - labeled_labels: category_id COCO de cada crop (1-indexed)
        # En modo single-class también extrae muestras de background
        # usando Selective Search como clase negativa.
        # ─────────────────────────────────────────────────────────────────
        labeled_imgs   = []
        labeled_labels = []
        back_imgs_context  = []
        back_label_context = []

        for (batch_num, batch) in tqdm(
            enumerate(labeled_loader), total=len(labeled_loader), desc="Extract labeled images"
        ):
            # batch = (tensor_imgs, dict_metadata)
            # batch[1]['img_idx'].numel() = número de imágenes en el batch
            for idx in range(batch[1]['img_idx'].numel()):

                # get_foreground: recorta cada bbox del GT de la imagen idx
                # Retorna:
                #   imgs_f  : lista de tensores PIL crop de cada bbox
                #   labels_f: lista de category_id (int) de cada bbox
                imgs_f, labels_f = get_foreground(
                    batch, idx, self.trans_norm, self.use_sam_embeddings
                )
                labeled_imgs   += imgs_f
                labeled_labels += labels_f
                print(f"Flow1: batch {batch_num}, img_idx {idx} → {len(imgs_f)} foreground crops, labels: {labels_f}")

                # Solo en single-class: obtener background como clase negativa
                # get_background usa Selective Search para proponer regiones
                # que no se solapan con los GT boxes → clase negativa
                if get_background_samples and self.is_single_class:
                    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                    imgs_b, labels_b = get_background(
                        batch, idx, self.trans_norm, ss,
                        num_classes, self.use_sam_embeddings
                    )
                    back_imgs_context  += imgs_b
                    back_label_context += labels_b

        # COCO usa category_id desde 1, internamente trabajamos desde 0
        # labeled_labels: [1,2,3,...] → labels: [0,1,2,...]
        labels = [int(i - 1) for i in labeled_labels]
        print(f"Flow2: labels mapeados a 0-indexed: {labels[:10]} (primeros 10)")

        # ─────────────────────────────────────────────────────────────────
        # PASO 2: Extraer feature vectors de todas las imágenes labeled
        # get_all_features pasa cada crop por el backbone (timm o SAM)
        # y retorna una lista de tensores 1D (un vector por crop)
        # ─────────────────────────────────────────────────────────────────
        all_labeled_features = self.get_all_features(labeled_imgs)
        print(f"Flow3: featuresV de {len(all_labeled_features)} crops labeled")

        # ─────────────────────────────────────────────────────────────────
        # PASO 3: Construir contexto y features totales según el modo
        #
        # Single-class:
        #   - all_features       = foreground + background
        #   - all_context_features = foreground + muestra aleatoria de background
        #   - labels             = todos 0 (una sola clase)
        #
        # Multiclase:
        #   - all_features       = solo foreground (todas las clases)
        #   - all_context_features = mismo foreground global
        #   - labels             = preserva el category_id real de cada crop
        # ─────────────────────────────────────────────────────────────────
        if self.is_single_class:
            if len(back_imgs_context) > 512:
                back_imgs_context = random.Random(seed).sample(back_imgs_context, 512)
            all_background_features = self.get_all_features(back_imgs_context)
            all_context_features = all_labeled_features + random.Random(seed).sample(
                all_background_features, len(all_labeled_features)
            )
            all_features = all_labeled_features + all_background_features
            # En single-class todos los foreground son clase 0
            labels = np.zeros(len(all_labeled_features))
        else:
            all_background_features = []
            all_context_features    = all_labeled_features
            all_features            = all_labeled_features
            # En multiclase preservar los labels reales (0-indexed)
            labels = np.array(labels)

        # Convertir listas a tensores para operaciones matriciales
        all_labeled_features = torch.stack(all_labeled_features)
        all_context_features = torch.stack(all_context_features)
        all_features         = torch.stack(all_features)
        dim_original         = all_features.shape[1]


        # ─────────────────────────────────────────────────────────────────
        # PASO 4: Reducción de dimensionalidad + estimación de la
        # distribución Mahalanobis
        # ─────────────────────────────────────────────────────────────────
        if self.is_single_class:
            self._fit_single_class(
                all_features, all_labeled_features, all_context_features,
                mahalanobis_method, beta, lambda_mahalanobis
            )
            distances = self.predict(all_labeled_features)

        else:
            distances, means, inv_covs, thresholds, cls_to_category_id, all_svd = self._fit_multiclass(
                all_features, all_labeled_features, all_context_features,
                labels, labeled_labels,
                mahalanobis_method, beta, lambda_mahalanobis
            )

        # ─────────────────────────────────────────────────────────────────
        # PASO 5: Guardar estadísticas del entrenamiento
        # ─────────────────────────────────────────────────────────────────
        stats_count = {
            "lambda_support_set"  : float(lambda_mahalanobis),
            "labeled"             : int(all_labeled_features.shape[0]),
            "dimension"           : int(dim_original),
            "reduced_dimension"   : int(all_svd.shape[1]),
            "all"                 : int(all_features.shape[0]),
            "context"             : int(all_context_features.shape[0]),
        }
        
        if self.is_single_class:
            stats_count.update({
                "threshold"              : float(self.threshold),
                "max"                    : float(np.max(distances.numpy())),
                "positive_definite"      : bool(self.is_positive_definite(self.inv_cov)),
                "semi_positive_definite" : bool(self.is_positive_semidefinite(self.inv_cov)),
            })
        else:
            per_class_stats = {}
            for cls in inv_covs:
                per_class_stats[str(cls)] = {
                    "threshold"              : float(thresholds[cls]),
                    "max_distance"           : float(np.max(distances[cls])),
                    "positive_definite"      : bool(self.is_positive_definite(inv_covs[cls])),
                    "semi_positive_definite" : bool(self.is_positive_semidefinite(inv_covs[cls])),
                }
            stats_count["classes"] = per_class_stats                        

        self.save_stats(dir_filtered_root, stats_count)

        # ─────────────────────────────────────────────────────────────────
        # PASO 6: Evaluar sobre test y validación
        # evaluate itera el loader, genera propuestas SAM, calcula
        # distancias Mahalanobis y filtra por threshold → bbox_results.json
        # ─────────────────────────────────────────────────────────────────
        self.evaluate(unlabeled_loader,  dir_filtered_root, "bbox_results", means, inv_covs, thresholds, cls_to_category_id)
        self.evaluate(validation_loader, dir_filtered_root, "bbox_results_val", means, inv_covs, thresholds, cls_to_category_id)

    def _fit_multiclass(self, all_features, all_labeled_features, all_context_features,
                        labels, labeled_labels,
                        mahalanobis_method, beta, lambda_mahalanobis):
        """
        Ajusta una distribución Mahalanobis independiente por cada clase.
        Aplica una reducción de dimensionalidad compartida y estima media,
        covarianza inversa y threshold para cada clase presente en el labeled set.

        Params:
            all_features         (Tensor)    : todos los features labeled (para fit SVD/PCA).
            all_labeled_features (Tensor)    : features del support set.
            all_context_features (Tensor)    : contexto global (foreground de todas las clases).
            labels               (np.array)  : category_id 0-indexed de cada feature.
            labeled_labels       (list)      : category_id COCO original (1-indexed).
            mahalanobis_method   (str)       : 'regularization' o 'normal'.
            beta                 (float)     : peso de la identidad en regularización.
            lambda_mahalanobis   (float)     : peso del support set.

        Returns:
            distances (Tensor): distancias concatenadas de todas las clases válidas,
                                 usadas para estadísticas de entrenamiento.
        """
        # Reducción de dimensionalidad compartida entre todas las clases
        # Se entrena una sola vez para que todas las clases vivan en el mismo espacio
        if self.dim_red == DimensionalityReductionMethod.SVD:
            self.fit_svd(all_features.detach().numpy(), n_components=self.n_components)
            all_labeled_features = torch.Tensor(self.svd.transform(all_labeled_features.detach().numpy()))
        elif self.dim_red == DimensionalityReductionMethod.PCA:
            self.fit_pca(all_features.detach().numpy(), n_components=self.n_components)
            all_labeled_features = torch.Tensor(self.pca.transform(all_labeled_features.detach().numpy()))

        # Inicializar estructuras por clase
        threshold      = float('inf')  # no hay un threshold global en multiclase
        class_means      = {}           # media por clase
        class_inv_covs   = {}           # covarianza inversa por clase
        class_thresholds = {}           # threshold IQR por clase
        class_distances = {}
        
        unique_classes = np.unique(labels)        

        # Construir mapeo: índice interno 0-based → category_id COCO 1-based
        # Necesario porque COCO espera category_id originales en las predicciones
        # labeled_labels: [1,2,3,...] / labels: [0,1,2,...]
        cls_to_category_id = {}
        
        for orig_label, shifted_label in zip(labeled_labels, labels):
            cls_to_category_id[int(shifted_label)] = int(orig_label)
        
        print(f"Flow4: mapeo a diccionario de category_ids entre las clases originales y las clases 0-indexed: \n {cls_to_category_id}")

        # Por cada clase: estimar gaussiana y calcular threshold propio
        for cls in unique_classes:
            cls_mask     = labels == cls
            cls_features = all_labeled_features[cls_mask]
            print(f"Flow5: máscara de clases para clase {cls} → {cls_mask.sum()} samples")
            print(f"Flow6: features totales {cls} → {all_labeled_features.shape[0]} samples")
            print(f"Flow6.1: features de clase {cls} → {cls_features.shape[0]} samples")

            # Necesitamos al menos 2 samples para estimar covarianza
            if cls_features.shape[0] < 2:
                print(f"ADVERTENCIA: clase {cls} tiene {cls_features.shape[0]} sample — se omite.")
                continue
            
            cls_mean = torch.mean(cls_features, axis=0)

            if mahalanobis_method == "regularization":
                # Contexto = features de TODAS las otras clases
                # Esto permite que la covarianza de cada clase sea contrastiva
                other_features = all_labeled_features[~cls_mask]
                context = other_features if other_features.shape[0] >= 2 else all_context_features
                print(f"Flow7: contexto para clase {cls} → {context.shape[0]} samples")

                # fit_regularization: estima self.mean y self.inv_cov para cls_features
                # usando context_features como distribución de referencia
                cls_inv_cov = self.fit_regularization(
                    cls_features, beta=beta,
                    context_features=context,
                    lambda_mahalanobis=lambda_mahalanobis
                )
            else:
                # fit_normal: estima self.mean y self.inv_cov sin regularización
                cls_inv_cov = self.fit_normal(cls_features)

            # Validar que la covarianza sea numéricamente estable
            if cls_inv_cov is None:
                print(f"ADVERTENCIA: clase {cls} -> inv_cov = None, se omite.")
                continue

            if torch.isnan(cls_inv_cov).any() or torch.isinf(cls_inv_cov).any():
                print(f"ADVERTENCIA: clase {cls} -> inv_cov inválida, se omite.")
                continue

            # Guardar parámetros de esta clase
            print(f"Flow9: media y covarianza inversa estimadas para clase {cls} (category_id COCO {cls_to_category_id[cls]}), mean sample: {cls_mean[:5]}, inv_cov sample: {cls_inv_cov.flatten()[:5]}")
            class_means[cls]    = cls_mean.clone()
            class_inv_covs[cls] = cls_inv_cov.clone()

            # Calcular distancias del support set de esta clase para el threshold
            # predict: calcula distancia Mahalanobis usando self.mean y self.inv_cov
            self.crr_cls_print = cls
            cls_distance = self.predict(cls_features, cls_mean, cls_inv_cov).numpy()
            class_distances[cls] = cls_distance

            if np.isnan(cls_distance).any() or np.isinf(cls_distance).any():
                print(f"ADVERTENCIA: clase {cls} → distancias inválidas, se omite.")
                del class_means[cls]
                del class_inv_covs[cls]
                continue

            # Threshold por clase usando criterio de Tukey (Q3 + 1.5*IQR)
            Q1  = np.percentile(cls_distance, 25)
            Q3  = np.percentile(cls_distance, 75)
            IQR = Q3 - Q1
            class_thresholds[cls] = Q3 + 1.5 * IQR
            
        
        print(f"|===============Flow10: resultados finales por clase de fit===============")
        print(f"Clases (Medias - muestra 5 val): {json.dumps({int(k): v.tolist()[:5] for k, v in class_means.items()}, indent=4)}")
        print(f"Class inv covs (Muestra 2x2): {json.dumps({int(k): v[:2, :2].tolist() for k, v in class_inv_covs.items()}, indent=4)}")
        print(f"Thresholds: {json.dumps({int(k): float(v) for k, v in class_thresholds.items()}, indent=4)}")

        if len(class_means) == 0:
            raise RuntimeError(
                "Ninguna clase tiene suficientes samples. "
                f"Necesitas >= 2 samples por clase."
            )

        return class_distances, class_means, class_inv_covs, class_thresholds, cls_to_category_id, all_labeled_features

    def evaluate(self, dataloader, dir_filtered_root, result_name, class_means=None, class_inv_covs=None, class_thresholds=None, cls_to_category_id=None):
        """
        Aplica el filtro Mahalanobis sobre un dataloader usando propuestas de SAM.
        Para cada imagen genera regiones candidatas con SAM, extrae sus features,
        calcula la distancia Mahalanobis y acepta o rechaza cada región según
        el threshold (global en single-class, por clase en multiclase).

        Params:
            dataloader       (DataLoader): loader de imágenes a evaluar.
            dir_filtered_root(str)       : carpeta donde guardar el resultado.
            result_name      (str)       : nombre base del archivo de salida
                                           (sin .json). Ej: 'bbox_results'.

        Returns:
            None. Escribe {result_name}.json en dir_filtered_root con formato
            COCO: lista de dicts con image_id, category_id, score, bbox.
        """
        imgs_ids        = []  # image_id COCO de cada propuesta SAM
        imgs_box_coords = []  # coordenadas bbox de cada propuesta
        imgs_scores     = []  # score de confianza SAM de cada propuesta
        distances_all   = 0   # acumulador de distancias Mahalanobis

        # En multiclase también acumulamos la clase asignada a cada propuesta
        assigned_class = []

        # ─────────────────────────────────────────────────────────────────
        # PASO 1: Iterar el dataloader y generar propuestas SAM
        # Por cada imagen: SAM genera N regiones candidatas (bboxes)
        # Cada región se convierte en un feature vector
        # ─────────────────────────────────────────────────────────────────
        for (batch_num, batch) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Iterate dataloader"
        ):
            unlabeled_imgs = []

            for idx in range(batch[1]['img_idx'].numel()):
                # get_unlabeled_samples: SAM genera propuestas de región para
                # la imagen idx del batch.
                # Retorna:
                #   imgs_s    : lista de tensores crop de cada propuesta
                #   box_coords: lista de bboxes [x,y,w,h] de cada propuesta
                #   scores    : lista de scores de confianza SAM
                imgs_s, box_coords, scores = self.sam_model.get_unlabeled_samples(
                    batch, idx, self.trans_norm, self.use_sam_embeddings
                )
                unlabeled_imgs += imgs_s
                imgs_ids        += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
                imgs_box_coords += box_coords
                imgs_scores     += scores

            # ─────────────────────────────────────────────────────────────
            # PASO 2: Extraer features y reducir dimensionalidad
            # get_all_features: pasa cada crop por el backbone
            # Retorna lista de tensores 1D (un vector por propuesta)
            # ─────────────────────────────────────────────────────────────
            featuremaps_list = self.get_all_features(unlabeled_imgs)
            if len(featuremaps_list) == 0:
                continue
            featuremaps = torch.stack(featuremaps_list)

            # Aplicar la misma reducción entrenada en run_filter
            if self.dim_red == DimensionalityReductionMethod.SVD:
                featuremaps = torch.Tensor(self.svd.transform(featuremaps.detach().numpy()))
            elif self.dim_red == DimensionalityReductionMethod.PCA:
                featuremaps = torch.Tensor(self.pca.transform(featuremaps.detach().numpy()))

            # ─────────────────────────────────────────────────────────────
            # PASO 3: Calcular distancias Mahalanobis
            # Multiclase: nearest-centroid en espacio Mahalanobis
            #   Para cada feature, calcular distancia a cada clase
            #   y asignar la clase con menor distancia
            # ─────────────────────────────────────────────────────────────
            if self.is_single_class:
                distances = self.predict(featuremaps)
                distances_all = distances if batch_num == 0 else torch.cat((distances_all, distances), 0)

            else:
                batch_min_distances    = []
                batch_assigned_classes = []

                for feat in featuremaps:
                    best_cls  = None
                    best_dist = float('inf')

                    # Por cada clase entrenada: calcular distancia Mahalanobis
                    # usando los parámetros específicos de esa clase
                    # predict: usa temporalmente self.mean y self.inv_cov de cls
                    for cls in class_means:
                        mean    = class_means[cls]
                        inv_cov = class_inv_covs[cls]
                        dist = self.predict(feat.unsqueeze(0), mean, inv_cov).item()
                        

                        if np.isnan(dist) or np.isinf(dist) or dist > 1e5:
                            continue
                        if dist < best_dist:
                            best_dist = dist
                            best_cls  = cls

                    batch_min_distances.append(best_dist if best_cls is not None else float('inf'))
                    batch_assigned_classes.append(best_cls)

                batch_distances = torch.tensor(batch_min_distances)
                if batch_num == 0:
                    distances_all        = batch_distances
                    assigned_class = batch_assigned_classes
                else:
                    distances_all        = torch.cat((distances_all, batch_distances), 0)
                    assigned_class += batch_assigned_classes
                    

        # ─────────────────────────────────────────────────────────────────
        # PASO 4: Filtrar propuestas por threshold y construir resultados
        # Single-class : threshold global, category_id = 1 siempre
        # Multiclase   : threshold por clase, category_id = COCO id real
        # ─────────────────────────────────────────────────────────────────
        scores  = np.array([distances_all[j].item() for j in range(distances_all.shape[0])]).reshape(-1, 1)
        results = []

        for index, score in enumerate(scores):
            if self.is_single_class:
                limit    = self.threshold
                cat_id   = 1
                accepted = score.item() <= limit
            else:
                cls = assigned_class[index]
                if cls is None:
                    continue
                limit    = class_thresholds[cls]
                cat_id   = cls_to_category_id[cls]  # mapeo 0-indexed → COCO id
                accepted = score.item() <= limit

            if accepted:
                results.append({
                    'image_id'   : imgs_ids[index],
                    'category_id': cat_id,
                    'score'      : imgs_scores[index],
                    'bbox'       : imgs_box_coords[index],
                })

        # ─────────────────────────────────────────────────────────────────
        # PASO 5: Escribir resultados en formato COCO
        # ─────────────────────────────────────────────────────────────────
        results_file = f"{dir_filtered_root}/{result_name}.json"
        if os.path.isfile(results_file):
            os.remove(results_file)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

            
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
    
    def _fit_single_class(self, all_features, all_labeled_features, all_context_features,
                          mahalanobis_method, beta, lambda_mahalanobis):
        """
        Ajusta la distribución Mahalanobis para el modo single-class.
        Aplica reducción de dimensionalidad y estima una sola gaussiana
        global sobre todos los features de foreground.

        Params:
            all_features         (Tensor): foreground + background (para fit SVD/PCA).
            all_labeled_features (Tensor): solo foreground (support set).
            all_context_features (Tensor): foreground + muestra de background
                                           (usado como contexto en regularización).
            mahalanobis_method   (str)   : 'regularization' o 'normal'.
            beta                 (float) : peso de la identidad en regularización.
            lambda_mahalanobis   (float) : peso del support set.

        Returns:
            None. Actualiza self.mean, self.inv_cov, self.threshold.
        """
        # Reducción de dimensionalidad: SVD/PCA se entrena sobre todas las
        # features (foreground + background) para capturar la varianza global
        if self.dim_red == DimensionalityReductionMethod.SVD:
            self.fit_svd(all_features.detach().numpy(), n_components=self.n_components)
            all_labeled_features = torch.Tensor(self.svd.transform(all_labeled_features.detach().numpy()))
            all_context_features = torch.Tensor(self.svd.transform(all_context_features.detach().numpy()))
        elif self.dim_red == DimensionalityReductionMethod.PCA:
            self.fit_pca(all_features.detach().numpy(), n_components=self.n_components)
            all_labeled_features = torch.Tensor(self.pca.transform(all_labeled_features.detach().numpy()))
            all_context_features = torch.Tensor(self.pca.transform(all_context_features.detach().numpy()))

        # Estimar media y covarianza inversa de la gaussiana
        if mahalanobis_method == "regularization":
            self.fit_regularization(
                all_labeled_features, beta=beta,
                context_features=all_context_features,
                lambda_mahalanobis=lambda_mahalanobis
            )
        else:
            self.fit_normal(all_labeled_features)

        # Calcular threshold usando IQR sobre las distancias del support set
        # threshold = Q3 + 1.5 * IQR (criterio de Tukey para outliers)
        distances = self.predict(all_labeled_features)
        Q1  = np.percentile(distances.numpy(), 25)
        Q3  = np.percentile(distances.numpy(), 75)
        IQR = Q3 - Q1
        self.threshold = Q3 + 1.5 * IQR