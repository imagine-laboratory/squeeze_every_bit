""" Dataset factory
Copyright 2020 Ross Wightman
"""
import os
import json
import numpy as np
import pickle
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
import random
from .dataset_config import *
from .parsers import *
from .dataset import DetectionDatset
from .parsers import create_parser

from sklearn.model_selection import train_test_split


def create_dataset(name, root, splits=('train', 'val'), use_semi_split=False, seed=42, semi_percentage=1.0):
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    dataset_cls = DetectionDatset
    datasets = OrderedDict()
    if name.startswith('coco'):
        if 'coco2014' in name:
            dataset_cfg = Coco2014Cfg()
        else:
            dataset_cfg = Coco2017Cfg()

        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']

            if use_semi_split and s.startswith('train'):
                #--------------------------------------------
                # open file if semi to split the labeled dict
                with open(ann_file, 'r') as f:
                    new_labeled = json.load(f)
                with open(ann_file, 'r') as f2:
                    new_unlabeled = json.load(f2)
                # get all image ids 
                ids = [item['id'] for item in new_labeled['images']]

                # split the ids
                y_dumpy = np.zeros(len(ids))
                labeled_idx, _, _, _ = train_test_split(
                ids, y_dumpy, 
                train_size = semi_percentage / 100.0, 
                shuffle = True, 
                random_state = seed
                )
                # keep just the necessary ids
                new_labeled['images'] = [i for i in new_labeled['images'] if i['id'] in labeled_idx]
                new_labeled['annotations'] = [i for i in new_labeled['annotations'] if i['image_id'] in labeled_idx]
                
                new_unlabeled['images'] = [i for i in new_unlabeled['images'] if i['id'] not in labeled_idx]
                new_unlabeled['annotations'] = [i for i in new_unlabeled['annotations'] if i['image_id'] not in labeled_idx]
                #--------------------------------------------

                # labeled dataset
                parser_cfg_labeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_labeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_labeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_labeled),
                )

                # unlabeled dataset
                parser_cfg_unlabeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_unlabeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_unlabeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_unlabeled),
                )
                
            else:
                parser_cfg = CocoParserCfg(
                    ann_filename=ann_file,
                    has_labels=split_cfg['has_labels']
                )
                datasets[s] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
                )
    elif name.startswith('voc'):
        if 'voc0712' in name:
            dataset_cfg = Voc0712Cfg()
        elif 'voc2007' in name:
            dataset_cfg = Voc2007Cfg()
        else:
            dataset_cfg = Voc2012Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            if isinstance(split_cfg['split_filename'], (tuple, list)):
                assert len(split_cfg['split_filename']) == len(split_cfg['ann_filename'])
                parser = None
                for sf, af, id in zip(
                        split_cfg['split_filename'], split_cfg['ann_filename'], split_cfg['img_dir']):
                    parser_cfg = VocParserCfg(
                        split_filename=root / sf,
                        ann_filename=os.path.join(root, af),
                        img_filename=os.path.join(id, dataset_cfg.img_filename))
                    if parser is None:
                        parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                    else:
                        other_parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                        parser.merge(other=other_parser)
            else:
                parser_cfg = VocParserCfg(
                    split_filename=root / split_cfg['split_filename'],
                    ann_filename=os.path.join(root, split_cfg['ann_filename']),
                    img_filename=os.path.join(split_cfg['img_dir'], dataset_cfg.img_filename),
                )
                parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
            datasets[s] = dataset_cls(data_dir=root, parser=parser)
    elif name.startswith('openimages'):
        if 'challenge2019' in name:
            dataset_cfg = OpenImagesObjChallenge2019Cfg()
        else:
            dataset_cfg = OpenImagesObjV5Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            parser_cfg = OpenImagesParserCfg(
                categories_filename=root / dataset_cfg.categories_map,
                img_info_filename=root / split_cfg['img_info'],
                bbox_filename=root / split_cfg['ann_bbox'],
                img_label_filename=root / split_cfg['ann_img_label'],
                img_filename=dataset_cfg.img_filename,
                prefix_levels=split_cfg['prefix_levels'],
                has_labels=split_cfg['has_labels'],
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg)
            )
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]

def _sample_balanced_subset(annotations, candidate_ids, n_samples, rng, label="subset"):
    """
    Selecciona n_samples imágenes de candidate_ids con distribución
    balanceada por clase. Mismo algoritmo que sample_balanced_labeled
    pero opera sobre un pool arbitrario de candidatos.

    Params:
        annotations   (list[dict]): anotaciones COCO con 'image_id' y 'category_id'.
        candidate_ids (list[int]) : pool de image_id disponibles para este subset.
        n_samples     (int)       : número de imágenes a seleccionar.
        rng           (Random)    : instancia de random con seed fijo.
        label         (str)       : nombre del subset para prints de debug.

    Returns:
        list[int]: image_ids seleccionados.
    """
    candidate_set = set(candidate_ids)

    # Construir mapa clase → imágenes disponibles dentro del pool
    class_to_imgs = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in candidate_set:
            continue
        cat_id = ann['category_id']
        if cat_id not in class_to_imgs:
            class_to_imgs[cat_id] = set()
        class_to_imgs[cat_id].add(img_id)

    class_to_imgs  = {cls: list(imgs) for cls, imgs in class_to_imgs.items()}
    unique_classes = sorted(class_to_imgs.keys())
    n_classes      = len(unique_classes)
    imgs_per_class = max(1, n_samples // n_classes)

    print(f"  [balanced_{label}] {n_classes} clases, objetivo: {imgs_per_class} imgs/clase")

    selected = set()

    # Fase 1: estratificado por clase
    for cls in unique_classes:
        available = [i for i in class_to_imgs[cls] if i not in selected]
        k = min(imgs_per_class, len(available))
        if k > 0:
            chosen = rng.sample(available, k=k)
            selected.update(chosen)
            print(f"  [balanced_{label}] Clase {cls}: {k} imgs")
        else:
            print(f"  [balanced_{label}] ADVERTENCIA clase {cls}: sin imgs disponibles")

    # Fase 2: completar si faltan cupos
    remaining    = [i for i in candidate_ids if i not in selected]
    extra_needed = n_samples - len(selected)
    if extra_needed > 0 and len(remaining) > 0:
        extra = rng.sample(remaining, k=min(extra_needed, len(remaining)))
        selected.update(extra)
        print(f"  [balanced_{label}] +{len(extra)} imgs extra")

    result = list(selected)
    print(f"  [balanced_{label}] Total: {len(result)} imgs")
    return result

# ── Rama multiclase: sampling balanceado por clase ──────────
def sample_balanced_labeled(annotations, all_image_ids, n_labeled, rng):
    # Convertir a set para búsquedas O(1)
    all_image_ids_set = set(all_image_ids)

    # Construir class_to_imgs directamente en un solo pase
    # filtrando solo imágenes relevantes desde el inicio
    class_to_imgs = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in all_image_ids_set:  # O(1) ahora
            continue
        cat_id = ann['category_id']
        if cat_id not in class_to_imgs:
            class_to_imgs[cat_id] = set()  # set para evitar duplicados
        class_to_imgs[cat_id].add(img_id)

    # Convertir sets a listas para rng.sample
    class_to_imgs = {cls: list(imgs) for cls, imgs in class_to_imgs.items()}

    unique_classes = sorted(class_to_imgs.keys())
    n_classes = len(unique_classes)
    imgs_per_class = max(1, n_labeled // n_classes)

    print(f"DEBUG balanced: {n_classes} clases, {imgs_per_class} imgs/clase")
    print(f"DEBUG balanced: clases disponibles = {unique_classes}")

    selected = set()
    for cls in unique_classes:
        available = [i for i in class_to_imgs[cls] if i not in selected]
        k = min(imgs_per_class, len(available))
        if k > 0:
            chosen = rng.sample(available, k=k)
            selected.update(chosen)
            print(f"DEBUG balanced: clase {cls} → {k} imgs seleccionadas")
        else:
            print(f"ADVERTENCIA balanced: clase {cls} sin imgs disponibles")

    remaining = [i for i in all_image_ids if i not in selected]
    extra_needed = n_labeled - len(selected)
    if extra_needed > 0 and len(remaining) > 0:
        extra = rng.sample(remaining, k=min(extra_needed, len(remaining)))
        selected.update(extra)
        print(f"DEBUG balanced: +{len(extra)} imgs extra para completar {n_labeled}")

    result = list(selected)
    print(f"DEBUG balanced: total = {len(result)} imgs")
    return result

def create_dataset_ood(name, root, splits=('train'), 
    seed=None, labeled_samples=1, unlabeled_samples=5, validation_samples=1,
    balanced_classes=False):
    """
    This method splits the ds, format appropriate for the loader.

    Params
    :name (str)
    :root (str)
    :splits (List<str>)
    :seed (int)
    :labeled_samples (int)
    :unlabeled_samples (int)
    :validation_samples (int)

    Return
    :datasets (type for dataloaders)
    """
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    dataset_cls = DetectionDatset
    datasets = OrderedDict()

    if name.startswith('coco'):
        if 'coco2014' in name:
            dataset_cfg = Coco2014Cfg()
        else:
            dataset_cfg = Coco2017Cfg()

        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']

            if s.startswith('train'):
                #--------------------------------------------
                # open file if semi to split the labeled dict
                with open(ann_file, 'r') as f:
                    new_labeled = json.load(f)
                with open(ann_file, 'r') as f2:
                    new_test = json.load(f2)
                with open(ann_file, 'r') as f3:
                    new_full_labeled = json.load(f3)
                with open(ann_file, 'r') as f4:
                    new_unlabeled = json.load(f4)
                with open(ann_file, 'r') as f5:
                    new_validation = json.load(f5)

                # get all image ids 
                ids = [item['id'] for item in new_labeled['images']]
                total_samples = labeled_samples + unlabeled_samples + validation_samples
                assert total_samples <= len(ids), "size mismatch"

                # get ids
                root_ids = root_ids = Path("seeds") / root.name / f"seed{seed}_{labeled_samples}_{unlabeled_samples}_{validation_samples}"
                ids_labeled = f"{root_ids}/labeled.txt"
                ids_test = f"{root_ids}/test.txt"
                ids_full_labeled = f"{root_ids}/full_labeled.txt"
                ids_unlabeled = f"{root_ids}/unlabeled.txt"
                ids_validation = f"{root_ids}/validation.txt"

                if not os.path.isfile(ids_labeled):
                    rng = random.Random(seed) if seed is not None else random

                    if balanced_classes:
                        labeled_idx = sample_balanced_labeled(
                            new_labeled['annotations'], ids, labeled_samples, rng
                        )
                        # ───────────────────────────────────────────────────────────

                    else:
                        # ── Rama original single-class: intacta ─────────────────────
                        all_idx = rng.sample(ids, k=total_samples)
                        labeled_idx = rng.sample(all_idx, k=labeled_samples)
                        # ───────────────────────────────────────────────────────────

                    # El resto del split es igual para ambas ramas
                    not_labeled_idx = [i for i in ids if i not in labeled_idx]

                    # Balancear test
                    test_idx = _sample_balanced_subset(
                        annotations  = new_labeled['annotations'],
                        candidate_ids= not_labeled_idx,
                        n_samples    = unlabeled_samples,
                        rng          = rng,
                        label        = "test"
                    )

                    # Balancear validación (del pool restante después de test)
                    remaining_after_test = [i for i in not_labeled_idx if i not in test_idx]
                    validation_idx = _sample_balanced_subset(
                        annotations  = new_labeled['annotations'],
                        candidate_ids= remaining_after_test,
                        n_samples    = validation_samples,
                        rng          = rng,
                        label        = "validation"
                    )
                    #test_idx         = [i for i in remaining_pool if i not in validation_idx]
                    full_labeled_idx = [i for i in ids if i not in test_idx]
                    unlabeled_idx    = [i for i in full_labeled_idx if i not in labeled_idx]

                    # ── DEBUG post-split ─────────────────────────────────────────────
                    labeled_anns = [a for a in new_labeled['annotations'] if a['image_id'] in labeled_idx]
                    from collections import Counter
                    class_dist = Counter(a['category_id'] for a in labeled_anns)
                    print(f"DEBUG post-split: {len(labeled_idx)} imgs labeled, "
                          f"distribución clases = {dict(sorted(class_dist.items()))}")
                    print(f"DEBUG post-split: test={len(test_idx)}, "
                          f"validation={len(validation_idx)}, unlabeled={len(unlabeled_idx)}")
                    # ────────────────────────────────────────────────────────────────
                    # create folder
                    #ids_labeled.parent.mkdir(exist_ok=True, parents=True)
                    root_ids.mkdir(parents=True, exist_ok=True)

                    
                    # save files    
                    with open(ids_labeled, "wb") as fp:
                        pickle.dump(labeled_idx, fp)
                    with open(ids_test, "wb") as fp:
                        pickle.dump(test_idx, fp)
                    with open(ids_full_labeled, "wb") as fp:
                        pickle.dump(full_labeled_idx, fp)
                    with open(ids_unlabeled, "wb") as fp:
                        pickle.dump(unlabeled_idx, fp)
                    with open(ids_validation, "wb") as fp:
                        pickle.dump(validation_idx, fp)
                else:
                    #load pickle files
                    with open(ids_labeled, "rb") as fp:
                        labeled_idx = pickle.load(fp)
                    with open(ids_test, "rb") as fp:
                        test_idx = pickle.load(fp)
                    with open(ids_full_labeled, "rb") as fp:
                        full_labeled_idx = pickle.load(fp)
                    with open(ids_unlabeled, "rb") as fp:
                        unlabeled_idx = pickle.load(fp)
                    with open(ids_validation, "rb") as fp:
                        validation_idx = pickle.load(fp)
                # Añadir después del bloque if/else de ids_labeled (tanto en la rama nueva como al cargar del cache)
                print(f"DEBUG splits: labeled={len(labeled_idx)}, test={len(test_idx)}, "
                    f"validation={len(validation_idx)}, unlabeled={len(unlabeled_idx)}")
                print(f"DEBUG splits: labeled_idx muestra = {sorted(labeled_idx)[:5]}")
                print(f"DEBUG splits: test_idx muestra = {sorted(test_idx)[:5]}")

                # Verificar que no hay solapamiento
                overlap_labeled_test = set(labeled_idx) & set(test_idx)
                overlap_test_val     = set(test_idx) & set(validation_idx)
                print(f"DEBUG splits: solapamiento labeled∩test = {overlap_labeled_test}")
                print(f"DEBUG splits: solapamiento test∩validation = {overlap_test_val}")

                # keep just the necessary ids
                new_labeled['images'] = [i for i in new_labeled['images'] if i['id'] in labeled_idx]
                new_labeled['annotations'] = [i for i in new_labeled['annotations'] if i['image_id'] in labeled_idx]
                
                new_test['images'] = [i for i in new_test['images'] if i['id'] in test_idx]
                new_test['annotations'] = [i for i in new_test['annotations'] if i['image_id'] in test_idx]

                new_full_labeled['annotations'] = [i for i in new_full_labeled['annotations'] if i['image_id'] in full_labeled_idx]
                new_full_labeled['images'] = [i for i in new_full_labeled['images'] if i['id'] in full_labeled_idx]

                new_unlabeled['annotations'] = [i for i in new_unlabeled['annotations'] if i['image_id'] in unlabeled_idx]
                new_unlabeled['images'] = [i for i in new_unlabeled['images'] if i['id'] in unlabeled_idx]

                new_validation['annotations'] = [i for i in new_validation['annotations'] if i['image_id'] in validation_idx]
                new_validation['images'] = [i for i in new_validation['images'] if i['id'] in validation_idx]
                #--------------------------------------------

                # labeled dataset
                parser_cfg_labeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_labeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_labeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_labeled),
                )

                # test dataset
                parser_cfg_test = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_test,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_test'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_test),
                )

                # unlabeled dataset
                parser_cfg_unlabeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_unlabeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_unlabeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_unlabeled),
                )

                # full_labeled dataset
                parser_cfg_full_labeled = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_full_labeled,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_full_labeled'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_full_labeled),
                )

                # validation dataset
                parser_cfg_validation = CocoParserCfg(
                    ann_filename = "",
                    json_dict=new_validation,
                    has_labels=split_cfg['has_labels']
                )
                datasets[f'{s}_validation'] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg_validation),
                )

                validation_dataset = datasets[f'{s}_labeled']
                # Información básica
                print(f"Nombre del dataset: {s}_labeled")
                print(f"Tamaño del dataset: {len(validation_dataset)} imágenes")
                print(f"Tipo de dataset: {type(validation_dataset)}")
                print(f"Directorio de imágenes: {validation_dataset.data_dir}")
                print(f"Parser usado: {type(validation_dataset.parser)}")

                
            else:
                parser_cfg = CocoParserCfg(
                    ann_filename=ann_file,
                    has_labels=split_cfg['has_labels']
                )
                datasets[s] = dataset_cls(
                    data_dir=root / Path(split_cfg['img_dir']),
                    parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
                )
                
    elif name.startswith('voc'):
        if 'voc0712' in name:
            dataset_cfg = Voc0712Cfg()
        elif 'voc2007' in name:
            dataset_cfg = Voc2007Cfg()
        else:
            dataset_cfg = Voc2012Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            if isinstance(split_cfg['split_filename'], (tuple, list)):
                assert len(split_cfg['split_filename']) == len(split_cfg['ann_filename'])
                parser = None
                for sf, af, id in zip(
                        split_cfg['split_filename'], split_cfg['ann_filename'], split_cfg['img_dir']):
                    parser_cfg = VocParserCfg(
                        split_filename=root / sf,
                        ann_filename=os.path.join(root, af),
                        img_filename=os.path.join(id, dataset_cfg.img_filename))
                    if parser is None:
                        parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                    else:
                        other_parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
                        parser.merge(other=other_parser)
            else:
                parser_cfg = VocParserCfg(
                    split_filename=root / split_cfg['split_filename'],
                    ann_filename=os.path.join(root, split_cfg['ann_filename']),
                    img_filename=os.path.join(split_cfg['img_dir'], dataset_cfg.img_filename),
                )
                parser = create_parser(dataset_cfg.parser, cfg=parser_cfg)
            datasets[s] = dataset_cls(data_dir=root, parser=parser)
    elif name.startswith('openimages'):
        if 'challenge2019' in name:
            dataset_cfg = OpenImagesObjChallenge2019Cfg()
        else:
            dataset_cfg = OpenImagesObjV5Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            parser_cfg = OpenImagesParserCfg(
                categories_filename=root / dataset_cfg.categories_map,
                img_info_filename=root / split_cfg['img_info'],
                bbox_filename=root / split_cfg['ann_bbox'],
                img_label_filename=root / split_cfg['ann_img_label'],
                img_filename=dataset_cfg.img_filename,
                prefix_levels=split_cfg['prefix_levels'],
                has_labels=split_cfg['has_labels'],
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg)
            )
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets #if len(datasets) > 1 else datasets[0]