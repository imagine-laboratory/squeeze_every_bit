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


def create_dataset_ood(name, root, splits=('train'), 
    seed=None, labeled_samples=1, unlabeled_samples=5, validation_samples=1):
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
                root_ids = f"./seeds/{str(root).split('/')[-1]}/seed{seed}_{labeled_samples}_{unlabeled_samples}_{validation_samples}"
                ids_labeled = f"{root_ids}/labeled.txt"
                ids_test = f"{root_ids}/test.txt"
                ids_full_labeled = f"{root_ids}/full_labeled.txt"
                ids_unlabeled = f"{root_ids}/unlabeled.txt"
                ids_validation = f"{root_ids}/validation.txt"

                if not os.path.isfile(ids_labeled):
                    if seed is not None:
                        all_idx = random.Random(seed).sample(ids, k=total_samples)
                        # get labeled, test, unlabeled, and full labeled sets.
                        labeled_idx = random.Random(seed).sample(all_idx, k=labeled_samples)
                        # get validation samples
                        not_labeled_idx = [i for i in all_idx if i not in labeled_idx]
                        validation_idx = random.Random(seed).sample(not_labeled_idx, k=validation_samples)
                        test_idx = [i for i in all_idx if (i not in validation_idx + labeled_idx)]
                        full_labeled_idx = [i for i in ids if i not in test_idx]
                        unlabeled_idx = [i for i in full_labeled_idx if i not in labeled_idx]                  
                    else:
                        all_idx = random.sample(ids, k=total_samples)
                        labeled_idx = random.sample(all_idx, k=labeled_samples)
                        # get validation samples
                        not_labeled_idx = [i for i in all_idx if i not in labeled_idx]
                        validation_idx = random.sample(not_labeled_idx, k=validation_samples)
                        test_idx = [i for i in all_idx if i not in validation_idx + labeled_idx]
                        full_labeled_idx = [i for i in ids if i not in test_idx]
                        unlabeled_idx = [i for i in full_labeled_idx if i not in labeled_idx] 
                    # create folder
                    #ids_labeled.parent.mkdir(exist_ok=True, parents=True)
                    Path(root_ids).mkdir(parents=True, exist_ok=True)
                    
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