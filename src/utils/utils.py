import argparse
import psutil
import os
import json
import torchvision
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from data import create_dataset_ood
from data import transforms_toNumpy, create_loader

def add_bool_arg(parser, name, default=False, help=''):
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})

def seed_everything(seed: int = 42, rank: int = 0):
    """ Try to seed everything to reproduce results """
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed + rank)
    os.environ['PYTHONHASHSEED'] = str(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameters():
    # parameters
    parser = argparse.ArgumentParser(description='TIMM + new data aug')
    # Model 
    parser.add_argument('--root',type=str, default='.')
    parser.add_argument('--num-classes', type=int, default=1) 
    add_bool_arg(parser, 'load-pretrained', default=False) 
    # add_bool_arg(parser, 'use-sam-embeddings', default=False) 
    parser.add_argument('--use-sam-embeddings', type=int, default=0)
    parser.add_argument('--timm-model', type=str, default="")  
    parser.add_argument('--loss', type=str, default="mse")  
    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--val-freq', type=int, default=1)
    parser.add_argument('--ood-labeled-samples', type=int, default=1)
    parser.add_argument('--ood-unlabeled-samples', type=int, default=10)
    parser.add_argument('--ood-thresh', type=float, default=0.8)
    parser.add_argument('--ood-histogram-bins', type=int, default=15)

    # semi
    add_bool_arg(parser, 'use-semi-split', default=False) 
    parser.add_argument('--semi-percentage', type=float, default=10.)

    # training
    parser.add_argument('--epochs', type=int, default=1)

    # dataset
    parser.add_argument('--dataset', default='coco17', type=str, metavar='DATASET')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--batch-size-val', type=int, default=64)
    parser.add_argument('--reprob', type=float, default=0.)
    parser.add_argument('--aug-method', type=str, default="no_augmentation")
    parser.add_argument('--img-resolution', type=int, default=512)
    parser.add_argument('--new-sample-size', type=int, default=224)
    parser.add_argument('--batch-size-labeled', type=int, default=1)
    parser.add_argument('--batch-size-unlabeled', type=int, default=4)
    parser.add_argument('--method', default='None', type=str)

    # add_bool_arg(parser, 'store-val', default=False) 

    # general
    parser.add_argument('--numa', type=int, default=None)
    parser.add_argument('--output-folder', type=str, default=None)  
    parser.add_argument('--run-name', type=str, default=None)  
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sam-model', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    
    # Possible values for sam-proposal to generate object proposals, 'sam' 'semanticsam' 'mobilesam' 'fastsam'
    parser.add_argument('--sam-proposal', type=str, default="sam")

    # Dimensionality reduction parameters
    parser.add_argument('--dim-red', type=str, default="svd")
    parser.add_argument('--n-components', type=int, default=10)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--mahalanobis', type=str, default="normal")
    parser.add_argument('--batch-size-validation', type=int, default=4)
    parser.add_argument('--ood-validation-samples', type=int, default=10)
    parser.add_argument('--mahalanobis-lambda', type=float, default=-1.0)


    return parser.parse_args()

def get_cpu_list(val: int = None):
    """
    Accodring lscpu the NUMA nodes are:
    
    NUMA node0 CPU(s):     0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38
    NUMA node1 CPU(s):     1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39

    so, accordint to this distribution every gpu will be assign 10 cpus in the same NUMA node.
    cpu list goes from 0 to 39.
    """
    # numa 0
    if val == 0:
        return list(range(0,20,2))
    if val == 1:
        return list(range(20,40,2))

    # numa 1
    if val == 2:
        return list(range(1,20,2))
    if val == 3:
        return list(range(21,40,2))

# LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(numa: int = None):
    cpu_list = get_cpu_list(numa)
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])

def save_loader_to_json(unlabeled_loader, output_root, filename=None):
    """ Save new json file with gt.
    Params
    :unlabeled_loader (loader) -> to get all unlabeled imgs.
    :output_root (str) -> path to save the file.
    :filename (str) -> name of the output file.

    Return
    :None
    """
    if filename == None:
        raise Exception("filename must be a string")

    # get gt
    img_ids = []
    categories = []
    bboxes = []
    annotations = []
    id_count = 0
    # get all information from the batch 
    for batch in unlabeled_loader:
        (
            im, ca, bb
        ) = get_groundtruth(batch)
        img_ids += im
        categories += ca
        bboxes += bb

    # create the new annotations
    for index in range(0, len(img_ids)):
        ann_item = {
            'id': id_count,
            'image_id': img_ids[index],
            'category_id': categories[index],
            'bbox': bboxes[index],
            'area': int(bboxes[index][2] * bboxes[index][3]),
            'segmentation': [],
            'iscrowd': 0
        }
        id_count += 1
        annotations.append(ann_item)

    # load original json file and save it with just the unlabeled annotations
    gt_file_path = f"{str(unlabeled_loader.dataset.data_dir.parent)}/annotations/instances_train2017.json"
    with open(gt_file_path) as gt_file:
        json_contents = gt_file.read()
    gt_json = json.loads(json_contents)
    gt_json['annotations'] = annotations

    # keep only the imgs I need
    #--------------------------
    img_ids_from_anns = [i['image_id'] for i in gt_json['annotations']]
    img_ids_from_anns = list(set(img_ids_from_anns)) # unique elements
    images_res = []
    for img_node in gt_json['images']:
        if img_node['id'] in img_ids_from_anns:
            images_res.append(img_node)
    
    gt_json['images'] = images_res
    #--------------------------

    # write output
    gt_file = f"{output_root}/{filename}.json"
    if os.path.isfile(gt_file):
        os.remove(gt_file)
    json.dump(gt_json, open(gt_file, 'w'), indent=4)

def get_groundtruth(batch):
        """ From a batch get the gt
        Params
        :batch (<tensor, >) -> batch of images
        """
        img_ids = []
        categories = []
        bboxes = []

        # batch[1] has the metadata
        for idx in list(range(batch[1]['img_idx'].numel())):
            img_id = batch[1]['img_orig_id'][idx].item()
            bbox_indx = (batch[1]['bbox'][idx].sum(axis=1)>0).nonzero(as_tuple=True)[0].cpu()
            boxes = batch[1]['bbox'][idx][bbox_indx].cpu().tolist()
            classes = batch[1]['cls'][idx][bbox_indx].cpu().tolist()
            classes = [int(i) for i in classes]

            img_ids += [img_id] * len(classes) 
            categories += classes
            bboxes += boxes

        # translate bbox format to json xywh
        bboxes_xywh = []
        for bbox in bboxes:
            #FOR SOME REASON, THE COORDINATES COME:
            # [y1,x1,y2,x2] and need to be translated to: [x1,y1,x2,y2]
            xyxy = np.asarray(bbox)[[1,0,3,2]]
            xywh = torchvision.ops.box_convert(
                torch.tensor(xyxy), in_fmt='xyxy', out_fmt='xywh'
            ).tolist()
            xywh = [int(i) for i in xywh]
            bboxes_xywh.append(xywh)
        
        return img_ids, categories, bboxes_xywh

def create_datasets_and_loaders(args):
    """ Setup datasets, transforms, loaders, evaluator.
    Params
    :args -> Model specific configuration dict / struct

    Returns
    :loaders
    """
    datasets = create_dataset_ood(
        args.dataset, args.root, 
        seed=args.seed,
        labeled_samples=args.ood_labeled_samples,
        unlabeled_samples=args.ood_unlabeled_samples,
        validation_samples=args.ood_validation_samples,
    )
    dataset_label = datasets[0]
    dataset_test = datasets[1]
    dataset_unlabel = datasets[2]
    dataset_full_label = datasets[3]
    dataset_validation = datasets[4]

    # create data loaders
    trans_numpy = transforms_toNumpy()
    normalize_imgs = False
    loader_label = create_loader(
        dataset_label,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_labeled,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    loader_test = create_loader(
        dataset_test,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_unlabeled,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    loader_unlabel = create_loader(
        dataset_unlabel,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_unlabeled,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    loader_full_label = create_loader(
        dataset_full_label,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_unlabeled,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    loader_validation = create_loader(
        dataset_validation,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_validation,
        is_training=False,
        transform_fn = trans_numpy,
        normalize_img=normalize_imgs
    )
    return loader_label, loader_test, loader_unlabel, loader_full_label, loader_validation
    
