import os
import json
import numpy as np
import torch
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision
from data import transforms_toNumpy
from pycocotools.cocoeval import COCOeval

from data.transforms import Transform_To_Models

def save_inferences_simple(
    sam_model, unlabeled_loader, 
    filepath, use_sam_embeddings
    ):
    results = []
    trans_norm = Transform_To_Models(
        size=33, force_resize=False, keep_aspect_ratio=True
    )
    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    for (_,batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get foreground samples (from sam predictions)
            imgs_s, box_coords, scores = sam_model.get_unlabeled_samples( #Habria que ver si el modelo tiene forma de devolver clase y cuales son sus ID
                batch, idx, trans_norm, use_sam_embeddings
            )
            # accumulate SAM info (inferences)
            # no need to store the imgs, just the rest of the information
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    for idx_,_ in enumerate(imgs_ids):
        image_result = {
            'image_id': imgs_ids[idx_],
            'category_id': 1,
            'score': imgs_scores[idx_],
            'bbox': imgs_box_coords[idx_],
        }
        results.append(image_result)
        
    if len(results) > 0:
        # write output
        if os.path.exists(filepath):
            os.remove(filepath)
        json.dump(results, open(filepath, 'w'), indent=4)

def save_inferences_singleclass(
        fs_model, unlabeled_loader,
        sam_model, filepath,
        trans_norm,
        use_sam_embeddings, val=False
    ):
    fs_model.backbone.use_fc = False
    
    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    unlabeled_imgs = []

    # collect all inferences from SAM
    for (_, batch) in tqdm(enumerate(unlabeled_loader), total= len(unlabeled_loader)):

        # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
        # ITERATE: IMAGE
        for idx in list(range(batch[1]['img_idx'].numel())):
            # get foreground samples (from sam predictions)
            imgs_s, box_coords, scores = sam_model.get_unlabeled_samples(
                batch, idx, trans_norm, use_sam_embeddings
            )
            # accumulate SAM info (inferences)
            unlabeled_imgs += imgs_s
            imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores += scores

    # store std for 1 and for 2 and 3
    # for idx_1 in range(1,4):
    idx_1 = 2
    idx_float = float(idx_1)
    lower = fs_model.mean - (idx_float*fs_model.std)
    upper = fs_model.mean + (idx_float*fs_model.std)

    results = []
    for idx_,sample in enumerate(unlabeled_imgs):
        distance = fs_model(sample).cpu().item()

        # distance is inside the first std from the mean
        if  distance <= upper and distance >= lower:
            image_result = {
                'image_id': imgs_ids[idx_],
                'category_id': 1,
                'score': imgs_scores[idx_],
                'bbox': imgs_box_coords[idx_],
            }
            results.append(image_result)
        
    if len(results) > 0:
        # write output
        if val:
            f_ = f"{filepath}/bbox_results_val_std{idx_1}.json"
            if os.path.exists(f_):
                os.remove(f_)
            json.dump(results, open(f_, 'w'), indent=4)
        else:
            f_ = f"{filepath}/bbox_results_std{idx_1}.json"
            if os.path.exists(f_):
                os.remove(f_)
            json.dump(results, open(f_, 'w'), indent=4)

def save_inferences_twoclasses(
    fs_model, unlabeled_loader, sam_model, 
    filepath, trans_norm, use_sam_embeddings,
    val=False):
    results = []
    fs_model.backbone.use_fc = False

    imgs_ids = []
    imgs_box_coords = []
    imgs_scores = []
    unlabeled_imgs = []

    for (_, batch) in tqdm(enumerate(unlabeled_loader), total=len(unlabeled_loader)):
        for idx in list(range(batch[1]['img_idx'].numel())):
            imgs_s, box_coords, scores = sam_model.get_unlabeled_samples(
                batch, idx, trans_norm, use_sam_embeddings
            )
            unlabeled_imgs += imgs_s
            imgs_ids       += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
            imgs_box_coords += box_coords
            imgs_scores    += scores

    # Verificar si el modelo tiene prototype_labels (multiclase)
    # o si es la rama original de 2 clases
    is_multiclass = hasattr(fs_model, 'prototype_labels') and \
                    len(fs_model.prototype_labels) > 2

    # ── DEBUG ────────────────────────────────────────────────────────
    print(f"DEBUG save_inferences: is_multiclass={is_multiclass}, "
          f"total_samples={len(unlabeled_imgs)}")
    if is_multiclass:
        print(f"DEBUG save_inferences: prototype_labels={fs_model.prototype_labels.tolist()}")
    # ────────────────────────────────────────────────────────────────

    count = 0
    for idx_, sample in enumerate(unlabeled_imgs):
        res   = fs_model(sample)
        # índice del prototipo más cercano (posición en self.prototypes)
        proto_idx = torch.max(res.detach().data, 1)[1].item()

        SCORE_THRESHOLD = 0.5  # ajustable - ===== NO DEJAR ESTO ESTATICO ======
        if is_multiclass:
            # ── Rama multiclase ──────────────────────────────────────
            # Recuperar el label original (0-indexed) desde prototype_labels
            cls_0indexed = int(fs_model.prototype_labels[proto_idx].item())
            # Convertir a category_id COCO (1-indexed)
            cat_id = cls_0indexed + 1
            # En multiclase aceptamos todas las predicciones
            # (no hay clase "background" que descartar)
            accepted = imgs_scores[idx_] >= SCORE_THRESHOLD
        else:
            # ── Rama original de 2 clases: intacta ───────────────────
            if proto_idx == 0:  # not background
                cat_id   = 1
                accepted = True
            else:
                accepted = False

        if accepted:
            image_result = {
                'image_id':    imgs_ids[idx_],
                'category_id': cat_id,
                'score':       imgs_scores[idx_],
                'bbox':        imgs_box_coords[idx_],
            }
            results.append(image_result)
            count += 1

    # ── DEBUG distribución de clases predichas ───────────────────────
    if len(results) > 0:
        from collections import Counter
        cat_dist = Counter(r['category_id'] for r in results)
        print(f"DEBUG save_inferences: distribución category_id = {dict(cat_dist)}")
    print(f"DEBUG save_inferences: count={count}/{len(unlabeled_imgs)}")
    # ────────────────────────────────────────────────────────────────

    # Escribir siempre el archivo para evitar FileNotFoundError en eval_sam
    if val:
        f_ = f"{filepath}/bbox_results_val.json"
    else:
        f_ = f"{filepath}/bbox_results.json"

    try:
        if os.path.exists(f_):
            os.remove(f_)
    except OSError as e:
        print(f"ADVERTENCIA: no se pudo eliminar {f_}: {e}")

    if val:
        gt_path = f"{filepath}/validation.json"
    else:
        gt_path = f"{filepath}/test.json"

    if os.path.isfile(gt_path):
        with open(gt_path) as f:
            gt_data = json.load(f)
        valid_img_ids = set(img['id'] for img in gt_data['images'])
        results_filtered = [r for r in results if r['image_id'] in valid_img_ids]
        
        removed = len(results) - len(results_filtered)
        if removed > 0:
            print(f"DEBUG save_inferences: {removed} predicciones eliminadas "
                  f"por image_id no presentes en GT")
        results = results_filtered

    json.dump(results, open(f_, 'w'), indent=4)
    print(f"DEBUG save_inferences: archivo escrito con {len(results)} resultados → {f_}")

def calculate_precision(coco_eval, iou_treshold_index, img_id_size):
    imgs = coco_eval.evalImgs
    print("Images len: ", len(imgs))

    fp_total      = 0
    tp_total      = 0
    detection_ids = 0
    ground_ids    = 0

    # evalImgs tiene una entrada por (image_id, category_id)
    # En single-class: len(evalImgs) == img_id_size
    # En multiclase:   len(evalImgs) == img_id_size * n_classes
    # Por eso NO se usa count == img_id_size como límite —
    # se procesan TODAS las entradas válidas independientemente del número de clases

    for img in imgs:
        if img is None:
            continue

        detection_ignore = img["dtIgnore"][iou_treshold_index]
        mask = ~detection_ignore
        tp = (img["dtMatches"][iou_treshold_index][mask] > 0).sum()
        fp = (img["dtMatches"][iou_treshold_index][mask] == 0).sum()

        gtIds_len = len(img['gtIds'])
        dtIds_len = len(img['dtIds'])

        fp_total      += fp
        tp_total      += tp
        detection_ids += dtIds_len
        ground_ids    += gtIds_len

    # Proteger contra división por cero cuando no hay GT
    p_dt_gt = float(detection_ids / ground_ids) if ground_ids > 0 else 0.0
    p_t_gt  = float(tp_total / ground_ids)      if ground_ids > 0 else 0.0

    return {
        "amount_imgs":        int(img_id_size),
        "detection_ids":      int(detection_ids),
        "ground_truth_ids":   int(ground_ids),
        "tp":                 int(tp_total),
        "fp":                 int(fp_total),
        "iou_treshold_index": int(iou_treshold_index),
        "p_dt_gt":            p_dt_gt,
        "p_t_gt":             p_t_gt
    }

def eval_sam(coco_gt, image_ids, pred_json_path, output_root, method="xyz", number=None, val=False):
    if not os.path.isfile(pred_json_path):
        print(f"ERROR eval_sam: {pred_json_path} no existe.")
        return
    with open(pred_json_path) as f:
        pred_data = json.load(f)
    if len(pred_data) == 0:
        print(f"ADVERTENCIA eval_sam: archivo vacío, se omite evaluación.")
        return
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    pred_ids = set([x["image_id"] for x in preds])
    gt_ids   = set(coco_gt.getImgIds())

    print("Pred ids sample:", list(pred_ids)[:10])
    print("GT ids sample:", list(gt_ids)[:10])

    print("Extra pred ids:")
    print(pred_ids - gt_ids)
    coco_pred = coco_gt.loadRes(pred_json_path)
    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_count = calculate_precision(coco_eval, 0, len(image_ids)) #En esta función hay que ver como se toma en cuenta la clase
    print("stats count: ", stats_count)
    val_str = ""
    if val:
        val_str = "_val"
    file_name_stats = f"{output_root}/stats{val_str}_{method}.json"
    with open(file_name_stats, 'w') as file:
        file.write(json.dumps(stats_count))
    
    # write results into a file
    if number is None:
        if val:
            file_name = f"{output_root}/mAP_val_{method}.txt"
        else:
            file_name = f"{output_root}/mAP_{method}.txt"
    else:
        if val:
            file_name = f"{output_root}/mAP_val_{method}_std{number}.txt"
        else:
            file_name = f"{output_root}/mAP_{method}_std{number}.txt"
    with open(file_name, 'w') as file:
        for i in coco_eval.stats:
            file.write(f"{str(i)}\n")
    