"""
visualize_detections.py
=======================
Utilidades para graficar:
  1. Ground Truth (GT) desde un DataLoader.
  2. Propuestas aceptadas y rechazadas desde los JSONs de resultados
     generados por evaluate() (bbox_results.json / bbox_results_val.json).

Estructura de salida:
    output_root/
        visualizations/
            gt/
                {image_id}.jpg
            proposals/
                accepted/
                    {image_id}.jpg
                rejected/
                    {image_id}.jpg

Uso típico (en mahalanobis_filter, después de run_filter):
    from utils.visualize_detections import visualize_gt, visualize_proposals

    img_dir = Path(args.root) / "train2017"

    visualize_gt(
        dataloader       = test_loader,
        img_dir          = img_dir,
        output_root      = output_root,
    )
    visualize_proposals(
        results_json     = f"{output_root}/bbox_results.json",
        gt_json          = f"{output_root}/test.json",
        img_dir          = img_dir,
        output_root      = output_root,
    )
"""

import os
import json
import numpy as np
import cv2
import torch
import torchvision
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de visualización
# ─────────────────────────────────────────────────────────────────────────────

# Color GT: verde sólido
COLOR_GT       = (0, 200, 0)

# Colores por clase para propuestas (BGR, hasta 8 clases)
# Se ciclan si hay más clases que colores definidos
CLASS_COLORS = [
    (255,  80,  80),   # 0 → multiclass   (azul claro)
    (255, 160,   0),   # 1 → apple        (naranja)
    ( 80, 200,  80),   # 2 → avocado      (verde)
    (  0, 140, 255),   # 3 → capsicum     (naranja oscuro)
    (200,   0, 200),   # 4 → mango        (magenta)
    (  0, 200, 200),   # 5 → orange       (cian)
    ( 80,  80, 255),   # 6 → pineapple    (rojo)
    (120, 120,   0),   # 7 → strawberry   (oliva)
]

COLOR_ACCEPTED = (0, 230, 0)    # verde  → propuesta aceptada
COLOR_REJECTED = (0, 0, 230)    # rojo   → propuesta rechazada

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.5
FONT_THICKNESS  = 1
BOX_THICKNESS   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Mapeo de categorías
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_NAMES = {
    0: "multiclass",
    1: "apple",
    2: "avocado",
    3: "capsicum",
    4: "mango",
    5: "orange",
    6: "pineapple",
    7: "strawberry",
}

def get_category_name(category_id: int) -> str:
    """
    Retorna el nombre de la categoría dado su ID.
    Si no está en el mapeo interno, retorna 'cls_{id}'.

    Params:
        category_id (int): ID COCO de la categoría.

    Returns:
        str: nombre legible de la categoría.
    """
    return CATEGORY_NAMES.get(int(category_id), f"cls_{category_id}")


def get_class_color(category_id: int):
    """
    Retorna el color BGR asociado a una categoría.
    Cicla si category_id supera la lista definida.

    Params:
        category_id (int): ID COCO de la categoría.

    Returns:
        tuple: color en formato BGR.
    """
    return CLASS_COLORS[int(category_id) % len(CLASS_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs(output_root: str):
    """Crea la estructura de carpetas de salida si no existe."""
    paths = {
        "gt"       : os.path.join(output_root, "visualizations", "gt"),
        "accepted" : os.path.join(output_root, "visualizations", "proposals", "accepted"),
        "rejected" : os.path.join(output_root, "visualizations", "proposals", "rejected"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _load_image(img_dir: Path, image_id: int, gt_json_images: list) -> np.ndarray:
    """
    Carga una imagen desde disco dado su image_id COCO.

    Busca el filename en la lista gt_json_images (campo 'file_name').
    Si no lo encuentra, intenta el nombre estándar COCO ({image_id:012d}.jpg).

    Params:
        img_dir        (Path): carpeta con las imágenes (ej. train2017/).
        image_id       (int) : image_id COCO.
        gt_json_images (list): lista de dicts {'id', 'file_name', ...} del JSON.

    Returns:
        np.ndarray: imagen en BGR, o None si no se pudo cargar.
    """
    # Buscar file_name en el JSON
    filename = None
    for img_node in gt_json_images:
        if img_node['id'] == image_id:
            filename = img_node['file_name']
            break

    # Fallback: nombre estándar COCO
    if filename is None:
        filename = f"{image_id:012d}.jpg"

    img_path = img_dir / filename
    if not img_path.exists():
        print(f"[visualize] ADVERTENCIA: imagen no encontrada → {img_path}")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[visualize] ADVERTENCIA: no se pudo leer → {img_path}")
    return img


def _draw_box(img: np.ndarray, bbox_xywh: list, color: tuple, label: str):
    """
    Dibuja un bounding box y su etiqueta sobre la imagen (in-place).

    Params:
        img       (np.ndarray): imagen BGR.
        bbox_xywh (list)      : bbox en formato [x, y, w, h].
        color     (tuple)     : color BGR del box.
        label     (str)       : texto a mostrar sobre el box.
    """
    x, y, w, h = [int(v) for v in bbox_xywh]
    x2, y2 = x + w, y + h

    cv2.rectangle(img, (x, y), (x2, y2), color, BOX_THICKNESS)

    # Fondo del texto para legibilidad
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 2, y), color, -1)
    cv2.putText(img, label, (x + 1, y - 2), FONT, FONT_SCALE,
                (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def _save_image(img: np.ndarray, out_path: str):
    """Guarda la imagen en disco."""
    cv2.imwrite(out_path, img)


# ─────────────────────────────────────────────────────────────────────────────
# Función pública 1: visualizar Ground Truth desde DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def visualize_gt(
    dataloader,
    img_dir: str,
    output_root: str,
    gt_json_path: str = None,
):
    """
    Recorre el dataloader, extrae las anotaciones GT de cada imagen
    y guarda una imagen por cada una con sus bboxes dibujados.

    Solo procesa imágenes que tengan al menos una anotación en el batch
    (imágenes sin anotaciones son ignoradas).

    Params:
        dataloader   (DataLoader): loader con imágenes etiquetadas.
                                   Debe exponer batch[1]['img_orig_id'],
                                   batch[1]['bbox'] y batch[1]['cls'].
        img_dir      (str)       : carpeta con las imágenes originales
                                   (ej. "dataset/train2017").
        output_root  (str)       : carpeta raíz de salida.
                                   Las imágenes se guardan en
                                   output_root/visualizations/gt/.
        gt_json_path (str|None)  : path al JSON COCO (para obtener file_name
                                   de cada imagen). Si es None, se usa el
                                   nombre estándar COCO ({id:012d}.jpg).

    Returns:
        None. Escribe imágenes en disco.
    """
    img_dir   = Path(img_dir)
    dirs      = _ensure_dirs(output_root)
    out_dir   = dirs["gt"]

    # Cargar lista de imágenes del JSON si se provee
    gt_images = []
    if gt_json_path and os.path.isfile(gt_json_path):
        with open(gt_json_path) as f:
            gt_images = json.load(f).get('images', [])

    print(f"[visualize_gt] Procesando {len(dataloader)} batches → {out_dir}")

    for batch in tqdm(dataloader, desc="visualize_gt"):
        # batch[0] → tensor de imágenes (N, C, H, W) o similar
        # batch[1] → dict de metadatos

        for idx in range(batch[1]['img_idx'].numel()):
            image_id = batch[1]['img_orig_id'][idx].item()

            # Extraer bboxes válidos (filas con suma > 0)
            # batch[1]['bbox'][idx] shape: [max_anns, 4] en formato [y1,x1,y2,x2]
            bbox_raw  = batch[1]['bbox'][idx]
            cls_raw   = batch[1]['cls'][idx]

            valid_mask = (bbox_raw.sum(axis=1) > 0).nonzero(as_tuple=True)[0].cpu()

            # Si la imagen no tiene anotaciones, saltar
            if len(valid_mask) == 0:
                continue

            boxes   = bbox_raw[valid_mask].cpu().tolist()   # [y1,x1,y2,x2]
            classes = cls_raw[valid_mask].cpu().tolist()
            classes = [int(c) for c in classes]

            # Cargar imagen desde disco
            img = _load_image(img_dir, image_id, gt_images)
            if img is None:
                continue

            # Convertir bboxes [y1,x1,y2,x2] → [x,y,w,h]
            for bbox_yx, cat_id in zip(boxes, classes):
                xyxy     = np.array(bbox_yx)[[1, 0, 3, 2]]  # [x1,y1,x2,y2]
                xywh     = torchvision.ops.box_convert(
                    torch.tensor(xyxy, dtype=torch.float32),
                    in_fmt='xyxy', out_fmt='xywh'
                ).tolist()
                xywh     = [int(v) for v in xywh]
                color    = get_class_color(cat_id)
                label    = f"{cat_id}:{get_category_name(cat_id)}"
                _draw_box(img, xywh, color, label)

            out_path = os.path.join(out_dir, f"{image_id}.jpg")
            _save_image(img, out_path)

    print(f"[visualize_gt] Listo. Imágenes guardadas en: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Función pública 2: visualizar propuestas aceptadas y rechazadas
# ─────────────────────────────────────────────────────────────────────────────

def visualize_proposals(
    results_json: str,
    gt_json: str,
    img_dir: str,
    output_root: str,
):
    """
    Lee el archivo de resultados generado por evaluate() y guarda una imagen
    por cada imagen del GT, mostrando:
      - Bboxes ACEPTADOS (pasaron el threshold) en verde.
      - Bboxes RECHAZADOS (no pasaron el threshold) en rojo.

    El archivo results_json contiene solo las propuestas aceptadas.
    Para las rechazadas se usa el universo completo de propuestas SAM, que
    NO está disponible directamente. Por eso este método toma un enfoque
    alternativo: compara las propuestas aceptadas contra el GT para mostrar
    cuáles coinciden (aceptadas) y cuáles anotaciones GT quedaron sin detectar
    (no detectadas).

    Concretamente:
      - Verde  (accepted)    : bbox del results_json (propuesta aceptada por Mahalanobis).
      - Rojo   (GT no detect): bbox del GT que NO fue cubierto por ninguna propuesta aceptada.

    Esto es más informativo que mostrar rechazadas crudas, porque permite ver
    directamente los falsos negativos del pipeline.

    Params:
        results_json (str): path a bbox_results.json o bbox_results_val.json.
        gt_json      (str): path al GT guardado (test.json o validation.json).
        img_dir      (str): carpeta con las imágenes originales.
        output_root  (str): carpeta raíz de salida.

    Returns:
        None. Escribe imágenes en:
            output_root/visualizations/proposals/accepted/
            output_root/visualizations/proposals/rejected/
    """
    img_dir = Path(img_dir)
    dirs    = _ensure_dirs(output_root)

    # ── Cargar resultados y GT ─────────────────────────────────────────────
    if not os.path.isfile(results_json):
        print(f"[visualize_proposals] ERROR: no existe {results_json}")
        return
    if not os.path.isfile(gt_json):
        print(f"[visualize_proposals] ERROR: no existe {gt_json}")
        return

    with open(results_json) as f:
        predictions = json.load(f)   # lista de {image_id, category_id, score, bbox}

    with open(gt_json) as f:
        gt_data = json.load(f)

    gt_images      = gt_data.get('images', [])
    gt_annotations = gt_data.get('annotations', [])

    # Agrupar predicciones por image_id
    preds_by_image = defaultdict(list)
    for pred in predictions:
        preds_by_image[pred['image_id']].append(pred)

    # Agrupar GT por image_id
    gt_by_image = defaultdict(list)
    for ann in gt_annotations:
        gt_by_image[ann['image_id']].append(ann)

    # Universo de imágenes a graficar: todas las que están en el GT
    all_image_ids = [img['id'] for img in gt_images]

    print(f"[visualize_proposals] {len(predictions)} predicciones sobre "
          f"{len(all_image_ids)} imágenes → {dirs['accepted']} / {dirs['rejected']}")

    for image_id in tqdm(all_image_ids, desc="visualize_proposals"):

        img = _load_image(img_dir, image_id, gt_images)
        if img is None:
            continue

        # ── Imagen para propuestas aceptadas ──────────────────────────────
        img_accepted = img.copy()
        accepted_preds = preds_by_image.get(image_id, [])

        for pred in accepted_preds:
            bbox    = pred['bbox']          # [x, y, w, h]
            cat_id  = pred['category_id']
            label   = f"ACC {cat_id}:{get_category_name(cat_id)}"
            _draw_box(img_accepted, bbox, COLOR_ACCEPTED, label)

        out_accepted = os.path.join(dirs['accepted'], f"{image_id}.jpg")
        _save_image(img_accepted, out_accepted)

        # ── Imagen para GT no detectado (falsos negativos) ────────────────
        # Una anotación GT se considera "detectada" si hay al menos una
        # predicción aceptada con IoU >= 0.5 sobre esa anotación.
        img_rejected = img.copy()
        gt_anns      = gt_by_image.get(image_id, [])

        for ann in gt_anns:
            gt_bbox  = ann['bbox']    # [x, y, w, h]
            cat_id   = ann['category_id']
            detected = _is_detected(gt_bbox, accepted_preds, iou_threshold=0.5)

            if detected:
                # GT cubierto por una propuesta → dibujar en verde tenue
                label = f"GT-DET {cat_id}:{get_category_name(cat_id)}"
                _draw_box(img_rejected, gt_bbox, (0, 160, 0), label)
            else:
                # GT no cubierto → falso negativo, dibujar en rojo
                label = f"GT-MISS {cat_id}:{get_category_name(cat_id)}"
                _draw_box(img_rejected, gt_bbox, COLOR_REJECTED, label)

        out_rejected = os.path.join(dirs['rejected'], f"{image_id}.jpg")
        _save_image(img_rejected, out_rejected)

    print(f"[visualize_proposals] Listo.")
    print(f"  Aceptadas → {dirs['accepted']}")
    print(f"  Rechazadas/GT-miss → {dirs['rejected']}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: IoU entre un bbox GT y lista de predicciones
# ─────────────────────────────────────────────────────────────────────────────

def _is_detected(gt_bbox: list, predictions: list, iou_threshold: float = 0.5) -> bool:
    """
    Determina si un bbox GT está cubierto por al menos una predicción
    con IoU >= iou_threshold.

    Params:
        gt_bbox      (list) : bbox GT en formato [x, y, w, h].
        predictions  (list) : lista de dicts con campo 'bbox' [x, y, w, h].
        iou_threshold(float): umbral mínimo de IoU para considerar detección.

    Returns:
        bool: True si al menos una predicción cubre el GT con IoU >= umbral.
    """
    gx, gy, gw, gh = gt_bbox
    gx2, gy2 = gx + gw, gy + gh

    for pred in predictions:
        px, py, pw, ph = pred['bbox']
        px2, py2 = px + pw, py + ph

        # Intersección
        ix1 = max(gx, px)
        iy1 = max(gy, py)
        ix2 = min(gx2, px2)
        iy2 = min(gy2, py2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue  # sin intersección

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        gt_area    = gw * gh
        pred_area  = pw * ph
        union_area = gt_area + pred_area - inter_area

        if union_area <= 0:
            continue

        iou = inter_area / union_area
        if iou >= iou_threshold:
            return True

    return False