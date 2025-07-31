import torch
from collections import Counter

from .iou import intersection_over_union

def mean_average_precision(
    true_boxes, pred_boxes, 
    iou_threshold=0.5, 
    box_format="corners", 
    num_classes=80
):
    """
    Calculates mean average precision 

    Params
    :true_boxes (list) -> Similar as pred_boxes except all the correct ones 
    :pred_boxes (list) -> list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    :iou_threshold (float) -> threshold where predicted bboxes is correct
    :box_format (str) -> "midpoint" or "corners" used to specify bboxes
    :num_classes (int) -> number of classes

    Returns
    mAP (float) -> mAP value across all classes given a specific IoU threshold 
    aRE (float) -> mean average recalls.
    mIoU (float) -> mean of IoU between all predictions and ground truth.
    [
        gt_detected (list[bool]) -> whether a ground truth was detected or not.
        gt_img_ids (list[int])   -> img ids indicating the image the ground truth belongs to.
    ]
    preds_oracle (list[tuples])  -> list with corrections from the oracle.
            e.g. [(False, [0.0, 0.0, 0.0, 0.0]), (False, [0.0, 0.0, 0.0, 0.0])]
    """
    # list storing all AP for respective classes
    avg_precisions = []
    avg_recalls = []

    # used for numerical stability later on
    epsilon = 1e-6
    mIoU = mIoU_total = 0.0

    # ------------------
    # record if a ground truth was detected or not
    gt_detected = [False] * len(true_boxes)
    gt_img_ids = [bbox[0] for bbox in true_boxes]

    # oracle with respect predictions
    preds_oracle = [(False,[0.0,0.0,0.0,0.0])] * len(pred_boxes)

    if len(pred_boxes) == 0:
        return [0.0, 0.0, 0.0, [gt_img_ids,gt_detected], preds_oracle]

    # add a positional index to the ground truth and preds
    # avoid if it has been done in previous calls
    if len(true_boxes[0]) == 7:
        # from [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        # to   [train_idx, class_prediction, prob_score, x1, y1, x2, y2, positional_arg]
        for idx,bboxes in enumerate(true_boxes):
            bboxes.append(idx)
        for idx, pred in enumerate(pred_boxes):
            pred.append(idx)
    # ------------------

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        TP_new = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        FP_new = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            oracle_item = [False, 0.0, 0.0, 0.0, 0.0]
            best_iou_imgid = 0

            # compare current detection against all gt and 
            # record the best 
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

                    # keep track of the ground truth id
                    # so we can mark that ground truth as
                    # FOUND!
                    best_iou_imgid = gt[-1]

                    # keep track of the positional predictions
                    # to see if this can be replaced as oracle prediction
                    oracle_item = (detection[-1], gt[3:])


            # keep track of the whole intersection
            mIoU += best_iou
            mIoU_total += 1.0

            if best_iou > iou_threshold:
                # only detect ground truth detection once ==> regular mAP
                # ALSO, I keep track even if the ground truth is detected more
                # than once because I care about the quality of the bboxes 
                # because they represent pseudolabels.
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                    #TP_new[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
                    #TP_new[detection_idx] = 1

                TP_new[detection_idx] = 1

                # mark this ground truth as found
                gt_detected[best_iou_imgid] = True

                # use the oracle to correct the coordinates
                preds_oracle[oracle_item[0]] = (True, oracle_item[1]) 

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
                FP_new[detection_idx] = 1


        # classic calculation of mAP
        # necessary to get the recall
        TP_cumsum = torch.cumsum(TP, dim=0) # [1,1,0,1,0] -> [1,2,2,3,3]
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # old code
        #precisions = torch.cat((torch.tensor([1]), precisions))
        #recalls = torch.cat((torch.tensor([0]), recalls))
        #torch.trapz for numerical integration
        #average_precisions.append(torch.trapz(precisions, recalls))

        # new mAP using all detections (cumulative) that are considered TP 
        TP_cumsum_new = torch.cumsum(TP_new, dim=0) # [1,1,0,1,0] -> [1,2,2,3,3]
        FP_cumsum_new = torch.cumsum(FP_new, dim=0)
        recalls_new = TP_cumsum_new / (total_true_bboxes + epsilon)
        precisions_new = TP_cumsum_new / (TP_cumsum_new + FP_cumsum_new + epsilon)

        #append results
        if precisions_new.numel() > 0:
            avg_precisions.append(precisions_new[-1].item())
        else:
            avg_precisions.append(0.0)
        if recalls.numel() > 0:
            avg_recalls.append(recalls[-1].item())
        else:
            avg_recalls.append(0.0)

    mIoU_ = 0.0
    if isinstance((mIoU / mIoU_total), float):
        mIoU_ = (mIoU / mIoU_total) if mIoU_total > 0.0 else 0.0
    else:
        mIoU_ = (mIoU / mIoU_total).item() if mIoU_total > 0.0 else 0.0
    return [ (sum(avg_precisions) / len(avg_precisions)) if len(avg_precisions) > 0.0 else 0.0,
             (sum(avg_recalls) / len(avg_recalls)) if len(avg_recalls) > 0.0 else 0.0, 
             mIoU_,
             [gt_img_ids, gt_detected],
             preds_oracle
            ]