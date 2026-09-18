"""
Detection evaluation kit — the Ultralytics validation protocol, on cached detections.

  * matching:  same greedy IoU matching as ultralytics.engine.validator.match_predictions
  * mAP:       ultralytics.utils.metrics.ap_per_class (101-point interpolated AP)
  * per image: F1 at IoU 0.5 / conf >= 0.25, used only as a per-image quality proxy
"""
import numpy as np

from ultralytics.utils.metrics import ap_per_class

IOU_THRS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def match(det_xyxy, det_cls, gt_xyxy, gt_cls) -> np.ndarray:
    """tp[n_det, 10]: is each detection a true positive at IoU thresholds 0.50:0.05:0.95."""
    tp = np.zeros((len(det_xyxy), len(IOU_THRS)), bool)
    if len(det_xyxy) == 0 or len(gt_xyxy) == 0:
        return tp
    iou = box_iou(np.asarray(gt_xyxy, np.float32), det_xyxy)            # [n_gt, n_det]
    iou = iou * (np.asarray(gt_cls)[:, None] == det_cls[None])
    for k, thr in enumerate(IOU_THRS):
        g, d = np.nonzero(iou >= thr)
        if len(g) == 0:
            continue
        m = np.stack([g, d, iou[g, d]], 1)
        m = m[m[:, 2].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[m[:, 2].argsort()[::-1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
        tp[m[:, 1].astype(int), k] = True
    return tp


def mean_ap(dets, gts):
    """
    dets: list of (xyxy, conf, cls) per image; gts: list of (xyxy, cls) per image.
    Returns (mAP50-95, mAP50).
    """
    tps, confs, pcls, tcls = [], [], [], []
    for (bx, cf, cl), (gb, gc) in zip(dets, gts):
        tps.append(match(bx, cl, gb, gc))
        confs.append(cf)
        pcls.append(cl)
        tcls.append(np.asarray(gc, np.int64))
    tp = np.concatenate(tps)
    res = ap_per_class(tp, np.concatenate(confs), np.concatenate(pcls), np.concatenate(tcls))
    ap = res[5]                                                          # [n_classes, 10]
    return float(ap.mean()), float(ap[:, 0].mean())


def image_f1(det, gt, conf_thr=0.25, iou_thr=0.5) -> float:
    """Per-image F1 (conf >= conf_thr, IoU >= iou_thr); 1.0 if nothing to find and nothing found."""
    bx, cf, cl = det
    keep = cf >= conf_thr
    bx, cl = bx[keep], cl[keep]
    gb, gc = gt
    n_gt, n_det = len(gc), len(bx)
    if n_gt == 0 and n_det == 0:
        return 1.0
    tp = int(match(bx, cl, gb, gc)[:, 0].sum()) if n_gt and n_det else 0
    return 2.0 * tp / (n_det + n_gt)


def image_stats(det, conf_thr=0.25):
    """Label-free features of a detection set (available at deployment)."""
    bx, cf, cl = det
    k = cf >= conf_thr
    n = int(k.sum())
    top = np.sort(cf)[::-1]
    area = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]))[k]
    return {
        "n": n,
        "n50": int((cf >= 0.5).sum()),
        "max": float(top[0]) if len(top) else 0.0,
        "mean": float(cf[k].mean()) if n else 0.0,
        "top3": float(top[:3].mean()) if len(top) else 0.0,
        "small": int((area < 32 * 32).sum()) if n else 0,
        "area": float(np.sqrt(area).mean()) if n else 0.0,
    }
