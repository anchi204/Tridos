import os
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from yolox.utils import postprocess

from src.dataset import TridosDataset, collate_fn
from src.model import TridosYOLOX
from src.config import DATA_ROOT, CHECKPOINT_DIR, INPUT_SIZE, TIME_WINDOW


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


@torch.no_grad()
def evaluate(ckpt_path, dataset, thresholds=(0.05, 0.2, 0.4), nms_thre=0.65,
             batch_stride=4, iou_match=0.5):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    for bi, (frames, labels) in enumerate(loader):
        if bi % batch_stride != 0:
            continue
        with torch.cuda.amp.autocast(enabled=True):
            preds = model(frames.cuda())
        preds = preds.float()

        for t in thresholds:
            dets = postprocess(preds.clone(), num_classes=1, conf_thre=t, nms_thre=nms_thre)
            for b in range(frames.shape[0]):
                gt = labels[b]
                gt = gt[gt.sum(1) > 0]
                gt_boxes = cxcywh_to_xyxy(gt[:, 1:5]).cuda()
                det = dets[b]
                n_det = 0 if det is None else det.shape[0]
                n_gt = gt_boxes.shape[0]
                if n_det == 0:
                    stats[t]["fn"] += n_gt
                    continue
                if n_gt == 0:
                    stats[t]["fp"] += n_det
                    continue

                order = (det[:, 4] * det[:, 5]).argsort(descending=True)
                ious = box_iou(det[order][:, :4], gt_boxes)
                matched, tp = set(), 0
                for i in range(n_det):
                    j = ious[i].argmax().item()
                    if ious[i, j] >= iou_match and j not in matched:
                        matched.add(j)
                        tp += 1
                stats[t]["tp"] += tp
                stats[t]["fp"] += n_det - tp
                stats[t]["fn"] += n_gt - tp

    print(f"\nResults for {ckpt_path}")
    for t in thresholds:
        s = stats[t]
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f" conf>{t}: precision={p*100:.1f}% recall={r*100:.1f}% F1={f1*100:.1f}% "
              f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})")


train_videos, val_videos = __import__("src.dataset", fromlist=["split_videos_real"]).split_videos_real(DATA_ROOT)
val_dataset = TridosDataset(DATA_ROOT, input_size=INPUT_SIZE, time_window=TIME_WINDOW, video_list=val_videos)
model = TridosYOLOX(time_window=TIME_WINDOW).cuda()
evaluate(os.path.join(CHECKPOINT_DIR, "best.pth"), val_dataset)
