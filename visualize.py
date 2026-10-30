import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolox.utils import postprocess

from src.dataset import collate_fn
from src.config import CHECKPOINT_DIR


def cxcywh_to_xyxy_np(b):
    cx, cy, w, h = b
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


@torch.no_grad()
def visualize_predictions(ckpt_path, dataset, model, num_samples=6, conf_thre=0.3, nms_thre=0.65, seed=0):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        frames, labels = dataset[idx]
        clip = frames.unsqueeze(0).cuda()         # [1, T, 3, H, W]

        with torch.cuda.amp.autocast(enabled=True):
            preds = model(clip)             # inference mode, model.eval() -> head.training False
        preds = preds.float()

        dets = postprocess(preds.clone(), num_classes=1, conf_thre=conf_thre, nms_thre=nms_thre)[0]

        # keyframe is the last frame in the clip, matches how labels were built
        img = frames[-1].permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        ax.imshow(img)

        gt = labels[labels.sum(1) > 0]
        for box in gt:
            _, cx, cy, w, h = box.tolist()
            x1, y1, x2, y2 = cxcywh_to_xyxy_np([cx, cy, w, h])
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

        if dets is not None:
            for det in dets:
                x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.tolist()
                score = obj_conf * cls_conf
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, max(y1 - 5, 0), f"{score:.2f}", color="red", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.6, pad=0.5))

        n_gt, n_det = gt.shape[0], (0 if dets is None else dets.shape[0])
        ax.set_title(f"idx={idx} | GT={n_gt} (green) Pred={n_det} (red)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    output_path = "/content/drive/MyDrive/TridosColab/sample_predictions.png"
    plt.savefig(output_path, dpi=120)
    plt.show()
    print(f"Also saved to Drive: {output_path}")
