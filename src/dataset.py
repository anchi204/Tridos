import os
import glob
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_annotations(ann_path):
    ann = {}
    if not os.path.exists(ann_path):
        return ann
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            filename = parts[0]
            boxes = []
            for box_str in parts[1:]:
                x1, y1, x2, y2, cls = box_str.split(",")
                boxes.append([float(cls), float(x1), float(y1), float(x2), float(y2)])
            ann[filename] = boxes
    return ann


class TridosDataset(Dataset):
    MAX_BOXES = 30

    def __init__(self, data_root, input_size=512, time_window=5, video_list=None):
        self.data_root = data_root
        self.images_root = os.path.join(data_root, "images")
        self.ann_root = os.path.join(data_root, "annotations")
        self.input_size = input_size
        self.T = time_window

        if video_list is None:
            video_list = sorted(d for d in os.listdir(self.images_root)
                                if os.path.isdir(os.path.join(self.images_root, d)))
        self.video_list = video_list

        self.samples = []
        self.video_cache = {}
        for video in self.video_list:
            frame_dir = os.path.join(self.images_root, video)
            frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.bmp")) +
                                 glob.glob(os.path.join(frame_dir, "*.jpg")) +
                                 glob.glob(os.path.join(frame_dir, "*.png")))
            if len(frame_files) < self.T:
                continue
            ann = _load_annotations(os.path.join(self.ann_root, f"{video}.txt"))
            self.video_cache[video] = (frame_files, ann)
            for t in range(self.T - 1, len(frame_files)):
                self.samples.append((video, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, keyframe_idx = self.samples[idx]
        frame_files, ann = self.video_cache[video]
        window_files = frame_files[keyframe_idx - self.T + 1: keyframe_idx + 1]

        frames = []
        scale_x = scale_y = 1.0
        for fp in window_files:
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (self.input_size, self.input_size))
            scale_x, scale_y = self.input_size / w0, self.input_size / h0
            img = img.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img).permute(2, 0, 1))
        frames = torch.stack(frames, dim=0)

        keyframe_name = os.path.basename(window_files[-1])
        raw_boxes = ann.get(keyframe_name, [])

        labels = torch.zeros(self.MAX_BOXES, 5)
        for i, (cls, x1, y1, x2, y2) in enumerate(raw_boxes[:self.MAX_BOXES]):
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            labels[i] = torch.tensor([cls, cx, cy, w, h])

        return frames, labels


def split_videos_real(data_root):
    splits_path = os.path.join(data_root, "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    return splits["train"], splits["val"]
