import os
import time
import torch
from torch.utils.data import DataLoader

from src.dataset import TridosDataset, split_videos_real
from src.model import TridosYOLOX
from src.config import (
    DATA_ROOT, CHECKPOINT_DIR, INPUT_SIZE, TIME_WINDOW, BATCH_SIZE,
    EPOCHS, INIT_LR, FP16, TIME_BUDGET_MIN, VAL_RESERVE_MIN, VAL_MAX_BATCHES,
)


def collate_fn(batch):
    frames = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return frames, labels


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RUN_START = time.time()
DEADLINE = RUN_START + TIME_BUDGET_MIN * 60
TRAIN_DEADLINE = DEADLINE - VAL_RESERVE_MIN * 60

train_videos, val_videos = split_videos_real(DATA_ROOT)
train_dataset = TridosDataset(DATA_ROOT, input_size=INPUT_SIZE, time_window=TIME_WINDOW, video_list=train_videos)
val_dataset = TridosDataset(DATA_ROOT, input_size=INPUT_SIZE, time_window=TIME_WINDOW, video_list=val_videos)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

model = TridosYOLOX(time_window=TIME_WINDOW).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=FP16)

start_epoch = 0
best_val_loss = float("inf")
resume_path = os.path.join(CHECKPOINT_DIR, "last.pth")
if os.path.exists(resume_path):
    print(f"Resuming from {resume_path}")
    ckpt = torch.load(resume_path, map_location="cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    for _ in range(start_epoch):
        scheduler.step()
    print(f"Resumed at epoch {start_epoch + 1}, best_val_loss={best_val_loss}")


def run_epoch(loader, train=True, deadline=None, max_batches=None):
    model.train(mode=train)
    if not train:
        model.head.training = True   # head returns losses, not predictions
    total_loss, n_batches = 0.0, 0
    cut_short = False
    t0 = time.time()

    for frames, labels in loader:
        if deadline is not None and time.time() > deadline:
            cut_short = True
            break
        if max_batches is not None and n_batches >= max_batches:
            break

        frames = frames.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=FP16):
                loss, loss_iou, loss_obj, loss_cls, loss_l1, fg_ratio = model(frames, labels=labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % 100 == 0:
            elapsed = (time.time() - RUN_START) / 60
            print(f"  [{'train' if train else 'val'}] batch {n_batches}/{len(loader)} "
                  f"loss={loss.item():.4f} iou={loss_iou.item():.4f} obj={loss_obj.item():.4f} "
                  f"cls={loss_cls.item():.4f} fg={fg_ratio:.2f} | {elapsed:.1f} min elapsed")

    return total_loss / max(n_batches, 1), time.time() - t0, n_batches, cut_short


for epoch in range(start_epoch, EPOCHS):
    if time.time() > TRAIN_DEADLINE:
        print("Time budget reached before starting a new epoch - stopping.")
        break

    train_loss, train_time, n_train, cut_short = run_epoch(train_loader, True, deadline=TRAIN_DEADLINE)
    val_loss, val_time, n_val, _ = run_epoch(val_loader, False, max_batches=VAL_MAX_BATCHES)
    scheduler.step()

    note = f" (cut short after {n_train}/{len(train_loader)} batches)" if cut_short else ""
    print(f"[Epoch {epoch+1}/{EPOCHS}]{note} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"lr={optimizer.param_groups[0]['lr']:.6f} train_time={train_time:.0f}s val_time={val_time:.0f}s")

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    checkpoint = {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "last.pth"))
    if is_best:
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best.pth"))
        print(f"  -> new best model saved (val_loss={val_loss:.4f})")

    if cut_short:
        print(f"Time budget reached - stopping. Total run: {(time.time() - RUN_START)/60:.1f} min. Progress saved to Drive.")
        break
