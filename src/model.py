import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.swin_transformer import SwinTransformerBlock
from yolox.models import YOLOPAFPN, YOLOXHead
from yolox.utils import bboxes_iou


class NonLocalAttentionBlock(nn.Module):
    """NAB — self-attention over spatial positions, matching the diagram exactly."""
    def __init__(self, channels, gamma=1.0):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.scale = channels ** 0.5

    def forward(self, f_l):
        B, C, H, W = f_l.shape
        q = self.q_proj(f_l).flatten(2).permute(0, 2, 1)
        k = self.k_proj(f_l).flatten(2)
        v = self.v_proj(f_l).flatten(2).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) / self.scale, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return self.gamma * out + f_l


class MemoryEnhancementUnit(nn.Module):
    """MEU — four embedding layers (K_Q/V_Q from keyframe, K_M/V_M from NAB output)."""
    def __init__(self, channels):
        super().__init__()
        self.f_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.f_m = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_m = nn.Conv2d(channels, channels, kernel_size=1)
        self.matching = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, f_t, f_g):
        B, C, H, W = f_t.shape
        k_q = self.f_q(f_t).flatten(2)
        v_q = self.v_q(f_t).flatten(2)
        k_m = self.f_m(f_g).flatten(2)
        v_m = self.v_m(f_g).flatten(2)
        m_s = torch.softmax(torch.bmm(k_m.permute(0, 2, 1), k_q), dim=1)
        retrieved = torch.bmm(v_m, m_s)
        fused = torch.cat([retrieved, v_q], dim=1).view(B, 2 * C, H, W)
        return self.matching(fused)


class MSRM(nn.Module):
    """
    FIX applied here vs. the earlier laptop version: reference frames are
    concatenated (matching the diagram's 'C' node), not averaged, before the
    gate convolution — preserving each reference frame's individual signal.
    """
    def __init__(self, channels, time_window):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels * (time_window - 1), channels, kernel_size=3, padding=1)
        self.merge_conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.nab = NonLocalAttentionBlock(channels)
        self.meu = MemoryEnhancementUnit(channels)

    def forward(self, fc):
        """fc: [B, T, C, H, W], last index T-1 is the keyframe."""
        B, T, C, H, W = fc.shape
        f_t = fc[:, -1]
        ref_frames = fc[:, :-1]

        ref_concat = ref_frames.reshape(B, (T - 1) * C, H, W) # CONCAT, not mean
        gate = torch.sigmoid(self.gate_conv(ref_concat))
        f_hat_t = gate * f_t
        f_l = self.merge_conv(torch.cat([f_hat_t, f_t], dim=1))

        f_g = self.nab(f_l)
        f_s = self.meu(f_t, f_g)
        return f_s


# Test on dummy data matching Scale 2 (16x16, 512 channels) with T=5 (paper's real setting)
T = 5
msrm = MSRM(channels=512, time_window=T).cuda()
dummy_clip = torch.randn(2, T, 512, 16, 16).cuda()
out = msrm(dummy_clip)
print("MSRM output shape:", out.shape)


class ResBlock(nn.Module):
    """
    Diagram shows: Conv -> PReLU -> Conv -> (add residual).
    FIX vs. earlier laptop version: only ONE PReLU, positioned between the
    two convs — not one after each conv.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class TDEM(nn.Module):
    """
    Two fixes vs. the laptop version:
      1. Diffs are STACKED (concatenated), not averaged, preserving each
         individual frame-to-frame transition rather than blurring them together.
      2. Conv and ResB1 operate at LOW resolution (right after pooling), and
         upsampling happens AFTERWARD on each path separately — the earlier
         version upsampled first, which doesn't match the diagram's ordering.
    """
    def __init__(self, channels, time_window):
        super().__init__()
        T = time_window
        num_diffs = T - 1
        self.stack_conv = nn.Conv2d(channels * num_diffs, channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.shared_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.resb1 = ResBlock(channels)
        self.resb2 = ResBlock(channels)

    def forward(self, fc):
        """fc: [B, T, C, H, W]"""
        B, T, C, H, W = fc.shape
        diffs = fc[:, 1:] - fc[:, :-1]                          # [B, T-1, C, H, W]
        diffs_stacked = diffs.reshape(B, (T - 1) * C, H, W)      # STACK, not mean
        x = self.stack_conv(diffs_stacked)
        f_d = self.pool(x)                                        # low-res F_D

        conv_fd = self.shared_conv(f_d)                            # Conv(F_D), once, at low-res

        # Path A: direct — upsample right after the shared conv
        path_a = F.interpolate(conv_fd, size=(H, W), mode="bilinear", align_corners=False)
        f_t = fc[:, -1]
        f_u = f_t + path_a

        # Path B: refined via ResB1 first, then upsample
        path_b = self.resb1(conv_fd)
        path_b_up = F.interpolate(path_b, size=(H, W), mode="bilinear", align_corners=False)

        f_out = self.resb2(f_u) + path_b_up
        return f_out


# Test on the same scale (16x16, 512 channels), T=5
tdem = TDEM(channels=512, time_window=T).cuda()
out = tdem(dummy_clip)
print("TDEM output shape:", out.shape)


class AttentionGate(nn.Module):
    """
    The M_a / M_p gates in the diagram: avg+max pool -> DW Conv -> ReLU ->
    PW Conv -> Sigmoid. FIX vs. laptop version: this is a real depthwise+
    pointwise pair (matching the diagram), not a single plain convolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        self.pw = nn.Conv2d(2, channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        q = torch.cat([avg, mx], dim=1)
        out = F.relu(self.dw(q))
        return torch.sigmoid(self.pw(out))


class FrequencyRefine(nn.Module):
    """Per-frame amplitude/phase refinement, Eq. 10-16, matching the diagram."""
    def __init__(self, channels):
        super().__init__()
        self.amp_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.amp_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.amp_gate = AttentionGate(channels)

        self.phase_dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=False)
        self.phase_pw = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.phase_gate = AttentionGate(channels)

    def forward(self, x):
        spectrum = torch.fft.fft2(x.float(), norm="ortho")
        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        amp_feat = F.relu(self.amp_pw(F.relu(self.amp_dw(amplitude))))
        amp_gate_val = self.amp_gate(amp_feat)
        amp_refined = amp_feat * amp_gate_val
        amp_residual = amp_refined - amplitude

        phase_in = torch.cat([phase, amp_residual], dim=1)
        phase_feat = torch.sigmoid(self.phase_pw(F.relu(self.phase_dw(phase_in))))
        phase_gate_val = self.phase_gate(phase_feat)
        phase_refined = phase_feat * phase_gate_val

        phase_final = 2 * torch.pi * phase_refined - torch.pi
        real = amp_refined * torch.cos(phase_final)
        imag = amp_refined * torch.sin(phase_final)
        spectrum_refined = torch.complex(real.float(), imag.float())

        spatial = torch.fft.ifft2(spectrum_refined, norm="ortho").real
        return x + spatial


class GlobalFrequencyBranch(nn.Module):
    def __init__(self, channels, resolution, num_heads=8, window_size=8, depth=2):
        super().__init__()
        self.resolution = resolution
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=channels, input_resolution=resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
            ) for i in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # -> [B, H, W, C], what this timm version expects
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 3, 1, 2)  # back to [B, C, H, W]


class LGFM(nn.Module):
    def __init__(self, channels, time_window, resolution, swin_depth=2):
        super().__init__()
        self.freq_refine = FrequencyRefine(channels)
        concat_ch = channels * time_window
        self.conv_branch = nn.Sequential(
            nn.Conv2d(concat_ch, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
        )
        self.global_reduce = nn.Conv2d(concat_ch, channels, kernel_size=1)
        self.global_branch = GlobalFrequencyBranch(channels, resolution, depth=swin_depth)
        self.global_out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, fc):
        B, T, C, H, W = fc.shape
        refined = [self.freq_refine(fc[:, t]) for t in range(T)]
        refined = torch.cat(refined, dim=1)
        f_lf = self.conv_branch(refined)
        f_gf = self.global_out_conv(self.global_branch(self.global_reduce(refined)))
        return f_lf, f_gf


# Test at Scale 2's actual dimensions: 16x16 spatial, 512 channels, T=5
lgfm = LGFM(channels=512, time_window=T, resolution=(16, 16), swin_depth=2).cuda()
f_lf, f_gf = lgfm(dummy_clip)
print("F_lf shape:", f_lf.shape)
print("F_gf shape:", f_gf.shape)


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CSABBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cab = ChannelAttentionBlock(channels)
        self.sab = SpatialAttentionBlock()
    def forward(self, x):
        return self.sab(self.cab(x)) + x


class RCU(nn.Module):
    def __init__(self, channels, num_csab=2):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.csabs = nn.Sequential(*[CSABBlock(channels) for _ in range(num_csab)])
    def forward(self, feat_a, feat_b):
        return self.csabs(self.reduce(torch.cat([feat_a, feat_b], dim=1)))


# Quick sanity test at Scale 2's dims
rcu = RCU(channels=512).cuda()
out = rcu(f_lf, f_gf)
print("RCU output shape:", out.shape)


class ScaleTridosBlock(nn.Module):
    def __init__(self, channels, time_window, resolution, swin_depth=2):
        super().__init__()
        self.msrm = MSRM(channels, time_window)
        self.tdem = TDEM(channels, time_window)
        self.st_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
        )
        self.lgfm = LGFM(channels, time_window, resolution, swin_depth)
        self.rcu1 = RCU(channels)
        self.rcu2 = RCU(channels)
        self.rcu3 = RCU(channels)

    def forward(self, fc):
        f_s = self.msrm(fc)
        f_t = self.tdem(fc)
        f_st = self.st_fuse(torch.cat([f_s, f_t], dim=1))
        f_lf, f_gf = self.lgfm(fc)
        f_stf1 = self.rcu1(f_lf, f_st)
        f_stf2 = self.rcu2(f_gf, f_st)
        return self.rcu3(f_stf1, f_stf2)


def nwd_loss_cxcywh(pred, target, C=12.8):
    cx1, cy1, w1, h1 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    cx2, cy2, w2, h2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    center_term = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    size_term = ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
    wasserstein_sq = center_term + size_term
    nwd = torch.exp(-torch.sqrt(wasserstein_sq.clamp(min=1e-7)) / C)
    return 1 - nwd


class TridosYOLOXHead(YOLOXHead):
    def __init__(self, *args, alpha_iou=0.5, beta_nwd=0.5, nwd_c=12.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_iou = alpha_iou
        self.beta_nwd = beta_nwd
        self.nwd_c = nwd_c

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                         bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                         cls_preds, obj_preds, mode="gpu"):
        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
        )
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        if num_in_boxes_anchor == 0:
            return (
                gt_classes.new_zeros(0),
                fg_mask,
                gt_classes.new_zeros(0, dtype=torch.float),
                gt_classes.new_zeros(0, dtype=torch.int64),
                0,
            )
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()).sqrt()
            cls_preds_ = torch.nan_to_num(cls_preds_, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + float(1e6) * (~geometry_relation)
        (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
         ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        cls_targets, reg_targets, l1_targets, obj_targets, fg_masks = [], [], [], [], []
        num_fg = 0.0
        num_gts = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                (gt_matched_classes, fg_mask, pred_ious_this_matching,
                 matched_gt_inds, num_fg_img) = self.get_assignments(
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                    cls_preds, obj_preds,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                if num_fg_img == 0:
                    cls_target = outputs.new_zeros((0, self.num_classes))
                    reg_target = outputs.new_zeros((0, 4))
                    l1_target = outputs.new_zeros((0, 4))
                else:
                    cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                    reg_target = gt_bboxes_per_image[matched_gt_inds]
                    if self.use_l1:
                        l1_target = self.get_l1_target(
                            outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds],
                            expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask],
                        )
                obj_target = fg_mask.unsqueeze(-1)
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        num_fg = max(num_fg, 1)
        pred_boxes_pos = bbox_preds.view(-1, 4)[fg_masks]
        iou_term = self.iou_loss(pred_boxes_pos, reg_targets).sum() / num_fg
        nwd_term = nwd_loss_cxcywh(pred_boxes_pos, reg_targets, C=self.nwd_c).sum() / num_fg
        loss_iou = self.alpha_iou * iou_term + self.beta_nwd * nwd_term
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1)


class TridosYOLOX(nn.Module):
    def __init__(self, time_window=5, depth=0.33, width=0.50, num_classes=1):
        super().__init__()
        self.time_window = time_window
        self.backbone = YOLOPAFPN(depth=depth, width=width)
        scale_channels = [int(128 * width * 2), int(256 * width * 2), int(512 * width * 2)]
        scale_resolutions = [(64, 64), (32, 32), (16, 16)]
        self.scale_blocks = nn.ModuleList([
            ScaleTridosBlock(ch, time_window, res)
            for ch, res in zip(scale_channels, scale_resolutions)
        ])
        self.head = TridosYOLOXHead(num_classes=num_classes, width=width)

    def forward(self, clip, labels=None):
        B, T, C, H, W = clip.shape
        flat = clip.view(B * T, C, H, W)
        fpn_feats = self.backbone(flat)
        fused_feats = []
        for scale_idx, feat in enumerate(fpn_feats):
            _, Cs, Hs, Ws = feat.shape
            feat_seq = feat.view(B, T, Cs, Hs, Ws)
            fused = self.scale_blocks[scale_idx](feat_seq)
            fused_feats.append(fused)
        keyframe_imgs = clip[:, -1]
        if labels is not None:
            return self.head(fused_feats, labels=labels, imgs=keyframe_imgs)
        else:
            return self.head(fused_feats)
