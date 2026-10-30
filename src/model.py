import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.swin_transformer import SwinTransformerBlock
from yolox.models import YOLOPAFPN, YOLOXHead
from yolox.utils import bboxes_iou

""" + s6_clean + "
" + s7_clean + "
" + s8_clean + "
" + s9_clean + "
" + s10.rstrip() + "

" + s11.rstrip() + "

" + s12.rstrip() + "
