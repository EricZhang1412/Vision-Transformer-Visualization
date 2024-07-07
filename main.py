import argparse
import datetime
import json
import numpy as np
import cv2
import os
import time
from pathlib import Path
import importlib

import torch

# import torchinfo
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.5.4"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_spikformer as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.kd_loss import DistillationLoss
from PIL import Image
from torchvision import transforms
import models
# import models_sew
from urllib.request import urlretrieve
from models.engine_finetune import train_one_epoch, evaluate
from timm.data import create_loader
import torch
import torchinfo
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.clock_driven import layer
from timm.models import (
    create_model,
    resume_checkpoint
)
import models
from models.modeling import VisionTransformer, CONFIGS
from models import v2_models
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

os.makedirs("attention_data", exist_ok=True)
if not os.path.isfile("attention_data/ilsvrc2012_wordnet_lemmas.txt"):
    urlretrieve("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt", "attention_data/ilsvrc2012_wordnet_lemmas.txt")

imagenet_labels = dict(enumerate(open('attention_data/ilsvrc2012_wordnet_lemmas.txt')))


def get_args_parser():
    # important params
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--test_img_dir",
        default="",
        type=str,
        help="Directory of Image-Under-Test",
    )
    parser.add_argument(
        "--modelUT",
        default="sdt-v2_8_512",
        type=str,
        help="sdt-v2_8_512, sdt-v1, vanilla_ViT_b_16",
    )
    parser.add_argument(
        "--parameterUT",
        default="qk",
        type=str,
        help="qk[SDT-v1,SDT-v2,Vanilla-ViT], qk_hp[SDT-v1,SDT-v2], attn_mp[SDT-v2]",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)  # 20/30(T=4)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--data_path", default="/raid/ligq/imagenet1-k/", type=str, help="dataset path"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="spikformer_8_384_CAFormer",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--model_mode",
        default="ms",
        type=str,
        help="Mode of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=6e-4,
        metavar="LR",  # 1e-5,2e-5(T=4)
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--time_steps", default=1, type=int)

    # Dataset parameters

    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--output_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distillation parameters
    parser.add_argument(
        "--kd",
        action="store_true",
        default=False,
        help="kd or not",
    )
    parser.add_argument(
        "--teacher_model",
        default="caformer_b36_in21ft1k",
        type=str,
        metavar="MODEL",
        help='Name of teacher model to train (default: "caformer_b36_in21ft1k"',
    )
    parser.add_argument(
        "--distillation_type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="",
    )
    parser.add_argument("--distillation_alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation_tau", default=1.0, type=float, help="")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    model_under_test = args.modelUT
    if model_under_test == "sdt-v2_8_512":
        model = v2_models.__dict__[args.model](kd=args.kd)
        model.T = args.time_steps
        model.eval()
        checkpoint = torch.load("./checkpoint/55M_kd_T4.pth", map_location=torch.device('cpu'))
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict, strict=False)
    elif model_under_test == "sdt-v1":
        model = create_model(
            "sdt",
            T=4,
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.2,
            num_heads=8,
            num_classes=1000,
            pooling_stat='1111',
            img_size_h=224,
            img_size_w=224,
            embed_dims=384,
            mlp_ratios=4,
            in_channels=3,
            qkv_bias=False,
            depths=8,
            sr_ratios=1,
            spike_mode="lif",
            dvs_mode=False,
            TET=False,
            )
        model.eval()
        resume_checkpoint(
                model,
                "./checkpoint/8_384.pth.tar",
                optimizer=None,
                loss_scaler=None,
                log_info=True,
                )
    elif model_under_test == "vanilla_ViT_b_16":
        config = CONFIGS["ViT-B_16"]
        model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        model.load_from(np.load("attention_data/ViT-B_16-224.npz"))
        model.eval()

    

    # im = Image.open("/data/dataset/ImageNet/val/n07892512/ILSVRC2012_val_00033598.JPEG")
    im = Image.open(args.test_img_dir)
    x = transform(im)
    
    if model_under_test == "sdt-v2_8_512":
        y, qk, attn_lif_mp, qk_hadmard_product_sum = model(x)
    elif model_under_test == "sdt-v1":
        fr_dict, nz_dict = {"t0": dict(), "t1": dict(), "t2": dict(), "t3": dict()}, {
                    "t0": dict(),
                    "t1": dict(),
                    "t2": dict(),
                    "t3": dict(),
                    }
        cls_output, firing_dict, qk, kv_mp, qk_hadmard_product_sum = model(x, hook=dict())
    elif model_under_test == "vanilla_ViT_b_16":
        logits, att_mat = model(x.unsqueeze(0))


    parameter_under_test = args.parameterUT
    if parameter_under_test == "qk":
        if model_under_test != "vanilla_ViT_b_16":
            qk_time_avg = []
            for qk_item in qk:
                qk_item = qk_item.mean(dim=0)
                qk_item = qk_item.squeeze(0)
                qk_item = qk_item.mean(dim=0)
                qk_item = qk_item.squeeze(0)

                qk_time_avg.append(qk_item)


            print(len(qk_time_avg))
            print(qk_time_avg[0].shape)

            ########attn_lif_mp_visualization##########
            attention_maps = []
            for i in range(0, len(qk_time_avg)):
                tmp_attn_map = qk_time_avg[i].reshape(14, 14)
                # print(tmp_attn_map.shape)
                grid_size = tmp_attn_map.shape[0]
                mask = tmp_attn_map.detach().numpy()
                # print(mask.shape)
                mask = cv2.resize(mask / mask.max(), im.size)
                mask = (mask * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                output_filename = f'attention_mask_{i}.png'
                attention_maps.append(heatmap)
                cv2.imwrite(output_filename, heatmap)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps) + 1
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap
                # canvas.append(im)

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps) + 1
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap
                # canvas.append(im)

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)

        else:
            att_mat = torch.stack(att_mat).squeeze(1) # torch.stack()进行扩维拼接
            # print(att_mat.shape)

            # Average the attention weights across all heads.
            att_mat = torch.mean(att_mat, dim=1)
            # print(att_mat.shape)
            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(1))
            aug_att_mat = att_mat + residual_att
            # print(aug_att_mat.shape)
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
            # print(aug_att_mat.shape)

            #-----------my code ---------#
            #print each block's attention map
            print(aug_att_mat.shape)

            attention_maps = []
            for i in range(0, aug_att_mat.shape[0]):
                tmp_attn_map = aug_att_mat[i]
                grid_size = int(np.sqrt(aug_att_mat.size(-1)))
                mask = tmp_attn_map[0, 1:].reshape(grid_size, grid_size).detach().numpy()
                # print(mask.shape)
                mask = cv2.resize(mask / mask.max(), im.size)
                mask = (mask * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                output_filename = f'attention_mask_{i}.png'
                attention_maps.append(heatmap)
                # cv2.imwrite(output_filename, heatmap)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps)
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)

    elif parameter_under_test == "qk_hp":
        if model_under_test != "vanilla_ViT_b_16":
            qk_hadmard_product_sum_time_avg = []
            for qk_item in qk_hadmard_product_sum:
                qk_item = qk_item.mean(dim=0)
                qk_item = qk_item.squeeze(0)
                qk_item = qk_item.mean(dim=0)
                qk_item = qk_item.squeeze(0)
                qk_hadmard_product_sum_time_avg.append(qk_item)
            attention_maps = []
            for i in range(0, len(qk_hadmard_product_sum_time_avg)):
                tmp_attn_map = qk_hadmard_product_sum_time_avg[i].reshape(14, 14)
                # print(tmp_attn_map.shape)
                grid_size = tmp_attn_map.shape[0]
                mask = tmp_attn_map.detach().numpy()
                # print(mask.shape)
                mask = cv2.resize(mask / mask.max(), im.size)
                mask = (mask * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                output_filename = f'attention_mask_{i}.png'
                attention_maps.append(heatmap)
                cv2.imwrite(output_filename, heatmap)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps) + 1
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap
                # canvas.append(im)

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps) + 1
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap
                # canvas.append(im)

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)
    elif parameter_under_test == "attn_mp":
        if model_under_test == "sdt-v2_8_512":
            for i in range(len(attn_lif_mp)):
                attn_lif_mp[i] = attn_lif_mp[i].squeeze(0)
                attn_lif_mp[i] = attn_lif_mp[i].squeeze(0)
                attn_lif_mp[i] = attn_lif_mp[i].T
            print(attn_lif_mp[0].shape)
            attention_maps = []
            for i in range(0, len(attn_lif_mp)):
                tmp_attn_map = attn_lif_mp[i].mean(dim=1).reshape(14, 14)
                # print(tmp_attn_map.shape)
                grid_size = tmp_attn_map.shape[0]
                mask = tmp_attn_map.detach().numpy()
                # print(mask.shape)
                mask = cv2.resize(mask / mask.max(), im.size)
                mask = np.clip(mask * 255 * 0.6, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                output_filename = f'attention_mask_{i}.png'
                attention_maps.append(heatmap)
                cv2.imwrite(output_filename, heatmap)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps) + 1
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap
                # canvas.append(im)

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)

            map_height, map_width, _ = attention_maps[0].shape
            num_maps = len(attention_maps) + 1
            canvas_height = map_height
            canvas_width = map_width * num_maps

            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            for i, heatmap in enumerate(attention_maps):
                canvas[:, i * map_width:(i + 1) * map_width, :] = heatmap
                # canvas.append(im)

            # 将原始图像转换为 numpy 数组
            im = np.array(im)
            if im.ndim == 2:  # 如果是灰度图像，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:  # 如果有 alpha 通道，则转换为 BGR
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            # 将原始图像放在最后
            canvas[:, -map_width:, :] = cv2.resize(im, (map_width, map_height))
            cv2.imwrite('attention_maps_canvas.png', canvas)

    # cv2.imwrite('attention_maps_canvas.png', canvas)
    # # # Recursively multiply the weight matrices
    # # joint_attentions = torch.zeros(aug_att_mat.size())
    # # joint_attentions[0] = aug_att_mat[0]

    # # for n in range(1, aug_att_mat.size(0)):
    # #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # # # Attention from the output token to the input space.
    # # v = joint_attentions[-1]
    # # # print(v.shape)
    # # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # # # print(aug_att_mat.size(-1))
    # # # print(np.sqrt(aug_att_mat.size(-1)))
    # # # print(grid_size)
    # # pre_mask = v[0, 1:].detach().numpy()
    # # print(pre_mask)
    # # mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    # # print(mask)

    # # mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    # # print(mask)
    # # result = (mask * im).astype("uint8")

    # # im = np.array(im)
    # # black_image = np.zeros_like(im)
    # # colored_mask = (mask * black_image + (1 - mask) * im).astype(np.uint8)

    # # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    # # ax1.set_title('Original')
    # # ax2.set_title('Attention Map')
    # # _ = ax1.imshow(im)
    # # _ = ax2.imshow(result)

    # # fig.savefig('output_image.png')

    # # probs = torch.nn.Softmax(dim=-1)(logits)
    # # top5 = torch.argsort(probs, dim=-1, descending=True)
    # # print("Prediction Label and Attention Map!\n")
    # # for idx in top5[0, :5]:
    # #     print(f'{probs[0, idx.item()]:.5f} : {imagenet_labels[idx.item()]}', end='')



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)