# Copyright (c) 2023, Alibaba Group
"""
Train an object detection model using domain adaptation / semi-supervised training.
Modified for paper-style alternating training:
- server: supervised only with labeled data
- client: unlabeled target only
"""

import json
import logging
import copy
from copy import deepcopy
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision

from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tqdm import tqdm

from .trainer import Trainer

# import val   # 你自己确认这里的导入路径，后面 after_epoch / after_train 需要用到 val.run
from models.backbone.experimental import attempt_load
from models.detector.yolo_ssod import Model

from models.loss.loss import DomainLoss, TargetLoss
# from models.loss import build_ssod_loss
from models.loss import build_loss, build_ssod_loss
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.datasets_ssod import create_target_dataloader, augment_hsv, cutout
from utils.downloads import attempt_download
from utils.general import (
    labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds,
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size,
    check_requirements, check_file, check_yaml, check_suffix, print_args,
    print_mutation, set_logging, one_cycle, colorstr, methods
)
from utils.labelmatch import LabelMatch
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.plots import plot_images, plot_labels, plot_results, plot_images_debug, output_to_target
from utils.self_supervised_utils import (
    FairPseudoLabel,
    check_pseudo_label_with_gt,
    check_pseudo_label
)
from utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device,
    torch_distributed_zero_first, is_parallel, time_sync, SemiSupModelEMA, CosineEMA
)

LOGGER = logging.getLogger(__name__)


class SSODTrainer(Trainer):
    def __init__(self, cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE):
        self.cfg = cfg
        self.set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)

        self.build_model(cfg, device)
        self.build_optimizer(cfg)
        self.build_dataloader(cfg, callbacks)

        num_workers = 0
        if getattr(self, "train_loader", None) is not None:
            num_workers = self.train_loader.num_workers
        elif getattr(self, "unlabeled_dataloader", None) is not None:
            num_workers = self.unlabeled_dataloader.num_workers

        LOGGER.info(
            f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
            f'Using {num_workers} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for {self.epochs} epochs...'
        )

        # ===== pseudo label creator =====
        # 论文版 client-only unlabeled，优先建议 FairPseudoLabel
        if cfg.SSOD.pseudo_label_type == 'FairPseudoLabel':
            self.pseudo_label_creator = FairPseudoLabel(cfg)
        elif cfg.SSOD.pseudo_label_type == 'LabelMatch':
            unlabeled_len = int(self.unlabeled_dataset.__len__() / self.WORLD_SIZE) \
                if self.unlabeled_dataset is not None else 0
            self.pseudo_label_creator = LabelMatch(
                cfg,
                unlabeled_len,
                self.label_num_per_image if self.label_num_per_image is not None else None,
                cls_ratio_gt=self.cls_ratio_gt
            )
        else:
            raise ValueError(f"Unsupported pseudo_label_type: {cfg.SSOD.pseudo_label_type}")

        self.build_ddp_model(cfg, device)
        self.device = device

    def set_env(self, cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks):
        super().set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)

        # role: "server" or "client"
        self.role = getattr(cfg, "role", "client")

        # target may not exist for server
        self.data_dict['target'] = getattr(cfg.Dataset, 'target', None)

        self.target_with_gt = cfg.SSOD.ssod_hyp.with_gt
        self.break_epoch = -1
        self.epoch_adaptor = cfg.SSOD.epoch_adaptor
        self.da_loss_weights = cfg.SSOD.da_loss_weights
        self.cosine_ema = cfg.SSOD.cosine_ema
        self.fixed_accumulate = cfg.SSOD.fixed_accumulate

    def build_optimizer(self, cfg, optinit=True, weight_masks=None, ckpt=None):
        super().build_optimizer(cfg, optinit, weight_masks, ckpt)

        if cfg.SSOD.multi_step_lr:
            milestones = cfg.SSOD.milestones
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            self.scheduler.last_epoch = self.epoch - 1
            print('self scheduler:', milestones)
            self.scaler = amp.GradScaler(enabled=self.cuda)

    def build_model(self, cfg, device):
        check_suffix(cfg.weights, '.pt')
        pretrained = cfg.weights.endswith('.pt')

        if pretrained:
            with torch_distributed_zero_first(self.LOCAL_RANK):
                weights = attempt_download(cfg.weights)
            ckpt = torch.load(weights, map_location=device)

            self.model = Model(cfg or ckpt['model'].yaml).to(device)
            exclude = ['anchor'] if (cfg or cfg.Model.anchors) and not cfg.resume else []
            csd = ckpt['model'].float().state_dict()

            if cfg.prune_finetune:
                dynamic_load(self.model, csd)
                self.model.info()

            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)
            self.model.load_state_dict(csd, strict=False)
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')
        else:
            self.model = Model(cfg).to(device)

        # Freeze
        freeze = [f'model.{x}.' for x in range(cfg.freeze_layer_num)]
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False

        # EMA
        self.ema = ModelEMA(self.model)

        # 对 client-only unlabeled，通常建议 burn_epochs=0
        if self.cfg.hyp.burn_epochs > 0:
            self.semi_ema = None
        else:
            if self.cosine_ema:
                self.semi_ema = CosineEMA(self.ema.ema, decay_start=self.cfg.SSOD.ema_rate, total_epoch=self.epochs)
            else:
                self.semi_ema = SemiSupModelEMA(self.ema.ema, self.cfg.SSOD.ema_rate)

        # Resume
        self.start_epoch = 0
        if pretrained:
            if ckpt['optimizer'] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except Exception:
                    LOGGER.info('pretrain model with different type of optimizer')

            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
                self.ema.updates = ckpt['updates']

            if self.semi_ema and ckpt.get('ema'):
                self.semi_ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)

            self.start_epoch = ckpt['epoch'] + 1
            if cfg.resume:
                assert self.start_epoch > 0, f'{weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
                self.epochs += ckpt['epoch']

            del ckpt, csd

        self.epoch = self.start_epoch
        self.model_type = self.model.model_type

        # extra teachers
        self.extra_teacher_models = []
        self.extra_teacher_class_idxs = []
        if len(self.cfg.SSOD.extra_teachers) > 0 and len(self.cfg.SSOD.extra_teachers_class_names) > 0:
            assert len(self.cfg.SSOD.extra_teachers) == len(self.cfg.SSOD.extra_teachers_class_names)
            for i, extra_teacher_path in enumerate(self.cfg.SSOD.extra_teachers):
                teacher_model = attempt_load(extra_teacher_path, map_location=device)
                self.extra_teacher_models.append(teacher_model)

                if self.RANK in [-1, 0]:
                    print(f'load {i} teacher model and class...')

                teacher_class_idx = {}
                assert len(self.cfg.SSOD.extra_teachers_class_names[i]) > 0

                if self.RANK in [-1, 0]:
                    print("origin name: {} current name: {}".format(teacher_model.names, self.cfg.Dataset.names))

                for na in self.cfg.SSOD.extra_teachers_class_names[i]:
                    origin_idx = -1
                    curr_idx = -1

                    for idx, origin_name in enumerate(teacher_model.names):
                        if na == origin_name:
                            origin_idx = idx
                            break

                    for idx, name in enumerate(self.cfg.Dataset.names):
                        if na == name:
                            curr_idx = idx

                    if len(self.cfg.SSOD.extra_teachers_class_names[i]) == 1:
                        if self.RANK in [-1, 0]:
                            print('single cls change ')
                        origin_idx = 0

                    teacher_class_idx[origin_idx] = curr_idx

                if self.RANK in [-1, 0]:
                    print('class_idx dic: ', teacher_class_idx)

                self.extra_teacher_class_idxs.append(teacher_class_idx)
                assert len(self.extra_teacher_class_idxs) == len(self.extra_teacher_models)
                assert len(self.extra_teacher_models) > 0

    def build_dataloader(self, cfg, callbacks):
        gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(cfg.Dataset.img_size, gs, floor=gs * 2)

        # DP mode
        if self.cuda and self.RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning(
                'DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.'
            )
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.sync_bn and self.cuda and self.RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')

        # init
        self.train_loader = None
        self.dataset = None
        self.unlabeled_dataloader = None
        self.unlabeled_dataset = None
        self.cls_ratio_gt = None
        self.label_num_per_image = None

        if self.role == "server":
            # ===== server: labeled only =====
            self.train_loader, self.dataset = create_dataloader(
                self.data_dict['train'],
                self.imgsz,
                self.batch_size // self.WORLD_SIZE,
                gs,
                self.single_cls,
                hyp=cfg.hyp,
                augment=True,
                cache=cfg.cache,
                rect=cfg.rect,
                rank=self.LOCAL_RANK,
                workers=cfg.Dataset.workers,
                prefix=colorstr('train: '),
                cfg=cfg
            )

            self.cls_ratio_gt = self.dataset.cls_ratio_gt
            self.label_num_per_image = self.dataset.label_num_per_image

            mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max()) \
                if len(self.dataset.labels) > 0 and any(len(l) > 0 for l in self.dataset.labels) else 0

            self.nb = len(self.train_loader)
            assert mlc < self.nc, (
                f'Label class {mlc} exceeds nc={self.nc} in {cfg.Dataset.data_name}. '
                f'Possible class labels are 0-{self.nc - 1}'
            )

        elif self.role == "client":
            # ===== client: unlabeled target only =====
            assert self.data_dict['target'] is not None and str(self.data_dict['target']) != "", \
                "client role requires cfg.Dataset.target"

            self.unlabeled_dataloader, self.unlabeled_dataset = create_target_dataloader(
                self.data_dict['target'],
                self.imgsz,
                self.batch_size // self.WORLD_SIZE,
                gs,
                self.single_cls,
                hyp=cfg.hyp,
                augment=True,
                cache=cfg.cache,
                rect=cfg.rect,
                rank=self.LOCAL_RANK,
                workers=cfg.Dataset.workers,
                cfg=cfg,
                prefix=colorstr('target: ')
            )

            self.nb = len(self.unlabeled_dataloader)

        else:
            raise ValueError(f"Unknown role: {self.role}")

        # ===== val_loader =====
        if self.RANK in [-1, 0]:
            self.val_loader = create_dataloader(
                self.data_dict['val'],
                self.imgsz,
                self.batch_size // self.WORLD_SIZE * 2,
                gs,
                self.single_cls,
                hyp=cfg.hyp,
                cache=None if self.noval else cfg.cache,
                rect=True,
                rank=-1,
                workers=cfg.Dataset.workers,
                pad=0.5,
                prefix=colorstr('val: '),
                cfg=cfg
            )[0]

            # only server checks anchors/labels
            if self.role == "server" and (not cfg.resume):
                if len(self.dataset.labels) > 0 and any(len(l) > 0 for l in self.dataset.labels):
                    labels = np.concatenate(self.dataset.labels, 0)

                    if self.plots:
                        plot_labels(labels, self.names, self.save_dir)

                    if not cfg.noautoanchor:
                        check_anchors(self.dataset, model=self.model, thr=cfg.Loss.anchor_t, imgsz=self.imgsz)

                self.model.half().float()

            callbacks.run('on_pretrain_routine_end')

        self.no_aug_epochs = cfg.hyp.no_aug_epochs
        print(f"[ROLE] {self.role}")
        print(f"[TRAIN] {self.data_dict.get('train', None)}")
        print(f"[TARGET] {self.data_dict.get('target', None)}")
        print(f"[NC] {self.nc}")
        print(f"[NAMES] {self.names}")

    # def build_ddp_model(self, cfg, device):
    #     super().build_ddp_model(cfg, device)
    #     self.compute_un_sup_loss = build_ssod_loss(self.model, cfg)
    #     self.domain_loss = DomainLoss()
    #     self.target_loss = TargetLoss()
    
    # def build_ddp_model(self, cfg, device):
    #     if self.role == "server":
    #         print("[DDP] role=server -> use super().build_ddp_model()", flush=True)
    #         super().build_ddp_model(cfg, device)
    
    #     elif self.role == "client":
    #         print("[DDP] role=client -> skip super().build_ddp_model()", flush=True)
    
    #         if self.cuda and self.RANK != -1:
    #             self.model = DDP(
    #                 self.model,
    #                 device_ids=[self.LOCAL_RANK],
    #                 output_device=self.LOCAL_RANK,
    #                 find_unused_parameters=cfg.find_unused_parameters
    #             )
    
    #         self.model.class_weights = torch.ones(self.nc, device=device)
    #         print(f"[DDP] client class_weights shape = {tuple(self.model.class_weights.shape)}", flush=True)
    
    #     else:
    #         raise ValueError(f"Unknown role: {self.role}")
    
    #     self.compute_un_sup_loss = build_ssod_loss(self.model, cfg)
    #     self.domain_loss = DomainLoss()
    #     self.target_loss = TargetLoss()

    def build_ddp_model(self, cfg, device):
        if self.role == "server":
            print("[DDP] role=server -> use super().build_ddp_model()", flush=True)
            super().build_ddp_model(cfg, device)
    
        elif self.role == "client":
            print("[DDP] role=client -> skip super().build_ddp_model()", flush=True)
    
            if self.cuda and self.RANK != -1:
                self.model = DDP(
                    self.model,
                    device_ids=[self.LOCAL_RANK],
                    output_device=self.LOCAL_RANK,
                    find_unused_parameters=cfg.find_unused_parameters
                )
    
            # client 没有 labeled dataset，但很多后续逻辑默认模型上有 class_weights
            self.model.class_weights = torch.ones(self.nc, device=device)
            print(f"[DDP] client class_weights shape = {tuple(self.model.class_weights.shape)}", flush=True)
    
            # ✅ 关键：client 也补上 supervised compute_loss，供 after_epoch / val.run 使用
            self.compute_loss = build_loss(self.model, cfg)
    
        else:
            raise ValueError(f"Unknown role: {self.role}")
    
        # 两个角色都需要
        if cfg.SSOD.loss_type == "ComputeStudentMatchLoss":
            self.compute_un_sup_loss = build_ssod_loss(self.model, cfg)
        else:
            self.compute_un_sup_loss = build_ssod_loss(self.model, cfg)
    
        self.domain_loss = DomainLoss()
        self.target_loss = TargetLoss()
    
    def update_train_logger(self):
        """
        server: only supervised keys
        client: only unsupervised keys
        """
        self.log_contents = []

        if self.role == "server":
            for (imgs, targets, paths, _) in self.train_loader:
                imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale
                targets = targets.to(self.device)
                with torch.no_grad():
                    with amp.autocast(enabled=self.cuda):
                        pred, sup_feats = self.model(imgs)
                        _, loss_items = self.compute_loss(pred, targets)

                if self.RANK in [-1, 0]:
                    for loss_key in loss_items.keys():
                        self.log_contents.append(loss_key)
                break

        elif self.role == "client":
            if self.RANK in [-1, 0]:
                self.log_contents.extend(['ss_box', 'ss_obj', 'ss_cls', 'tp', 'fp_cls', 'fp_loc', 'pse_num', 'gt_num'])

        LOGGER.info(('\n' + '%10s' * len(self.log_contents)) % tuple(self.log_contents))

    def train_in_epoch(self, callbacks):
        """
        server: always supervised only
        client: always unlabeled only
        """
        if self.role == "server":
            self.train_without_unlabeled(callbacks)
            if self.RANK in [-1, 0]:
                print(f'[SERVER] epoch={self.epoch}')

        elif self.role == "client":
            if self.epoch == self.cfg.hyp.burn_epochs:
                msd = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()
                for k, v in self.ema.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        msd[k] = v

                if self.cosine_ema:
                    self.semi_ema = CosineEMA(
                        self.ema.ema,
                        decay_start=self.cfg.SSOD.ema_rate,
                        total_epoch=self.epochs - self.cfg.hyp.burn_epochs
                    )
                else:
                    self.semi_ema = SemiSupModelEMA(self.ema.ema, self.cfg.SSOD.ema_rate)

            self.train_with_unlabeled(callbacks)

        else:
            raise ValueError(f"Unknown role: {self.role}")

    def _make_pseudo_targets(self, teacher_pred, unlabeled_imgs, unlabeled_M, unlabeled_imgs_ori, unlabeled_gt):
        """
        兼容两种接口：
        1) create_pseudo_label_online(...)
        2) create_pseudo_label_online_with_gt(...)
        """
        plc = self.pseudo_label_creator

        # 优先用真正无 GT 接口
        if hasattr(plc, "create_pseudo_label_online"):
            return plc.create_pseudo_label_online(
                teacher_pred,
                copy.deepcopy(unlabeled_imgs),
                unlabeled_M,
                copy.deepcopy(unlabeled_imgs_ori),
                self.RANK
            )

        # fallback 到旧接口
        elif hasattr(plc, "create_pseudo_label_online_with_gt"):
            return plc.create_pseudo_label_online_with_gt(
                teacher_pred,
                copy.deepcopy(unlabeled_imgs),
                unlabeled_M,
                copy.deepcopy(unlabeled_imgs_ori),
                unlabeled_gt,
                self.RANK
            )

        else:
            raise AttributeError("pseudo_label_creator has neither create_pseudo_label_online nor create_pseudo_label_online_with_gt")

    def train_instance_unlabeled_only(self, unlabeled_imgs, unlabeled_imgs_ori, unlabeled_gt, unlabeled_M, ni, pbar, callbacks):
        invalid_target_shape = True
        unlabeled_targets = torch.zeros(8)

        with amp.autocast(enabled=self.cuda):
            with torch.no_grad():
                if self.model_type in ['yolov5']:
                    (teacher_pred, train_out), teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                else:
                    raise NotImplementedError

        plc = getattr(self, "pseudo_label_creator", None)
        if self.RANK in [-1, 0] and (not hasattr(self, "_dbg_plc_once")):
            self._dbg_plc_once = True
            print(f"[CLIENT][PL] plc id={id(plc)}")

        if len(self.extra_teacher_models) == 0:
            unlabeled_targets, invalid_target_shape = self._make_pseudo_targets(
                teacher_pred,
                unlabeled_imgs,
                unlabeled_M,
                unlabeled_imgs_ori,
                unlabeled_gt
            )
            unlabeled_imgs = unlabeled_imgs.to(self.device)
        else:
            raise NotImplementedError

        if invalid_target_shape:
            return

        total_imgs = torch.cat([unlabeled_imgs], 0)

        with amp.autocast(enabled=self.cuda):
            total_pred, total_feature = self.model(total_imgs)
            un_sup_pred = total_pred
            un_sup_feature = total_feature

            unlabeled_targets_dev = unlabeled_targets.to(self.device, non_blocking=True).float()
            un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(un_sup_pred, unlabeled_targets_dev)

            t_loss = self.target_loss(un_sup_feature)
            if self.cfg.SSOD.with_da_loss:
                un_sup_loss = un_sup_loss + t_loss * self.da_loss_weights

            if self.RANK != -1:
                un_sup_loss *= self.WORLD_SIZE

        loss = un_sup_loss
        self.update_optimizer(loss, ni)

        if self.RANK in [-1, 0]:
            self.meter.update(un_sup_loss_items)

            if self.target_with_gt:
                tp_rate, fp_cls_rate, fp_loc_rate, pse_num, gt_num = check_pseudo_label_with_gt(
                    unlabeled_targets,
                    unlabeled_gt,
                    ignore_thres_low=self.compute_un_sup_loss.ignore_thres_low,
                    ignore_thres_high=self.compute_un_sup_loss.ignore_thres_high,
                    batch_size=self.batch_size // self.WORLD_SIZE
                )
            else:
                tp_rate, fp_loc_rate, pse_num, gt_num = check_pseudo_label(
                    unlabeled_targets,
                    ignore_thres_low=self.compute_un_sup_loss.ignore_thres_low,
                    ignore_thres_high=self.compute_un_sup_loss.ignore_thres_high,
                    batch_size=self.batch_size // self.WORLD_SIZE
                )
                fp_cls_rate = 0

            hit_rate = dict(tp=tp_rate, fp_cls=fp_cls_rate, fp_loc=fp_loc_rate, pse_num=pse_num, gt_num=gt_num)
            self.meter.update(hit_rate)

            mloss_count = len(self.meter.meters.items())
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(
                ('%10s' * 2 + '%10.4g' * (mloss_count + 2)) % (
                    f'{self.epoch}/{self.epochs - 1}',
                    mem,
                    0,
                    unlabeled_imgs.shape[-1],
                    *self.meter.get_avg()
                )
            )

    def train_with_unlabeled(self, callbacks):
        """
        论文版 client-only unlabeled 训练主循环：
        - 只遍历 self.unlabeled_dataloader
        - 不依赖 self.train_loader
        - 不再混入 supervised batch
        """
        assert self.role == "client", "train_with_unlabeled() should only be used in client role"
        assert self.unlabeled_dataloader is not None, "client role requires unlabeled_dataloader"

        self.nb = len(self.unlabeled_dataloader)
        pbar = enumerate(self.unlabeled_dataloader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)

        self.optimizer.zero_grad()

        for i, (target_imgs, target_gt, target_paths, _, target_imgs_ori, target_M) in pbar:
            ni = i + self.nb * self.epoch

            target_imgs = target_imgs.to(self.device, non_blocking=True).float() / 255.0
            target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0

            # expose target paths to external wrapper if needed
            plc = getattr(self, "pseudo_label_creator", None)
            if plc is not None:
                plc._cur_target_paths = target_paths
                plc._unlabeled_paths = target_paths


            if self.RANK in [-1, 0] and (not hasattr(self, "_dbg_paths_once")):
                self._dbg_paths_once = True
                print("[DBG][client-only] target_paths type:", type(target_paths))
                try:
                    print("[DBG][client-only] target_paths len:", len(target_paths), "first:", target_paths[0])
                except Exception as e:
                    print("[DBG][client-only] target_paths cannot index:", e)
                print("[DBG][client-only] plc id:", id(plc))

            self.train_instance_unlabeled_only(
                target_imgs,
                target_imgs_ori,
                target_gt,
                target_M,
                ni,
                pbar,
                callbacks
            )

        self.lr = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()

    def train_without_unlabeled(self, callbacks):
        """
        server supervised only
        """
        assert self.role == "server", "train_without_unlabeled() should only be used in server role"
        assert self.train_loader is not None, "server role requires train_loader"

        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)

        self.optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + self.nb * self.epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0

            with amp.autocast(enabled=self.cuda):
                pred, sup_feats = self.model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))

                if self.RANK != -1:
                    loss *= self.WORLD_SIZE

                loss = loss + 0 * (sup_feats[0].mean() + sup_feats[1].mean() + sup_feats[2].mean())

            self.update_optimizer(loss, ni)

            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count = len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(
                    ('%10s' * 2 + '%10.4g' * (mloss_count + 2)) % (
                        f'{self.epoch}/{self.epochs - 1}',
                        mem,
                        targets.shape[0],
                        imgs.shape[-1],
                        *self.meter.get_avg()
                    )
                )
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)

        self.lr = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()

    def update_optimizer(self, loss, ni):
        self.scaler.scale(loss).backward()

        if self.fixed_accumulate:
            self.accumulate = 1
        else:
            self.accumulate = max(round(64 / self.batch_size), 1)

        if ni <= self.nw:
            xi = [0, self.nw]
            if self.fixed_accumulate:
                self.accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            else:
                self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())

            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(
                    ni, xi,
                    [self.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)]
                )
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.ema.update(self.model)
            if self.semi_ema:
                self.semi_ema.update(self.ema.ema)
            self.last_opt_step = ni

    def after_epoch(self, callbacks, val):
        if self.cfg.SSOD.pseudo_label_type == 'LabelMatch' and self.epoch >= self.cfg.SSOD.dynamic_thres_epoch:
            self.pseudo_label_creator.update_epoch_cls_thr(self.epoch - self.start_epoch)
            self.compute_un_sup_loss.ignore_thres_high = self.pseudo_label_creator.cls_thr_high
            self.compute_un_sup_loss.ignore_thres_low = self.pseudo_label_creator.cls_thr_low

        if self.epoch >= self.cfg.hyp.burn_epochs:
            if self.model_type == 'tal':
                self.compute_un_sup_loss.cur_epoch = self.epoch - self.cfg.hyp.burn_epochs
            if self.cosine_ema and self.semi_ema is not None:
                self.semi_ema.update_decay(self.epoch - self.cfg.hyp.burn_epochs)

        if self.RANK in [-1, 0]:
            callbacks.run('on_train_epoch_end', epoch=self.epoch)
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

            final_epoch = (self.epoch + 1 == self.epochs)

            # ===== choose eval model =====
            # server: always ema
            # client: before burn -> ema, after burn -> semi_ema
            if self.role == "server":
                eval_model = self.ema.ema
            else:
                if (self.epoch >= self.cfg.hyp.burn_epochs) and (self.semi_ema is not None):
                    eval_model = self.semi_ema.ema
                else:
                    eval_model = self.ema.ema

            if not self.noval or final_epoch:
                val_ssod = self.cfg.SSOD.train_domain
                # # self.results, maps, _, cls_thr = val.run(
                # self.results, maps, _,= val.run(
                #     cls_thr=None
                #     self.data_dict,
                #     batch_size=self.batch_size // self.WORLD_SIZE * 2,
                #     imgsz=self.imgsz,
                #     model=eval_model,
                #     conf_thres=self.cfg.val_conf_thres,
                #     single_cls=self.single_cls,
                #     dataloader=self.val_loader,
                #     save_dir=self.save_dir,
                #     plots=False,
                #     callbacks=callbacks,
                #     compute_loss=self.compute_loss,
                #     num_points=self.cfg.Dataset.np,
                #     val_ssod=val_ssod,
                #     val_kp=self.cfg.Dataset.val_kp
                # )
                # self.model.train()
                self.results, maps, _ = val.run(
                self.data_dict,
                batch_size=self.batch_size // self.WORLD_SIZE * 2,
                imgsz=self.imgsz,
                model=eval_model,
                conf_thres=self.cfg.val_conf_thres,
                single_cls=self.single_cls,
                dataloader=self.val_loader,
                save_dir=self.save_dir,
                plots=False,
                callbacks=callbacks,
                compute_loss=self.compute_loss,
                num_points=self.cfg.Dataset.np,
                val_ssod=val_ssod,
                val_kp=self.cfg.Dataset.val_kp
            )
            cls_thr = None
            self.model.train()

            fi = fitness(np.array(self.results).reshape(1, -1))
            if fi > self.best_fitness:
                self.best_fitness = fi

            log_vals = list(self.meter.get_avg())[:3] + list(self.results) + self.lr
            callbacks.run('on_fit_epoch_end', log_vals, self.epoch, self.best_fitness, fi)

            if (not self.nosave) or final_epoch:
                if self.role == "server":
                    save_ema = deepcopy(self.ema.ema).half()
                else:
                    if (self.epoch >= self.cfg.hyp.burn_epochs) and (self.semi_ema is not None):
                        save_ema = deepcopy(self.semi_ema.ema).half()
                    else:
                        save_ema = deepcopy(self.ema.ema).half()

                ckpt = {
                    'epoch': self.epoch,
                    'best_fitness': self.best_fitness,
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': save_ema,
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'wandb_id': None
                }

                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
                    w = self.save_dir / 'weights'
                    torch.save(ckpt, w / f'epoch{self.epoch}.pt')

                del ckpt
                callbacks.run('on_model_save', self.last, self.epoch, final_epoch, self.best_fitness, fi)

    def after_train(self, callbacks, val):
        results = (0, 0, 0, 0, 0, 0, 0)

        if self.RANK in [-1, 0]:
            for f in self.last, self.best:
                if f.exists():
                    strip_optimizer(f)
                    if f is self.best:
                        LOGGER.info(f'\nValidating {f}...')
                        # results, _, _, _ = val.run(
                        results, _, _ = val.run(
                            self.data_dict,
                            batch_size=self.batch_size // self.WORLD_SIZE * 2,
                            imgsz=self.imgsz,
                            model=attempt_load(f, self.device).half(),
                            conf_thres=self.cfg.val_conf_thres,
                            iou_thres=0.65,
                            single_cls=self.single_cls,
                            dataloader=self.val_loader,
                            save_dir=self.save_dir,
                            save_json=False,
                            verbose=True,
                            plots=True,
                            callbacks=callbacks,
                            compute_loss=self.compute_loss,
                            num_points=self.cfg.Dataset.np,
                            val_ssod=self.cfg.SSOD.train_domain,
                            val_kp=self.cfg.Dataset.val_kp
                        )

            callbacks.run('on_train_end', self.last, self.best, self.plots, self.epoch)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        torch.cuda.empty_cache()
        return results

    def split_predict_and_feature(self, total_pred, total_feature, n_img):
        sup_feature = [
            total_feature[0][:n_img, :, :, :],
            total_feature[1][:n_img, :, :, :],
            total_feature[2][:n_img, :, :, :]
        ]
        un_sup_feature = [
            total_feature[0][n_img:, :, :, :],
            total_feature[1][n_img:, :, :, :],
            total_feature[2][n_img:, :, :, :]
        ]

        if self.model_type == 'yolov5':
            sup_pred = [
                total_pred[0][:n_img, :, :, :, :],
                total_pred[1][:n_img, :, :, :, :],
                total_pred[2][:n_img, :, :, :, :]
            ]
            un_sup_pred = [
                total_pred[0][n_img:, :, :, :, :],
                total_pred[1][n_img:, :, :, :, :],
                total_pred[2][n_img:, :, :, :, :]
            ]
        elif self.model_type in ['yolox', 'yoloxkp']:
            sup_pred = [
                total_pred[0][:n_img, :, :],
                total_pred[1][:n_img, :, :],
                total_pred[2][:n_img, :, :]
            ]
            un_sup_pred = [
                total_pred[0][n_img:, :, :],
                total_pred[1][n_img:, :, :],
                total_pred[2][n_img:, :, :]
            ]
        elif self.model_type == 'tal':
            sup_pred = [
                [
                    total_pred[0][0][:n_img, :, :, :],
                    total_pred[0][1][:n_img, :, :, :],
                    total_pred[0][2][:n_img, :, :, :]
                ],
                total_pred[1][:n_img, :, :],
                total_pred[2][:n_img, :, :]
            ]
            un_sup_pred = [
                [
                    total_pred[0][0][n_img:, :, :, :],
                    total_pred[0][1][n_img:, :, :, :],
                    total_pred[0][2][n_img:, :, :, :]
                ],
                total_pred[1][n_img:, :, :],
                total_pred[2][n_img:, :, :]
            ]
        else:
            raise NotImplementedError

        return sup_pred, sup_feature, un_sup_pred, un_sup_feature