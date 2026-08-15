import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*torch\.cuda\.amp\.autocast.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*x.T.*deprecated.*")

import gc
import os
import sys
import time
import re
import csv
import random
import logging
import hashlib
import shutil
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import torchvision

if not hasattr(np, "int"):
    np.int = int

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


# ============================================================
# 0) Debug
# ============================================================
DEBUG_TEST = False

if DEBUG_TEST:
    NUM_CLIENTS = 1
    ROUNDS = 3
    LOCAL_EPOCHS = 1
    SERVER_EPOCHS = 1
    PHASE1_ROUNDS = 2

    CLEAN_CLIENT_RUN_DIRS = False
    FORCE_SERVER_UPDATE_EVERY_ROUND = True
    FORCE_NO_RESUME = False
else:
    NUM_CLIENTS = 0
    ROUNDS = 100
    LOCAL_EPOCHS = 1
    SERVER_EPOCHS = 1
    PHASE1_ROUNDS = 50

    CLEAN_CLIENT_RUN_DIRS = True
    FORCE_SERVER_UPDATE_EVERY_ROUND = False
    FORCE_NO_RESUME = False

FRAC = 1.0
SEED = 42

# Server-only experiment hooks.
SERVER_LR_PHASE1 = 1e-4
SERVER_LR_PHASE2 = 1e-4
SERVER_UPDATE_MODE = "full"
SERVER_GRAD_MAX_BATCHES = None
V2_BOOTSTRAP_ROUND = 0
PROPOSAL_START_ROUND = 1
PROPOSAL_GLOBAL_LR = 0.05
PROPOSAL_BETA = float(os.environ.get("PROPOSAL_BETA", "1.0"))
PROPOSAL_EMA_MU = float(os.environ.get("PROPOSAL_EMA_MU", "0.3"))
PROPOSAL_RATIO_RHO = float(os.environ.get("PROPOSAL_RATIO_RHO", "0.5"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "0"))
DYNAMIC_SERVER_MODE = os.environ.get("DYNAMIC_SERVER_MODE", "gs")
LAMBDA_MIN = float(os.environ.get("LAMBDA_MIN", "0.2"))
LAMBDA_MAX = float(os.environ.get("LAMBDA_MAX", "1.0"))
RESUME_GLOBAL_PT = os.environ.get(
    "RESUME_GLOBAL_PT",
    "/home/jovyan/Documents/ssfod/data/ssfl_bdd_niid_4ds/526runs/526exp_yolov5l_coco_pretrained_4cls_50e2/weights/best.pt",
)
RESUME_START_ROUND = int(os.environ.get("RESUME_START_ROUND", "0"))


# ============================================================
# 1) Path configuration
# ============================================================
FED_PLA_ROOT = "/home/jovyan/Research/FedSemi-Teacher/514Fed_Pla"
EFFICIENT_TEACHER_PATH = "/home/jovyan/Documents/efficientteacher"
FEDAVG_DIR = "/home/jovyan/Research/FedSemi-Teacher/Fed"

sys.path.insert(0, FEDAVG_DIR)
from FedAvg import FedAvg

ET_SERVER_CFG_PATH = "/home/jovyan/Documents/efficientteacher/configs/ssod/custom/526server_4cls.yaml"
ET_CLIENT_CFG_PATH = "/home/jovyan/Research/FedSemi-Teacher/514Fed_Pla/100personal_train/ratio0p1/526client_raft_window5_4cls.yaml"

INITIAL_WEIGHTS_PATH = os.environ.get(
    "INIT_WEIGHTS_PATH",
    "/home/jovyan/Documents/ssfod/data/ssfl_bdd_niid_4ds/526runs/526exp_yolov5l_coco_pretrained_4cls_50e2/weights/best.pt",
)

def _infer_init_tag(weights_path: str) -> str:
    p = str(weights_path)
    base = Path(weights_path).name
    if base == "yolov5l.pt":
        return "init_yolov5l_pt"
    if "526exp_yolov5l_coco_pretrained_4cls_50e2" in p:
        return "init_bdd526_best"
    if "522exp_yolov5l_coco_pretrained_50e2" in p:
        return "init_bdd522_best"
    stem = Path(weights_path).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return f"init_{safe or 'custom'}"

INIT_TAG = _infer_init_tag(INITIAL_WEIGHTS_PATH)

# ---------------- server: Didi urban_standard，have labels ----------------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
RUN_NAME = SCRIPT_PATH.stem
DIDI_SPLIT_ROOT = Path("/home/jovyan/Datasets/personal_split/Didi_4cls")
DIDI_CLIENT_ROOT = DIDI_SPLIT_ROOT / "clients"
DIDI_SERVER_ROOT = DIDI_SPLIT_ROOT / "server"

def load_all_clients(root: Path):
    clients = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    )
    if not clients:
        raise RuntimeError(f"no client directories found in {root}")
    return clients

SELECTED_CLIENT_DS = load_all_clients(DIDI_CLIENT_ROOT)
if not DEBUG_TEST:
    NUM_CLIENTS = len(SELECTED_CLIENT_DS)

SERVER_TRAIN = str(DIDI_SERVER_ROOT / "images" / "train")
SERVER_VAL   = str(DIDI_SERVER_ROOT / "images" / "val")

CLIENT_TARGET_DIRS = [
    str(DIDI_CLIENT_ROOT / ds / "videos" / "train")
    for ds in SELECTED_CLIENT_DS
]
CLIENT_VAL_DIRS = [
    str(DIDI_CLIENT_ROOT / ds / "images" / "val")
    for ds in SELECTED_CLIENT_DS
]

WEATHER_NAMES = SELECTED_CLIENT_DS


# ============================================================
# 2) output file
# ============================================================
RUNS_FED = SCRIPT_DIR / RUN_NAME
GLOBAL_PT_DIR = RUNS_FED / "global_pt"
SUMMARY_CSV = RUNS_FED / "summary.csv"
TRAIN_LOG = RUNS_FED / "train.log"
V2_GLOBAL_PT_DIR = SCRIPT_DIR.parent.parent / "V2_Tracking" / "global_pt"
TEMP_RUNS_ROOT = Path("/tmp/fedsemi_v3_lambda_dyn_runtime")

# ============================================================
# 3) import EfficientTeacher
# ============================================================
sys.path.insert(0, EFFICIENT_TEACHER_PATH)

from configs import get_cfg
import trainer.ssod_trainer_raft_window5 as ssod_trainer_raft_window5
from utils.self_supervised_utils_raft_window5 import *
from utils.callbacks import Callbacks
from utils.torch_utils import select_device, intersect_dicts
import val as et_val

from models.detector.yolo_ssod import Model
from utils.datasets import create_dataloader
from utils.general import check_img_size, colorstr, check_suffix

class DynamicPromoteFairPseudoLabel(FairPseudoLabel):
    def __init__(self, cfg):
        super().__init__(cfg)
        ssod = cfg.SSOD
        self.ignore_thres_low = float(getattr(ssod, "ignore_thres_low", 0.1))
        self.ignore_thres_high = float(getattr(ssod, "ignore_thres_high", 0.6))
        self.promote_min_conf = float(getattr(ssod, "promote_min_conf", 0.45))
        self.promote_iou = float(getattr(ssod, "promote_iou", 0.60))
        self.promote_required_support = int(getattr(ssod, "promote_required_support", 2))
        self.promote_conf_margin = float(getattr(ssod, "promote_conf_margin", 1e-3))
        self.promote_center_dist_ratio = float(getattr(ssod, "promote_center_dist_ratio", 0.15))
        self.promote_area_ratio_min = float(getattr(ssod, "promote_area_ratio_min", 0.35))
        self.promote_area_ratio_max = float(getattr(ssod, "promote_area_ratio_max", 2.8))
        self.promote_support_iou = float(getattr(ssod, "promote_support_iou", 2.0))
        self.tracking_stats = []

    @staticmethod
    def _empty_tracking_stats():
        return {
            "batches": 0,
            "frames": 0,
            "center_raw": 0,
            "center_high_before": 0,
            "center_promote_candidates": 0,
            "center_promoted": 0,
            "center_high_after": 0,
            "support_patch": 0,
            "tracking_patch_raw": 0,
            "tracking_patch_after_filter": 0,
            "merged_after_nms": 0,
            "final_targets": 0,
            "frames_with_final": 0,
            "neighbors_checked": 0,
            "flow_fail": 0,
        }

    def consume_tracking_stats(self):
        merged = self._empty_tracking_stats()
        for item in self.tracking_stats:
            for k in merged:
                merged[k] += int(item.get(k, 0))
        self.tracking_stats = []
        return merged

    def _has_temporal_support(self, cand, patch_det):
        if patch_det is None or len(patch_det) == 0:
            return False

        same_cls = patch_det[:, 5].astype(np.int32) == int(cand[5])
        if not np.any(same_cls):
            return False

        cand_box = cand[:4].astype(np.float32)
        patch_box = patch_det[same_cls, :4].astype(np.float32)

        ious = torchvision.ops.box_iou(
            torch.tensor(cand_box[None, :], dtype=torch.float32),
            torch.tensor(patch_box, dtype=torch.float32),
        ).cpu().numpy()[0]
        if np.any(ious >= self.promote_iou):
            return True

        cand_w = max(float(cand_box[2] - cand_box[0]), 1.0)
        cand_h = max(float(cand_box[3] - cand_box[1]), 1.0)
        cand_cx = 0.5 * float(cand_box[0] + cand_box[2])
        cand_cy = 0.5 * float(cand_box[1] + cand_box[3])
        cand_area = cand_w * cand_h

        patch_cx = 0.5 * (patch_box[:, 0] + patch_box[:, 2])
        patch_cy = 0.5 * (patch_box[:, 1] + patch_box[:, 3])
        patch_w = np.maximum(patch_box[:, 2] - patch_box[:, 0], 1.0)
        patch_h = np.maximum(patch_box[:, 3] - patch_box[:, 1], 1.0)
        patch_area = patch_w * patch_h

        center_dist = np.sqrt((patch_cx - cand_cx) ** 2 + (patch_cy - cand_cy) ** 2)
        norm_dist = center_dist / max(cand_w, cand_h, 1.0)
        area_ratio = patch_area / max(cand_area, 1.0)

        geom_ok = (
            (norm_dist <= self.promote_center_dist_ratio) &
            (area_ratio >= self.promote_area_ratio_min) &
            (area_ratio <= self.promote_area_ratio_max)
        )
        return bool(np.any(geom_ok))

    def _promote_uncertain_center_det(self, center_dets, neighbor_patch_det_by_idx):
        if center_dets is None or len(center_dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        promoted = center_dets.copy().astype(np.float32)
        high_thr = float(self.ignore_thres_high)
        if high_thr <= self.promote_min_conf:
            return promoted

        candidate_mask = (promoted[:, 4] >= self.promote_min_conf) & (promoted[:, 4] < high_thr)
        if not np.any(candidate_mask):
            return promoted

        cand_idxs = np.where(candidate_mask)[0].tolist()
        for idx in cand_idxs:
            cand = promoted[idx]
            support_cnt = 0
            for _, patch_det in neighbor_patch_det_by_idx.items():
                if self._has_temporal_support(cand, patch_det):
                    support_cnt += 1

            if support_cnt >= self.promote_required_support:
                promoted[idx, 4] = min(0.999, max(float(promoted[idx, 4]), high_thr + self.promote_conf_margin))

        return promoted

    def create_pseudo_label_window5(self, out, center_M, window_imgs_ori, window_paths, RANK=-2):
        bsz, win_sz, _, height, width = window_imgs_ori.shape
        center_index = int(getattr(self, "center_index", 2))
        batch_stats = self._empty_tracking_stats()
        batch_stats["batches"] = 1
        out = non_max_suppression_ssod(
            out,
            conf_thres=self.nms_conf_thres,
            iou_thres=self.nms_iou_thres,
            num_points=self.num_points,
            multi_label=self.multi_label,
            labels=[],
        )
        out = [out_tensor.detach() for out_tensor in out]
        target_out_np = output_to_target_ssod(out)
        target_out_np = np.array(target_out_np) if len(target_out_np) > 0 else np.zeros((0, 9), dtype=np.float32)

        target_out_targets_perspective = []
        invalid_target_shape = True

        for b in range(bsz):
            batch_stats["frames"] += 1
            center_global_idx = b * win_sz + center_index
            center_targets = target_out_np[target_out_np[:, 0] == center_global_idx].copy()
            center_targets_nb = center_targets[:, 1:].copy() if len(center_targets) > 0 else np.zeros((0, 8), dtype=np.float32)

            center_img_rgb = window_imgs_ori[b, center_index]
            center_img_bgr = self._tensor_img_to_bgr(center_img_rgb)
            center_dets = self._rows_no_batch_to_dets(center_targets_nb)
            batch_stats["center_raw"] += int(len(center_dets))
            if len(center_dets) > 0:
                center_conf = center_dets[:, 4]
                high_thr = float(self.ignore_thres_high)
                batch_stats["center_high_before"] += int(np.sum(center_conf >= high_thr))
                batch_stats["center_promote_candidates"] += int(
                    np.sum((center_conf >= self.promote_min_conf) & (center_conf < high_thr))
                )

            patch_items = []
            neighbor_patch_det_by_idx = {}
            for j in range(win_sz):
                if j == center_index:
                    continue
                if abs(j - center_index) > self.window5_neighbor_radius:
                    continue
                batch_stats["neighbors_checked"] += 1
                neighbor_global_idx = b * win_sz + j
                neighbor_targets = target_out_np[target_out_np[:, 0] == neighbor_global_idx].copy()
                neighbor_targets_nb = neighbor_targets[:, 1:].copy() if len(neighbor_targets) > 0 else np.zeros((0, 8), dtype=np.float32)
                if len(neighbor_targets_nb) == 0:
                    neighbor_patch_det_by_idx[j] = np.zeros((0, 6), dtype=np.float32)
                    continue

                neighbor_img_rgb = window_imgs_ori[b, j]
                neighbor_img_bgr = self._tensor_img_to_bgr(neighbor_img_rgb)

                try:
                    flow = compute_flow(None, neighbor_img_bgr, center_img_bgr, None)
                    neighbor_dets = self._rows_no_batch_to_dets(neighbor_targets_nb)
                    # For promote support, collect broader temporal candidates around the center frame.
                    support_patch_boxes = find_raft_patches(
                        neighbor_dets,
                        np.zeros((0, 6), dtype=np.float32),
                        flow,
                        self.names,
                        iou_thresh=self.promote_support_iou,
                        min_track_conf=self.min_track_conf,
                    )
                    support_patch_boxes = sanitize_patch_items(
                        support_patch_boxes,
                        center_img_bgr.shape[1],
                        center_img_bgr.shape[0],
                        self.min_visible_ratio,
                    )
                    support_rows = self._patch_items_to_rows(support_patch_boxes, row_dim=8)
                    neighbor_patch_det_by_idx[j] = self._patch_rows_to_det(support_rows)
                    batch_stats["support_patch"] += int(len(support_patch_boxes))

                    new_patch_boxes = find_raft_patches(
                        neighbor_dets,
                        center_dets,
                        flow,
                        self.names,
                        iou_thresh=self.patch_iou,
                        min_track_conf=self.min_track_conf,
                    )
                    new_patch_boxes = sanitize_patch_items(
                        new_patch_boxes,
                        center_img_bgr.shape[1],
                        center_img_bgr.shape[0],
                        self.min_visible_ratio,
                    )
                    patch_items.extend(new_patch_boxes)
                    batch_stats["tracking_patch_raw"] += int(len(new_patch_boxes))
                except Exception:
                    neighbor_patch_det_by_idx[j] = np.zeros((0, 6), dtype=np.float32)
                    batch_stats["flow_fail"] += 1
                    if RANK in [-1, 0]:
                        pass

            promoted_center_dets = self._promote_uncertain_center_det(center_dets, neighbor_patch_det_by_idx)
            if len(center_dets) > 0 and len(promoted_center_dets) == len(center_dets):
                batch_stats["center_promoted"] += int(
                    np.sum(promoted_center_dets[:, 4] > center_dets[:, 4] + 1e-8)
                )
                batch_stats["center_high_after"] += int(np.sum(promoted_center_dets[:, 4] >= float(self.ignore_thres_high)))

            patch_rows = self._patch_items_to_rows(patch_items, row_dim=8)
            patch_det = self._patch_rows_to_det(patch_rows)
            patch_det = self._filter_window5_patch_det(patch_det, promoted_center_dets)
            batch_stats["tracking_patch_after_filter"] += int(len(patch_det))

            if len(promoted_center_dets) > 0 and len(patch_det) > 0:
                merged_det = np.concatenate([promoted_center_dets, patch_det], axis=0)
            elif len(promoted_center_dets) > 0:
                merged_det = promoted_center_dets
            else:
                merged_det = patch_det
            merged_det = self._per_class_nms(merged_det, iou_thres=self.window5_merge_nms_iou)
            batch_stats["merged_after_nms"] += int(len(merged_det))
            merged_targets_nb = self._dets_to_xyxy_rows(merged_det, row_dim=8)

            if len(merged_targets_nb) == 0:
                continue

            M_select = center_M[center_M[:, 0] == b, :]
            if M_select.shape[0] == 0:
                continue

            M = M_select[0][1:10].reshape([3, 3]).cpu().numpy()
            s = float(M_select[0][10])
            ud = int(M_select[0][11])
            lr = int(M_select[0][12])

            merged_xyxy = merged_targets_nb.copy()
            _, image_targets_random = online_label_transform(
                center_img_rgb,
                copy.deepcopy(merged_xyxy),
                M,
                s
            )

            image_targets_after = np.array(image_targets_random, dtype=np.float32)
            if image_targets_after.shape[0] != 0:
                image_targets_after = np.concatenate(
                    (np.ones([image_targets_after.shape[0], 1], dtype=np.float32) * b, image_targets_after), 1
                )
                image_targets_after[:, 2:6] = xyxy2xywh(image_targets_after[:, 2:6])
                image_targets_after[:, [3, 5]] /= height
                image_targets_after[:, [2, 4]] /= width
                if ud == 1:
                    image_targets_after[:, 3] = 1 - image_targets_after[:, 3]
                if lr == 1:
                    image_targets_after[:, 2] = 1 - image_targets_after[:, 2]
                target_out_targets_perspective.extend(image_targets_after.tolist())
                batch_stats["final_targets"] += int(image_targets_after.shape[0])
                batch_stats["frames_with_final"] += 1

        self.tracking_stats.append(batch_stats)

        if len(target_out_targets_perspective) > 0:
            target_out_targets_perspective = torch.from_numpy(
                np.array(target_out_targets_perspective, dtype=np.float32)
            )
            invalid_target_shape = False
        else:
            target_out_targets_perspective = torch.zeros((0, 9), dtype=torch.float32)

        return target_out_targets_perspective, invalid_target_shape

ssod_trainer_raft_window5.FairPseudoLabel = DynamicPromoteFairPseudoLabel
SSODTrainer = ssod_trainer_raft_window5.SSODTrainer


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


# ============================================================
# 4) Utility functions
# ============================================================
def dbg(msg):
    print(f"[DBG] {msg}", flush=True)

def dbg_cfg_brief(cfg, tag=""):
    dbg(f"{tag} role={getattr(cfg, 'role', 'N/A')}")
    dbg(f"{tag} weights={cfg.weights}")
    dbg(f"{tag} epochs={cfg.epochs} burn_epochs={cfg.hyp.burn_epochs}")
    dbg(f"{tag} train={cfg.Dataset.train}")
    dbg(f"{tag} target={cfg.Dataset.target}")
    dbg(f"{tag} val={cfg.Dataset.val}")
    dbg(f"{tag} nc={cfg.Dataset.nc}")
    dbg(f"{tag} names={cfg.Dataset.names}")
    dbg(f"{tag} train_domain={cfg.SSOD.train_domain}")
    dbg(f"{tag} epoch_adaptor={cfg.SSOD.epoch_adaptor}")
    dbg(f"{tag} with_gt={cfg.SSOD.ssod_hyp.with_gt}")

def assert_same_keys_and_shapes(state_dicts, tag=""):
    if len(state_dicts) <= 1:
        return
    base = state_dicts[0]
    base_set = set(base.keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        ks = set(sd.keys())
        if ks != base_set:
            miss = sorted(list(base_set - ks))[:20]
            extra = sorted(list(ks - base_set))[:20]
            raise RuntimeError(f"[KEY MISMATCH]{tag} idx={i} miss={miss} extra={extra}")

        for k in base.keys():
            if torch.is_tensor(base[k]) and torch.is_tensor(sd[k]):
                if tuple(base[k].shape) != tuple(sd[k].shape):
                    raise RuntimeError(
                        f"[SHAPE MISMATCH]{tag} idx={i} key={k} "
                        f"base={tuple(base[k].shape)} cur={tuple(sd[k].shape)}"
                    )

def sd_fingerprint(sd: dict, tag: str = ""):
    if not isinstance(sd, dict) or len(sd) == 0:
        print(f"[FPR][{tag}] invalid/empty")
        return
    float_keys = sorted([k for k, v in sd.items()
                         if torch.is_tensor(v) and v.dtype.is_floating_point])
    if not float_keys:
        return
    idxs = [0, len(float_keys)//3, (2*len(float_keys))//3, -1]
    print(f"[FPR][{tag}] total={len(sd)} float={len(float_keys)}")
    for j in idxs:
        k = float_keys[j]
        t = sd[k].detach().cpu().float()
        h = hashlib.sha1(t.numpy().tobytes()).hexdigest()[:10]
        print(f"  {k} | mean={t.mean().item():.6g} sha1={h}")

def load_state_strict(module, sd: dict, tag: str):
    try:
        module.load_state_dict(sd, strict=True)
    except Exception:
        print(f"[LOAD FAIL][{tag}]")
        raise

class ValModelWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *args, **kwargs):
        out = self.m(*args, **kwargs)
        return out[0] if isinstance(out, (tuple, list)) else out

    def __getattr__(self, name):
        if name in (
            "m", "_modules", "_parameters", "_buffers",
            "_backward_hooks", "_forward_hooks", "_forward_pre_hooks",
            "_state_dict_hooks", "_load_state_dict_pre_hooks"
        ):
            return super().__getattr__(name)
        return getattr(self.m, name)

_val_run_orig = et_val.run
def _val_run_patched(*args, **kwargs):
    m = kwargs.get("model", None)
    if m is not None and not isinstance(m, ValModelWrapper):
        kwargs["model"] = ValModelWrapper(m)
    return _val_run_orig(*args, **kwargs)
et_val.run = _val_run_patched

def set_client_train_stage(model, round_idx, phase1_rounds=50):
    backbone_only = round_idx <= phase1_rounds
    for name, p in model.named_parameters():
        p.requires_grad = (not backbone_only) or is_backbone_key(name)

    stage = "Backbone-only local training" if backbone_only else "Full-network local training"

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STAGE][r{round_idx:03d}] {stage} trainable={trainable}/{total} ({trainable/total:.2%})")

def keep_only_weights(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return
    for p in run_dir.iterdir():
        if p.name == "weights":
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except Exception:
                pass

def count_lines(path: str) -> int:
    p = Path(path)
    if p.is_dir():
        image_n = len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
        if image_n > 0:
            return image_n
        subdir_n = len([x for x in p.iterdir() if x.is_dir()])
        if subdir_n > 0:
            return subdir_n
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def labels_dir_for_images_dir(path: str) -> Path:
    img_dir = Path(path)
    if img_dir.parent.name != "images":
        raise ValueError(f"[DATASET CHECK] Expected images/<split> structure, but received: {img_dir}")
    return img_dir.parent.parent / "labels" / img_dir.name

def cache_path_for_images_dir(path: str) -> Path:
    return labels_dir_for_images_dir(path).with_suffix(".cache")

def sanitize_labels_for_images_dir(images_dir: str, split_tag: str, nc: int = 5):
    label_dir = labels_dir_for_images_dir(images_dir)
    allowed = set(range(int(nc)))
    changed_files = 0
    removed_labels = 0

    for lp in sorted(label_dir.glob("*.txt")):
        kept = []
        changed = False
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except Exception:
                changed = True
                removed_labels += 1
                continue
            if cls_id not in allowed:
                changed = True
                removed_labels += 1
                continue
            kept.append(" ".join(parts))
        if changed:
            changed_files += 1
            payload = "\n".join(kept)
            if payload:
                payload += "\n"
            lp.write_text(payload)

    cache_path = cache_path_for_images_dir(images_dir)
    if changed_files and cache_path.exists():
        try:
            cache_path.unlink()
        except Exception:
            pass

    if changed_files:
        print(
            f"[LABEL-SANITIZE][{split_tag}] removed_labels={removed_labels} "
            f"changed_files={changed_files} allowed_classes=0..{nc - 1}"
        )

def append_summary(round_idx, cid, role, res_tuple):
    P, R, mAP50, mAP50_95, *_ = res_tuple
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.writer(f).writerow([round_idx, role, cid, P, R, mAP50, mAP50_95])

def append_summary_scalar(round_idx, role, value):
    with open(SUMMARY_CSV, "a", newline="") as f:
        csv.writer(f).writerow([round_idx, role, "avg", "", "", value, ""])

TRACKING_STATS_FIELDS = [
    "round", "client_id", "weather",
    "batches", "frames",
    "center_raw", "center_high_before", "center_promote_candidates",
    "center_promoted", "center_high_after",
    "support_patch", "tracking_patch_raw", "tracking_patch_after_filter",
    "merged_after_nms", "final_targets", "frames_with_final",
    "neighbors_checked", "flow_fail",
]

def append_tracking_stats(round_idx, cid, weather, stats):
    print(
        f"[TRACKING][r{round_idx:03d}][c{cid:02d}({weather})] "
        f"batches={int(stats.get('batches', 0))} "
        f"frames={int(stats.get('frames', 0))} "
        f"center={int(stats.get('center_raw', 0))} "
        f"promoted={int(stats.get('center_promoted', 0))} "
        f"patch={int(stats.get('tracking_patch_after_filter', 0))} "
        f"final={int(stats.get('final_targets', 0))} "
        f"flow_fail={int(stats.get('flow_fail', 0))}"
    )

def prune_csv_after_round(path, keep_to_round):
    p = Path(path)
    if not p.exists():
        return
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, body = rows[0], rows[1:]
    kept = [header]
    removed = 0
    for row in body:
        if not row:
            continue
        try:
            row_round = int(row[0])
        except Exception:
            kept.append(row)
            continue
        if row_round <= keep_to_round:
            kept.append(row)
        else:
            removed += 1
    if removed:
        with open(p, "w", newline="") as f:
            csv.writer(f).writerows(kept)
        print(f"[RESUME-CLEAN] {p.name} removed_rows={removed} keep_to_round={keep_to_round}")


def load_best_global_client_avg(path, keep_to_round):
    p = Path(path)
    if not p.exists():
        return float("-inf"), keep_to_round

    best_map50 = float("-inf")
    best_round = keep_to_round
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_round = int(row["round"])
            except (TypeError, ValueError, KeyError):
                continue
            if row_round > keep_to_round or row.get("role") != "global_client_avg":
                continue
            try:
                map50 = float(row["mAP50"])
            except (TypeError, ValueError, KeyError):
                continue
            if map50 > best_map50 + 1e-12:
                best_map50 = map50
                best_round = row_round
    return best_map50, best_round

def cleanup_stale_run_dirs():
    removed = 0
    for p in RUNS_FED.iterdir():
        if not p.is_dir():
            continue
        if p.name == "global_pt":
            continue
        if p.name.startswith("raft_window5_r") or p.name.startswith("raft_window5_server_r"):
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    if removed:
        print(f"[STARTUP-CLEAN] removed stale run dirs = {removed}")

def cleanup_output_root():
    RUNS_FED.mkdir(parents=True, exist_ok=True)
    removed = 0
    keep_files = {"summary.csv", "train.log"}
    keep_suffixes = {".py", ".txt", ".md"}
    keep_dirs = {"global_pt"}
    for p in RUNS_FED.iterdir():
        if p.is_dir():
            if p.name in keep_dirs:
                continue
            if p.name.startswith("raft_window5_r") or p.name.startswith("raft_window5_server_r"):
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
            continue
        if p.name in keep_files or p.suffix in keep_suffixes:
            continue
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    if removed:
        print(f"[STARTUP-CLEAN] removed stale output entries = {removed}")

def _clean_sd(sd: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in sd.items()}


def _merge_back(base: dict, updated: dict):
    out = OrderedDict()
    base = _clean_sd(base)
    updated = _clean_sd(updated)
    for k, v in base.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    for k, v in updated.items():
        if k in out:
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out

def is_backbone_key(k: str) -> bool:
    k = k.replace("module.", "")
    return k.startswith("backbone.") or ("backbone" in k)


def use_backbone_only(round_idx: int, phase1_rounds: int = PHASE1_ROUNDS) -> bool:
    return round_idx <= phase1_rounds

def _merge_backbone_only(base: dict, updated: dict):
    out = OrderedDict(
        (k, v.detach().cpu() if torch.is_tensor(v) else v)
        for k, v in _clean_sd(base).items()
    )
    for k, v in _clean_sd(updated).items():
        if k in out and is_backbone_key(k):
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


# ============================================================
# 5) Initialize the global model
# ============================================================
def _init_global_from_best_pt():
    print(f"[INIT] best.pt inital: {INITIAL_WEIGHTS_PATH}")
    check_suffix(str(INITIAL_WEIGHTS_PATH), ['.pt'])
    ckpt = torch.load(str(INITIAL_WEIGHTS_PATH), map_location="cpu")

    cfg0 = get_cfg()
    cfg0.merge_from_file(ET_SERVER_CFG_PATH)
    cfg0.defrost()
    cfg0.role = "server"
    cfg0.Dataset.nc = 4
    cfg0.Dataset.names = ['person', 'car', 'truck', 'bus']
    cfg0.weights = INITIAL_WEIGHTS_PATH
    cfg0.resume = False
    cfg0.freeze()

    m0 = Model(cfg0).to("cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        csd = ckpt["model"].float().state_dict()
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        csd = ckpt["model"]
    elif isinstance(ckpt, dict):
        csd = ckpt
    else:
        csd = ckpt

    csd = {k.replace("module.", ""): v for k, v in csd.items()}
    csd = intersect_dicts(csd, m0.state_dict(), exclude=['anchor'])
    m0.load_state_dict(csd, strict=False)

    clean = OrderedDict(
        (k.replace("module.", ""), v.detach().cpu() if torch.is_tensor(v) else v)
        for k, v in m0.state_dict().items()
    )
    del m0, ckpt
    return clean

def _resume_last_round(global_dir: Path):
    global_dir = Path(global_dir)
    ckpts = sorted(global_dir.glob("fed_global_r*.pt"))
    if ckpts:
        last = ckpts[-1]
        m = re.search(r"fed_global_r(\d+)\.pt", last.name)
        r_last = int(m.group(1)) if m else 0
        sd = torch.load(str(last), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        clean = OrderedDict(
            (k.replace("module.", ""), v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in sd.items()
        )
        print(f"[RESUME] Resumed from {last.name}; starting from round {r_last+1}")
        return r_last, clean

    clean = _init_global_from_best_pt()
    return 0, clean


def _resume_last_round_or_explicit(global_dir: Path, ckpt_path: str, start_round: int):
    global_dir = Path(global_dir)
    ckpts = sorted(global_dir.glob("fed_global_r*.pt"))
    if ckpts:
        return _resume_last_round(global_dir)
    print(
        f"[RESUME] No local round checkpoint found under {global_dir}; "
        f"falling back to the explicitly specified r{start_round:03d} checkpoint"
    )
    return _load_explicit_round_ckpt(ckpt_path, start_round)


def _load_explicit_round_ckpt(ckpt_path: str, start_round: int):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"[RESUME ERROR] checkpoint not found: {ckpt}")

    # For a fresh r000 launch, the explicit checkpoint may be a pretrained
    # detector checkpoint instead of a federated global checkpoint. In that
    # case we need the same model-architecture adaptation path as best.pt init.
    if start_round == 0 and ckpt.resolve() == Path(INITIAL_WEIGHTS_PATH).resolve():
        clean = _init_global_from_best_pt()
        print(f"[RESUME] Explicitly initialized the federated global model from {ckpt}; starting from round {start_round + 1}")
        return start_round, clean

    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    clean = OrderedDict(
        (k.replace("module.", ""), v.detach().cpu() if torch.is_tensor(v) else v)
        for k, v in sd.items()
    )
    print(f"[RESUME] Explicitly resumed from {ckpt}; starting from round {start_round + 1}")
    return start_round, clean


def _state_delta(base_state: dict, updated_state: dict):
    base = _clean_sd(base_state)
    updated = _clean_sd(updated_state)
    delta = OrderedDict()
    for k, v in base.items():
        u = updated.get(k, v)
        if torch.is_tensor(v) and torch.is_tensor(u) and torch.is_floating_point(v) and torch.is_floating_point(u):
            delta[k] = (u.detach().cpu() - v.detach().cpu())
        else:
            delta[k] = u.detach().cpu() if torch.is_tensor(u) else u
    return delta


def _scaled_negative_grad_signal(server_grad: dict):
    client_cfg = get_cfg()
    client_cfg.merge_from_file(ET_CLIENT_CFG_PATH)
    local_scale = float(LOCAL_EPOCHS) * float(client_cfg.hyp.lr0)
    signal = OrderedDict()
    for k, v in _clean_sd(server_grad).items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            signal[k] = (-local_scale * v.detach().cpu())
        else:
            signal[k] = v
    return signal


# ============================================================
# 6) Build configuration
# ============================================================
def build_cfg_for_client(base_device, cid, round_idx, target_dir, val_dir):
    cfg = get_cfg()
    cfg.merge_from_file(ET_CLIENT_CFG_PATH)
    cfg.defrost()

    cfg.role = "client"
    cfg.epochs = int(LOCAL_EPOCHS)
    cfg.hyp.burn_epochs = 0
    cfg.hyp.warmup_epochs = 0

    call_id = f"{int(time.time()*1000)%1_000_000_000:09d}"
    cfg.project = str(TEMP_RUNS_ROOT)
    cfg.name = (
        f"raft_window5_r{round_idx:03d}_c{cid:02d}_TEST_call{call_id}"
        if DEBUG_TEST else
        f"raft_window5_r{round_idx:03d}_c{cid:02d}_{WEATHER_NAMES[cid-1]}_call{call_id}"
    )
    cfg.save_dir = str(Path(cfg.project) / cfg.name)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    cfg.exist_ok = False
    cfg.resume = False

    # Client: no labels; use target data only
    cfg.Dataset.train = ""
    cfg.Dataset.target = target_dir
    cfg.Dataset.val = val_dir

    cfg.Dataset.train_label = None
    cfg.Dataset.train_unlabel = None
    cfg.Dataset.train_both = False

    cfg.SSOD.train_domain = False
    cfg.SSOD.epoch_adaptor = False
    cfg.SSOD.with_da_loss = False
    cfg.SSOD.pseudo_label_type = "FairPseudoLabel"
    cfg.SSOD.ssod_hyp.with_gt = False
    cfg.SSOD.center_index = 1
    cfg.SSOD.window_radius = 1
    cfg.SSOD.window5_neighbor_radius = 1
    cfg.SSOD.ignore_thres_low = 0.40
    raw_dynamic_high = 0.40 + (0.60 - 0.40) * ((round_idx - 1) / max(ROUNDS - 1, 1))
    cfg.SSOD.promote_min_conf = 0.30
    cfg.SSOD.promote_iou = 0.50
    cfg.SSOD.promote_required_support = 2
    cfg.SSOD.promote_conf_margin = 1e-3
    cfg.SSOD.ignore_thres_high = raw_dynamic_high

    cfg.Dataset.nc = 4
    cfg.Dataset.names = ['person', 'car', 'truck', 'bus']

    if not cfg.device:
        cfg.device = str(base_device)

    cfg.weights = INITIAL_WEIGHTS_PATH

    cfg.freeze()
    dbg(
        f"[CLIENT THR] r{round_idx:03d} c{cid:02d} "
        f"low={cfg.SSOD.ignore_thres_low:.3f} raw_high={raw_dynamic_high:.3f} "
        f"high={cfg.SSOD.ignore_thres_high:.3f} "
        f"promote_min={cfg.SSOD.promote_min_conf:.2f} "
        f"support={cfg.SSOD.promote_required_support}"
    )
    dbg_cfg_brief(cfg, tag=f"CLIENT c{cid:02d}({WEATHER_NAMES[cid-1]})")
    return cfg


def get_server_lr(round_idx: int) -> float:
    return float(SERVER_LR_PHASE1 if round_idx <= PHASE1_ROUNDS else SERVER_LR_PHASE2)


def build_cfg_for_server_update(device, round_idx):
    cfg = get_cfg()
    cfg.merge_from_file(ET_SERVER_CFG_PATH)
    cfg.defrost()

    cfg.role = "server"
    cfg.hyp.lr0 = get_server_lr(round_idx)
    cfg.epochs = int(SERVER_EPOCHS)
    cfg.hyp.burn_epochs = 0

    call_id = f"{int(time.time()*1000)%1_000_000_000:09d}"
    cfg.project = str(TEMP_RUNS_ROOT)
    cfg.name = (
        f"raft_window5_server_r{round_idx:03d}_TEST_call{call_id}"
        if DEBUG_TEST else
        f"raft_window5_server_r{round_idx:03d}_call{call_id}"
    )
    cfg.save_dir = str(Path(cfg.project) / cfg.name)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    cfg.exist_ok = False
    cfg.resume = False

    cfg.Dataset.train = SERVER_TRAIN
    cfg.Dataset.val = SERVER_VAL
    cfg.Dataset.target = ""

    cfg.Dataset.train_label = None
    cfg.Dataset.train_unlabel = None
    cfg.Dataset.train_both = False

    cfg.SSOD.train_domain = False
    cfg.SSOD.epoch_adaptor = False
    cfg.SSOD.with_da_loss = False
    cfg.SSOD.pseudo_label_type = "FairPseudoLabel"
    cfg.SSOD.ssod_hyp.with_gt = True

    cfg.Dataset.nc = 4
    cfg.Dataset.names = ['person', 'car', 'truck', 'bus']

    if not cfg.device:
        cfg.device = str(device)

    cfg.weights = INITIAL_WEIGHTS_PATH

    cfg.freeze()
    dbg_cfg_brief(cfg, tag=f"SERVER r{round_idx:03d}")
    return cfg


# ============================================================
# 7) server / eval / sync
# ============================================================
def server_update_one_epoch(device, round_idx, global_state):
    dbg(f"[SERVER-UPDATE] round={round_idx} start")
    cfg_s = build_cfg_for_server_update(device, round_idx)
    callbacks = Callbacks()
    trainer = SSODTrainer(cfg_s, device, callbacks, -1, -1, 1)

    dbg(f"[SERVER-UPDATE] trainer role={getattr(trainer, 'role', 'N/A')}")
    dbg(f"[SERVER-UPDATE] train_loader is None? {trainer.train_loader is None}")
    dbg(f"[SERVER-UPDATE] unlabeled_dataloader is None? {trainer.unlabeled_dataloader is None}")

    gs = _clean_sd(global_state)
    ms = _clean_sd(trainer.model.state_dict())
    gs_matched = intersect_dicts(gs, ms, exclude=['anchor'])
    print(f"[SERVER-UPDATE][r{round_idx:03d}] matched={len(gs_matched)}/{len(ms)}")

    trainer.model.load_state_dict(gs_matched, strict=False)
    if getattr(trainer, "ema", None) is not None and getattr(trainer.ema, "ema", None) is not None:
        trainer.ema.ema.load_state_dict(gs_matched, strict=False)
    if getattr(trainer, "semi_ema", None) is not None and getattr(trainer.semi_ema, "ema", None) is not None:
        trainer.semi_ema.ema.load_state_dict(gs_matched, strict=False)

    dbg("[SERVER-UPDATE] begin trainer.train()")
    trainer.train(callbacks, et_val)
    dbg("[SERVER-UPDATE] trainer.train() done")

    new_global = _merge_back(global_state, _clean_sd(trainer.model.state_dict()))
    shutil.rmtree(trainer.save_dir, ignore_errors=True)
    if CLEAN_CLIENT_RUN_DIRS:
        shutil.rmtree(trainer.save_dir, ignore_errors=True)
    del trainer, callbacks
    torch.cuda.empty_cache()
    return new_global


def estimate_server_gradient(device, round_idx, global_state):
    dbg(f"[SERVER-GRAD] round={round_idx} start")
    cfg_s = build_cfg_for_server_update(device, round_idx)
    callbacks = Callbacks()
    trainer = SSODTrainer(cfg_s, device, callbacks, -1, -1, 1)

    gs = _clean_sd(global_state)
    ms = _clean_sd(trainer.model.state_dict())
    gs_matched = intersect_dicts(gs, ms, exclude=['anchor'])
    trainer.model.load_state_dict(gs_matched, strict=False)
    if getattr(trainer, "ema", None) is not None and getattr(trainer.ema, "ema", None) is not None:
        trainer.ema.ema.load_state_dict(gs_matched, strict=False)
    if getattr(trainer, "semi_ema", None) is not None and getattr(trainer.semi_ema, "ema", None) is not None:
        trainer.semi_ema.ema.load_state_dict(gs_matched, strict=False)

    trainer.model.train()
    trainer.optimizer.zero_grad()

    grad_sums = OrderedDict()
    used_batches = 0
    for i, (imgs, targets, paths, _) in enumerate(trainer.train_loader):
        if SERVER_GRAD_MAX_BATCHES is not None and i >= int(SERVER_GRAD_MAX_BATCHES):
            break

        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=getattr(trainer, "cuda", torch.cuda.is_available())):
            pred, sup_feats = trainer.model(imgs)
            loss, _ = trainer.compute_loss(pred, targets)
            if getattr(trainer, "RANK", -1) != -1:
                loss *= trainer.WORLD_SIZE
            loss = loss + 0 * (sup_feats[0].mean() + sup_feats[1].mean() + sup_feats[2].mean())

        trainer.model.zero_grad(set_to_none=True)
        loss.backward()

        for name, param in trainer.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if name not in grad_sums:
                grad_sums[name] = grad.clone()
            else:
                grad_sums[name].add_(grad)

        used_batches += 1

    if used_batches == 0:
        raise RuntimeError(f"[SERVER-GRAD ERROR] round {round_idx} used_batches=0")

    server_grad = OrderedDict(
        (k, (v / float(used_batches)).detach().cpu())
        for k, v in grad_sums.items()
    )
    print(f"[SERVER-GRAD][r{round_idx:03d}] used_batches={used_batches} grads={len(server_grad)}")

    shutil.rmtree(trainer.save_dir, ignore_errors=True)
    del trainer, callbacks
    torch.cuda.empty_cache()
    gc.collect()
    return server_grad


def build_server_signal(device, round_idx, theta_state):
    if DYNAMIC_SERVER_MODE == "server_pt":
        print(
            f"[SERVER-PT][r{round_idx:03d}] "
            "Perform one server update from the current theta_t to construct the server signal"
        )
        srv_state = server_update_one_epoch(device, round_idx, theta_state)
        sd_fingerprint(srv_state, tag=f"srv_signal_state_r{round_idx:03d}")
        return _state_delta(theta_state, srv_state)
    if DYNAMIC_SERVER_MODE == "gs":
        print(
            f"[SERVER-GRAD][r{round_idx:03d}] "
            "Compute one gradient from the current theta_t to construct the server signal"
        )
        server_grad = estimate_server_gradient(device, round_idx, theta_state)
        return _scaled_negative_grad_signal(server_grad)
    raise ValueError(f"unsupported DYNAMIC_SERVER_MODE: {DYNAMIC_SERVER_MODE}")


def _shared_float_keys_for_similarity(base_state, updated_state):
    base = _clean_sd(base_state)
    updated = _clean_sd(updated_state)
    keys = []
    for k, v in base.items():
        u = updated.get(k)
        if not torch.is_tensor(v) or not torch.is_tensor(u):
            continue
        if not (torch.is_floating_point(v) and torch.is_floating_point(u)):
            continue
        if v.shape != u.shape:
            continue
        keys.append(k)
    return keys


def _shared_float_keys_for_similarity_three(base_state, updated_state, signal_state):
    base = _clean_sd(base_state)
    updated = _clean_sd(updated_state)
    signal = _clean_sd(signal_state)
    keys = []
    for k, v in base.items():
        u = updated.get(k)
        s = signal.get(k)
        if not torch.is_tensor(v) or not torch.is_tensor(u) or not torch.is_tensor(s):
            continue
        if not (torch.is_floating_point(v) and torch.is_floating_point(u) and torch.is_floating_point(s)):
            continue
        if v.shape != u.shape or v.shape != s.shape:
            continue
        keys.append(k)
    return keys


def _flatten_signal(signal_state, keys):
    pieces = []
    clean = _clean_sd(signal_state)
    for k in keys:
        v = clean[k].detach().float().cpu().reshape(-1)
        if v.numel():
            pieces.append(v)
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _flatten_delta(base_state, updated_state, keys):
    base = _clean_sd(base_state)
    updated = _clean_sd(updated_state)
    pieces = []
    for k in keys:
        delta = updated[k].detach().float().cpu() - base[k].detach().float().cpu()
        pieces.append(delta.reshape(-1))
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _safe_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return 0.0
    denom = float(vec_a.norm(p=2) * vec_b.norm(p=2))
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(vec_a, vec_b) / denom)


def _compute_lambda_dyn(client_delta_vec, server_signal_vec):
    sim = _safe_cosine_similarity(client_delta_vec, server_signal_vec)
    alpha = (sim + 1.0) / 2.0
    lam = float(LAMBDA_MIN + (1.0 - alpha) * (LAMBDA_MAX - LAMBDA_MIN))
    return sim, alpha, lam


def _safe_l2_norm(vec: torch.Tensor) -> float:
    if vec.numel() == 0:
        return 0.0
    return float(vec.norm(p=2))


def _scale_vector_to_target_norm(vec: torch.Tensor, target_norm: float):
    current_norm = _safe_l2_norm(vec)
    if current_norm <= 0.0:
        return vec.clone(), 1.0, current_norm
    if target_norm <= 0.0:
        return torch.zeros_like(vec), 0.0, current_norm
    if current_norm <= target_norm:
        return vec.clone(), 1.0, current_norm
    scale = float(target_norm) / current_norm
    return vec * scale, scale, current_norm


def proposal_aggregate_per_client(global_state, client_states, client_sizes, server_signal_state, beta, ema_state):
    base = _clean_sd(global_state)
    srv = _clean_sd(server_signal_state)
    total_size = float(sum(client_sizes))
    if total_size <= 0.0:
        raise ValueError("sum(client_sizes) must be positive")

    cleaned_clients = [_clean_sd(state) for state in client_states]
    client_terms = []
    client_agg_vec = None
    server_agg_raw_vec = None
    for idx, (client_state, client_size) in enumerate(zip(cleaned_clients, client_sizes), start=1):
        shared_keys = _shared_float_keys_for_similarity_three(base, client_state, srv)
        client_delta_vec = _flatten_delta(base, client_state, shared_keys)
        server_signal_vec = _flatten_signal(srv, shared_keys)
        sim, alpha_c, lambda_dyn_c = _compute_lambda_dyn(client_delta_vec, server_signal_vec)
        eta_c = float(client_size) / total_size
        delta_norm = _safe_l2_norm(client_delta_vec)
        signal_norm = _safe_l2_norm(server_signal_vec)
        client_weighted_vec = eta_c * client_delta_vec
        server_weighted_vec = eta_c * float(lambda_dyn_c) * server_signal_vec
        raw_server_norm = abs(float(lambda_dyn_c)) * signal_norm
        raw_signal_to_delta = raw_server_norm / max(delta_norm, 1e-12)
        if client_agg_vec is None:
            client_agg_vec = client_weighted_vec.clone()
            server_agg_raw_vec = server_weighted_vec.clone()
        else:
            client_agg_vec = client_agg_vec + client_weighted_vec
            server_agg_raw_vec = server_agg_raw_vec + server_weighted_vec
        client_terms.append({
            "state": client_state,
            "eta_c": eta_c,
            "lambda_dyn_c": float(lambda_dyn_c),
        })
        print(
            f"[DYN-LAMBDA][c{idx:02d}] sim_c={sim:.6f} alpha_c={alpha_c:.6f} "
            f"lambda_dyn_c={lambda_dyn_c:.6f} eta_c={eta_c:.6f} mode={DYNAMIC_SERVER_MODE}"
        )
        print(
            f"[DYN-NORM][c{idx:02d}] "
            f"||client_delta_c||={delta_norm:.6e} "
            f"||server_signal||={signal_norm:.6e} "
            f"||lambda_dyn_c*server_signal_raw||={raw_server_norm:.6e} "
            f"ratio_raw={raw_signal_to_delta:.6e} shared_keys={len(shared_keys)}"
        )

    ema_prev = float(ema_state.get("client_norm_ema", 0.0))
    client_agg_norm = _safe_l2_norm(client_agg_vec) if client_agg_vec is not None else 0.0
    if ema_prev <= 0.0:
        client_norm_ema = client_agg_norm
    else:
        client_norm_ema = float(PROPOSAL_EMA_MU) * ema_prev + (1.0 - float(PROPOSAL_EMA_MU)) * client_agg_norm
    target_server_norm = float(PROPOSAL_RATIO_RHO) * client_norm_ema
    server_agg_vec, server_clip_scale, server_raw_norm = _scale_vector_to_target_norm(
        server_agg_raw_vec if server_agg_raw_vec is not None else torch.zeros_like(client_agg_vec),
        target_server_norm,
    )
    server_clipped_norm = _safe_l2_norm(server_agg_vec)
    ratio_raw = server_raw_norm / max(client_agg_norm, 1e-12)
    ratio_clipped = server_clipped_norm / max(client_agg_norm, 1e-12)
    ema_gap_ratio = client_norm_ema / max(client_agg_norm, 1e-12)
    client_server_cos = _safe_cosine_similarity(client_agg_vec, server_agg_vec)

    print(
        f"[EMA-RATIO] "
        f"client_agg_norm={client_agg_norm:.6e} "
        f"server_term_raw_norm={server_raw_norm:.6e} "
        f"server_term_clipped_norm={server_clipped_norm:.6e} "
        f"ratio_raw={ratio_raw:.6e} "
        f"ratio_clipped={ratio_clipped:.6e} "
        f"client_norm_ema={client_norm_ema:.6e} "
        f"ema_gap_ratio={ema_gap_ratio:.6e} "
        f"cos_client_server={client_server_cos:.6f} "
        f"target_server_norm={target_server_norm:.6e} "
        f"clip_scale={server_clip_scale:.6e} "
        f"mu={PROPOSAL_EMA_MU:.3f} rho={PROPOSAL_RATIO_RHO:.3f}"
    )

    adjusted = OrderedDict()

    for k, v in base.items():
        v_base = v.detach().cpu() if torch.is_tensor(v) else v
        v_srv = srv.get(k, v_base)

        if not torch.is_tensor(v_base):
            adjusted[k] = v_base
            continue

        if not torch.is_floating_point(v_base):
            adjusted[k] = v_base
            continue

        v_srv = v_srv.detach().cpu() if torch.is_tensor(v_srv) else torch.zeros_like(v_base)
        client_summed = torch.zeros_like(v_base)
        server_summed_raw = torch.zeros_like(v_base)

        for term in client_terms:
            client_state = term["state"]
            eta_c = term["eta_c"]
            lambda_dyn_c = term["lambda_dyn_c"]
            v_client = client_state.get(k, v_base)
            if not torch.is_tensor(v_client):
                v_client = v_base
            else:
                v_client = v_client.detach().cpu()

            client_delta_c = v_client - v_base
            client_summed = client_summed + eta_c * client_delta_c
            server_summed_raw = server_summed_raw + eta_c * float(lambda_dyn_c) * v_srv

        server_summed = server_summed_raw * float(server_clip_scale)
        adjusted[k] = (v_base + float(beta) * (client_summed + server_summed)).detach().cpu()

    if client_agg_vec is not None and server_agg_vec is not None:
        update_delta_vec = float(beta) * (client_agg_vec + server_agg_vec)
        print(
            f"[UPDATE-COS] "
            f"cos(update,client_agg)={_safe_cosine_similarity(update_delta_vec, client_agg_vec):.6f} "
            f"cos(update,server_term)={_safe_cosine_similarity(update_delta_vec, server_agg_vec):.6f} "
            f"||client_agg||={_safe_l2_norm(client_agg_vec):.6e} "
            f"||server_term||={_safe_l2_norm(server_agg_vec):.6e} "
            f"||update||={_safe_l2_norm(update_delta_vec):.6e}"
        )

    ema_state["client_norm_ema"] = client_norm_ema
    ema_state["last_client_agg_norm"] = client_agg_norm
    ema_state["last_server_term_raw_norm"] = server_raw_norm
    ema_state["last_server_term_clipped_norm"] = server_clipped_norm
    ema_state["last_ratio_raw"] = ratio_raw
    ema_state["last_ratio_clipped"] = ratio_clipped
    ema_state["last_ema_gap_ratio"] = ema_gap_ratio
    ema_state["last_clip_scale"] = server_clip_scale

    return adjusted

def global_eval(global_state, device, round_idx, val_path, eval_name):
    dbg(f"[GLOBAL-EVAL] round={round_idx} start eval={eval_name}")
    cfg_eval = get_cfg()
    cfg_eval.merge_from_file(ET_SERVER_CFG_PATH)
    cfg_eval.defrost()

    cfg_eval.role = "server"
    cfg_eval.Dataset.val = val_path
    cfg_eval.Dataset.train = val_path
    cfg_eval.Dataset.target = ""
    cfg_eval.Dataset.nc = 4
    cfg_eval.Dataset.names = ['person', 'car', 'truck', 'bus']
    cfg_eval.resume = False
    cfg_eval.weights = INITIAL_WEIGHTS_PATH
    if not cfg_eval.device:
        cfg_eval.device = str(device)
    cfg_eval.freeze()

    m = Model(cfg_eval).to(device)
    load_state_strict(m, global_state, tag=f"global_eval_{eval_name}_r{round_idx:03d}")

    imgsz = check_img_size(int(cfg_eval.Dataset.img_size), s=32)
    bs = int(cfg_eval.Dataset.batch_size)
    data = {
        "val": cfg_eval.Dataset.val,
        "nc": int(cfg_eval.Dataset.nc),
        "names": cfg_eval.Dataset.names
    }

    val_loader = create_dataloader(
        data["val"], imgsz, bs, 32,
        single_cls=False, pad=0.5, rect=True,
        cfg=cfg_eval, prefix=colorstr("val: ")
    )[0]

    res, _, _ = et_val.run(
        data=data,
        batch_size=bs,
        imgsz=imgsz,
        task="val",
        device=str(device),
        model=ValModelWrapper(m),
        dataloader=val_loader,
        plots=False,
        half=True
    )

    del m, val_loader
    torch.cuda.empty_cache()
    return res


def sync_global_to_trainer(trainer, global_state, cid, r):
    dbg(f"[SYNC] r{r:03d} c{cid:02d} start")
    dbg(f"[SYNC] trainer role={getattr(trainer, 'role', 'N/A')}")
    dbg(f"[SYNC] train_loader is None? {getattr(trainer, 'train_loader', None) is None}")
    dbg(f"[SYNC] unlabeled_dataloader is None? {getattr(trainer, 'unlabeled_dataloader', None) is None}")

    load_state_strict(trainer.model, global_state, tag=f"r{r:03d}_c{cid:02d}_student")
    dbg("[SYNC] model loaded")

    if getattr(trainer, "ema", None) is not None and getattr(trainer.ema, "ema", None) is not None:
        load_state_strict(trainer.ema.ema, global_state, tag=f"r{r:03d}_c{cid:02d}_ema")
        dbg("[SYNC] ema loaded")

    if getattr(trainer, "semi_ema", None) is not None and trainer.semi_ema.ema is not None:
        load_state_strict(trainer.semi_ema.ema, global_state, tag=f"r{r:03d}_c{cid:02d}_semi_ema")
        dbg("[SYNC] semi_ema loaded")

    dbg("[SYNC] done")


def _state_dict_for_fedavg(trainer):
    if getattr(trainer, "semi_ema", None) is not None and getattr(trainer.semi_ema, "ema", None) is not None:
        return {k: v.detach().cpu() for k, v in trainer.semi_ema.ema.state_dict().items()}
    if getattr(trainer, "ema", None) is not None and getattr(trainer.ema, "ema", None) is not None:
        return {k: v.detach().cpu() for k, v in trainer.ema.ema.state_dict().items()}
    return {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}


# ============================================================
# 8) Main program
# ============================================================
if __name__ == "__main__":
    RUNS_FED.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_LOG, "a", buffering=1) as _log_fp:
        sys.stdout = TeeStream(sys.__stdout__, _log_fp)
        sys.stderr = TeeStream(sys.__stderr__, _log_fp)

        for name in ("", "efficientteacher", "trainer", "utils", "models", __name__):
            logging.getLogger(name).handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )

        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["PYTHONHASHSEED"] = str(SEED)
        os.environ["FEDSEMI_GLOBAL_SEED"] = str(SEED)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        for p in ([EFFICIENT_TEACHER_PATH, FEDAVG_DIR,
                   ET_SERVER_CFG_PATH, ET_CLIENT_CFG_PATH,
                   INITIAL_WEIGHTS_PATH, SERVER_TRAIN, SERVER_VAL]
                  + CLIENT_TARGET_DIRS + CLIENT_VAL_DIRS):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[PATH ERROR] Path does not exist: {p}")
        print("[OK] All path checks passed")

        sanitize_labels_for_images_dir(SERVER_TRAIN, "server/train", nc=4)
        sanitize_labels_for_images_dir(SERVER_VAL, "server/val", nc=4)
        for i, val_dir in enumerate(CLIENT_VAL_DIRS, start=1):
            sanitize_labels_for_images_dir(val_dir, f"client{i}/val", nc=4)

        for i, (tdir, vdir, weather) in enumerate(zip(CLIENT_TARGET_DIRS, CLIENT_VAL_DIRS, WEATHER_NAMES), start=1):
            print(f"[DATA] client{i}({weather}) target windows = {count_lines(tdir)}")
            print(f"[DATA] client{i}({weather}) validation images = {count_lines(vdir)}")
        print(f"[DATA] server(cloudy) training images = {count_lines(SERVER_TRAIN)}")
        print(f"[DATA] server(cloudy) validation images = {count_lines(SERVER_VAL)}")

        GLOBAL_PT_DIR.mkdir(parents=True, exist_ok=True)
        cleanup_stale_run_dirs()
        cleanup_output_root()

        if not SUMMARY_CSV.exists():
            with open(SUMMARY_CSV, "w", newline="") as f:
                csv.writer(f).writerow(["round", "role", "client_id", "P", "R", "mAP50", "mAP50_95"])

        cfg_tmp = get_cfg()
        cfg_tmp.merge_from_file(ET_SERVER_CFG_PATH)
        device = select_device(
            "0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu",
            batch_size=cfg_tmp.Dataset.batch_size
        )
        print(f"[DEVICE] {device}")
        print(f"[SETUP] DEBUG_TEST={DEBUG_TEST} NUM_CLIENTS={NUM_CLIENTS} ROUNDS={ROUNDS}")
        print(f"[SETUP] server=ds_0(label) | clients={','.join(WEATHER_NAMES)}(unlabeled)")
        print("[SETUP] BDD personal split | all selected participants are included")
        print(f"[SETUP] target input = window directories | all {ROUNDS} rounds client full-network")
        print("[SETUP] V1 Navigation baseline + tracking/window5 + dynamic 0.4->0.6 threshold + temporal promote")
        print("[SETUP] aggregation = phased FedAvg | rounds 1-50 backbone-only train/upload | rounds 51-100 full-model train/upload | no Soft FedLAG | no global EMA | warmup=0")
        print(
            f"[SETUP] server update = mode={SERVER_UPDATE_MODE} "
            f"lr_phase1={SERVER_LR_PHASE1:.2e} lr_phase2={SERVER_LR_PHASE2:.2e}"
        )
        early_stop_desc = (
            f"early_stop_patience={EARLY_STOP_PATIENCE}"
            if EARLY_STOP_PATIENCE > 0
            else "early_stop=disabled"
        )
        print(
            f"[SETUP] explicit resume from r{RESUME_START_ROUND:03d} | "
            f"ckpt={RESUME_GLOBAL_PT} | "
            f"proposal starts at round {PROPOSAL_START_ROUND} | beta={PROPOSAL_BETA:.2f} | "
            f"mu={PROPOSAL_EMA_MU:.2f} | rho={PROPOSAL_RATIO_RHO:.2f} | "
            f"{early_stop_desc} | "
            f"dynamic lambda mode={DYNAMIC_SERVER_MODE} range=[{LAMBDA_MIN:.2f}, {LAMBDA_MAX:.2f}] | "
            "variant=B(server-last)"
        )

        if FORCE_NO_RESUME:
            dbg("[INIT] FORCE_NO_RESUME=True, explicitly initializing from the specified round checkpoint")
            start_round, global_state = _load_explicit_round_ckpt(
                RESUME_GLOBAL_PT,
                RESUME_START_ROUND,
            )
        else:
            start_round, global_state = _resume_last_round_or_explicit(
                GLOBAL_PT_DIR,
                RESUME_GLOBAL_PT,
                RESUME_START_ROUND,
            )

        prune_csv_after_round(SUMMARY_CSV, start_round)
        sd_fingerprint(global_state, tag=f"global_init_r{start_round:03d}")
        proposal_ema_state = {}
        best_global_client_map50, best_global_round = load_best_global_client_avg(SUMMARY_CSV, start_round)
        if best_global_client_map50 > float("-inf"):
            print(
                f"[RESUME] best global_client_avg before r{start_round + 1:03d}: "
                f"round=r{best_global_round:03d} mAP50={best_global_client_map50:.4f}"
            )
        no_improve_rounds = 0

        # ==================== Main loop ====================
        for r in range(start_round + 1, ROUNDS + 1):
            print(f"\n{'='*60}")
            phase = "client backbone-only" if use_backbone_only(r) else "client full-network"
            upload_mode = "upload backbone only" if use_backbone_only(r) else "upload full model"
            print(f"Round {r}/{ROUNDS}  [{phase} | {upload_mode}]")
            print(f"{'='*60}")

            theta_t_state = global_state

            client_states = []
            client_sizes = []

            client_indexes = list(range(NUM_CLIENTS))

            for idx in client_indexes:
                cid = idx + 1
                weather = WEATHER_NAMES[idx]
                target_dir = CLIENT_TARGET_DIRS[idx]
                val_dir = CLIENT_VAL_DIRS[idx]

                print(f"\n[r{r:03d}][c{cid:02d}({weather})] Start training")

                cfg_c = build_cfg_for_client(device, cid, r, target_dir, val_dir)
                callbacks = Callbacks()
                trainer = None
                try:
                    trainer = SSODTrainer(cfg_c, device, callbacks, -1, -1, 1)

                    dbg(f"[CLIENT BUILD] r{r:03d} c{cid:02d} role={trainer.role}")
                    dbg(f"[CLIENT BUILD] train_loader is None? {trainer.train_loader is None}")
                    dbg(f"[CLIENT BUILD] unlabeled_dataloader is None? {trainer.unlabeled_dataloader is None}")
                    dbg(f"[CLIENT BUILD] nb={trainer.nb}")

                    sync_global_to_trainer(trainer, theta_t_state, cid, r)
                    set_client_train_stage(trainer.model, r, phase1_rounds=PHASE1_ROUNDS)

                    dbg(f"[CLIENT TRAIN] r{r:03d} c{cid:02d} begin")
                    res_cli = trainer.train(callbacks, et_val)
                    dbg(f"[CLIENT TRAIN] r{r:03d} c{cid:02d} done")

                    append_summary(r, cid, f"client_{weather}", res_cli)
                    tracking_stats = trainer.pseudo_label_creator.consume_tracking_stats()
                    append_tracking_stats(r, cid, weather, tracking_stats)
                    print(f"[r{r:03d}][c{cid:02d}({weather})] "
                          f"P={res_cli[0]:.4f} R={res_cli[1]:.4f} "
                          f"mAP50={res_cli[2]:.4f} mAP50_95={res_cli[3]:.4f}")
                    print(
                        f"[TRACK][r{r:03d}][c{cid:02d}] "
                        f"center={tracking_stats['center_raw']} "
                        f"promoted={tracking_stats['center_promoted']} "
                        f"patch={tracking_stats['tracking_patch_after_filter']} "
                        f"final={tracking_stats['final_targets']} "
                        f"flow_fail={tracking_stats['flow_fail']}"
                    )

                    state_to_save = _state_dict_for_fedavg(trainer)
                    client_states.append(state_to_save)
                    client_sizes.append(count_lines(target_dir))

                    sd_fingerprint(state_to_save, tag=f"client_r{r:03d}_c{cid:02d}")
                finally:
                    if CLEAN_CLIENT_RUN_DIRS and trainer is not None:
                        shutil.rmtree(trainer.save_dir, ignore_errors=True)
                    del trainer, callbacks
                    torch.cuda.empty_cache()
                    gc.collect()

            dbg(f"[ROUND {r}] collected client_states={len(client_states)}")

            if not client_states:
                print(f"[Round {r}] No client completed successfully; terminating")
                break

            assert_same_keys_and_shapes(client_states, tag=f"r{r:03d}_before_FedAvg")

            if r < PROPOSAL_START_ROUND:
                dbg(f"[ROUND {r}] FedAvg start")
                avg_state = FedAvg(client_states, client_sizes)
                if use_backbone_only(r):
                    avg_state = _merge_backbone_only(global_state, avg_state)
                dbg(f"[ROUND {r}] FedAvg done")
                sd_fingerprint(avg_state, tag=f"avg_r{r:03d}")
                print(f"[SERVER-UPDATE][r{r:03d}] Update using ds_0(cloudy)")
                srv_state = server_update_one_epoch(device, r, avg_state)
                sd_fingerprint(srv_state, tag=f"srv_state_r{r:03d}")
                global_state = OrderedDict(
                    (k.replace("module.", ""),
                     v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in srv_state.items()
                )
                sd_fingerprint(global_state, tag=f"global_after_server_update_r{r:03d}")
            else:
                print(
                    f"[B][r{r:03d}] Directly use the previous round final global model as theta_t; "
                    "first build the proposal, then perform the labeled server update"
                )
                server_signal_state = build_server_signal(device, r, theta_t_state)
                proposal_state = proposal_aggregate_per_client(
                    theta_t_state,
                    client_states,
                    client_sizes,
                    server_signal_state,
                    beta=PROPOSAL_BETA,
                    ema_state=proposal_ema_state,
                )
                sd_fingerprint(proposal_state, tag=f"proposal_state_r{r:03d}")
                proposal_state = OrderedDict(
                    (k.replace("module.", ""),
                     v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in proposal_state.items()
                )
                sd_fingerprint(proposal_state, tag=f"proposal_state_clean_r{r:03d}")

                srv_state = server_update_one_epoch(device, r, proposal_state)
                sd_fingerprint(srv_state, tag=f"srv_state_after_proposal_r{r:03d}")
                global_state = OrderedDict(
                    (k.replace("module.", ""),
                     v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in srv_state.items()
                )
                sd_fingerprint(global_state, tag=f"global_after_server_update_r{r:03d}")

            ckpt_path = GLOBAL_PT_DIR / f"fed_global_r{r:03d}.pt"
            torch.save(global_state, ckpt_path)
            print(f"[Round {r}] global saved -> {ckpt_path}")

            res_server = global_eval(global_state, device, r, SERVER_VAL, "global_on_server")
            append_summary(r, None, "global_on_server", res_server)
            print(f"[r{r:03d}][global_on_server] "
                  f"P={res_server[0]:.4f} R={res_server[1]:.4f} "
                  f"mAP50={res_server[2]:.4f} mAP50_95={res_server[3]:.4f}")

            global_client_results = []
            for weather, val_dir in zip(WEATHER_NAMES, CLIENT_VAL_DIRS):
                res_glb = global_eval(global_state, device, r, val_dir, f"global_on_{weather}")
                append_summary(r, None, f"global_on_{weather}", res_glb)
                global_client_results.append(res_glb[:4])
                print(f"[r{r:03d}][global_on_{weather}] "
                      f"P={res_glb[0]:.4f} R={res_glb[1]:.4f} "
                      f"mAP50={res_glb[2]:.4f} mAP50_95={res_glb[3]:.4f}")

            avg_res = tuple(
                sum(metrics[i] for metrics in global_client_results) / len(global_client_results)
                for i in range(4)
            )
            append_summary(r, None, "global_client_avg", avg_res)
            print(f"[r{r:03d}][global_client_avg] "
                  f"P={avg_res[0]:.4f} R={avg_res[1]:.4f} "
                  f"mAP50={avg_res[2]:.4f} mAP50_95={avg_res[3]:.4f}")

            current_map50 = float(avg_res[2])
            if current_map50 > best_global_client_map50 + 1e-12:
                best_global_client_map50 = current_map50
                best_global_round = r
                no_improve_rounds = 0
                print(
                    f"[EARLY-STOP] new best global_client_avg mAP50={best_global_client_map50:.4f} "
                    f"at r{best_global_round:03d}"
                )
            else:
                no_improve_rounds += 1
                if EARLY_STOP_PATIENCE > 0:
                    print(
                        f"[EARLY-STOP] no improvement for {no_improve_rounds}/{EARLY_STOP_PATIENCE} rounds | "
                        f"best=r{best_global_round:03d} mAP50={best_global_client_map50:.4f}"
                    )
                    if no_improve_rounds >= EARLY_STOP_PATIENCE:
                        print(
                            f"[EARLY-STOP] triggered at r{r:03d} | "
                            f"best round=r{best_global_round:03d} mAP50={best_global_client_map50:.4f}"
                        )
                        break
                else:
                    print(
                        f"[EARLY-STOP] disabled | no improvement streak={no_improve_rounds} | "
                        f"best=r{best_global_round:03d} mAP50={best_global_client_map50:.4f}"
                    )

        print("\n✅ V4C EMA-ratio dynamic aggregation completed")
        print(f"Results summary: {SUMMARY_CSV}")
