# - server = Didi server (urban_standard), with labels
# - clients = 9 Didi clients, without labels
# - client training does not use GT (target is unlabeled)
# - client target input uses images/train
# - each client is evaluated on its own images/val
# - all selected participants take part
# - 100 rounds: first 50 rounds are backbone-only, last 50 rounds are full-network

import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*torch\.cuda\.amp\.autocast.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*x.T.*deprecated.*")

import os
import sys
import time
import re
import csv
import logging
import hashlib
import shutil
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

if not hasattr(np, "int"):
    np.int = int

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


# ============================================================
# 0) Debug switch
# ============================================================
DEBUG_TEST = False

if DEBUG_TEST:
    NUM_CLIENTS = 1
    ROUNDS = 3
    LOCAL_EPOCHS = 1
    SERVER_EPOCHS = 1
    PHASE1_ROUNDS = 1
    SERVER_WARMUP_EPOCHS = 1

    CLEAN_CLIENT_RUN_DIRS = False
    FORCE_SERVER_UPDATE_EVERY_ROUND = True
    FORCE_NO_RESUME = False
else:
    NUM_CLIENTS = 0
    ROUNDS = 100
    LOCAL_EPOCHS = 1
    SERVER_EPOCHS = 1
    PHASE1_ROUNDS = 50
    SERVER_WARMUP_EPOCHS = 1

    CLEAN_CLIENT_RUN_DIRS = True
    FORCE_SERVER_UPDATE_EVERY_ROUND = False
    FORCE_NO_RESUME = False

FRAC = 1.0
SEED = 42


# ============================================================
# 1) Path configuration
# ============================================================
FED_PLA_ROOT = "/home/jovyan/Research/FedSemi-Teacher/514Fed_Pla"
EFFICIENT_TEACHER_PATH = "/home/jovyan/Documents/efficientteacher"
FEDAVG_DIR = "/home/jovyan/Research/FedSemi-Teacher/Fed"

sys.path.insert(0, FEDAVG_DIR)
from FedAvg import FedAvg

ET_SERVER_CFG_PATH = "/home/jovyan/Documents/efficientteacher/configs/ssod/custom/526server_4cls.yaml"
ET_CLIENT_CFG_PATH = "/home/jovyan/Research/FedSemi-Teacher/514Fed_Pla/100personal_train/ratio0p1/526client_navigation_4cls.yaml"

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

# ---------------- server: Didi urban_standard, with labels ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
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
    str(DIDI_CLIENT_ROOT / ds / "images" / "train")
    for ds in SELECTED_CLIENT_DS
]
CLIENT_VAL_DIRS = [
    str(DIDI_CLIENT_ROOT / ds / "images" / "val")
    for ds in SELECTED_CLIENT_DS
]
WEATHER_NAMES = SELECTED_CLIENT_DS
# ============================================================
# 2) Output directories
# ============================================================
RUNS_FED = SCRIPT_DIR / "V1_Navigation"
GLOBAL_PT_DIR = RUNS_FED / "global_pt"
SUMMARY_CSV = RUNS_FED / "summary.csv"
TRAIN_LOG = RUNS_FED / "train.log"


# ============================================================
# 3) import EfficientTeacher
# ============================================================
sys.path.insert(0, EFFICIENT_TEACHER_PATH)

from configs import get_cfg
from trainer.ssod_trainer import SSODTrainer
from utils.callbacks import Callbacks
from utils.torch_utils import select_device, intersect_dicts
import val as et_val

from models.detector.yolo_ssod import Model
from utils.datasets import create_dataloader
from utils.general import check_img_size, colorstr, check_suffix


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


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

def set_client_train_stage(model, round_idx, phase1_rounds=50):
    for p in model.parameters():
        p.requires_grad = True

    if round_idx <= phase1_rounds:
        for name, p in model.named_parameters():
            if not (name.startswith("backbone.") or "backbone" in name):
                p.requires_grad = False
        stage = "Phase1(backbone-only)"
    else:
        stage = "Phase2(full-network)"

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

def cleanup_output_root():
    RUNS_FED.mkdir(parents=True, exist_ok=True)
    removed = 0
    keep_files = {"summary.csv", "train.log"}
    keep_dirs = {"global_pt"}
    for p in RUNS_FED.iterdir():
        if p.is_dir():
            if p.name in keep_dirs:
                continue
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
            continue
        if p.name in keep_files:
            continue
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    if removed:
        print(f"[STARTUP-CLEAN] removed stale output entries = {removed}")

def count_lines(path: str) -> int:
    p = Path(path)
    if p.is_dir():
        return len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
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

def _is_backbone_key(key: str) -> bool:
    return key.startswith("backbone.") or ".backbone." in key or "backbone" in key

def _merge_backbone_only(base: dict, updated: dict):
    out = OrderedDict()
    base = _clean_sd(base)
    updated = _clean_sd(updated)
    for k, v in base.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    for k, v in updated.items():
        if k in out and _is_backbone_key(k):
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


# ============================================================
# 5) Initialize global model
# ============================================================
def _init_global_from_best_pt():
    print(f"[INIT] Initialize from best.pt: {INITIAL_WEIGHTS_PATH}")
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


# ============================================================
# 6) Build cfg
# ============================================================
def build_cfg_for_client(base_device, cid, round_idx, video_txt, val_dir):
    cfg = get_cfg()
    cfg.merge_from_file(ET_CLIENT_CFG_PATH)
    cfg.defrost()

    cfg.role = "client"
    cfg.epochs = int(LOCAL_EPOCHS)
    cfg.hyp.burn_epochs = 0

    call_id = f"{int(time.time()*1000)%1_000_000_000:09d}"
    cfg.project = str(RUNS_FED)
    cfg.name = (
        f"fed_r{round_idx:03d}_c{cid:02d}_TEST_call{call_id}"
        if DEBUG_TEST else
        f"fed_r{round_idx:03d}_c{cid:02d}_{WEATHER_NAMES[cid-1]}_call{call_id}"
    )
    cfg.save_dir = str(Path(cfg.project) / cfg.name)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    cfg.exist_ok = False
    cfg.resume = False

    # client: no labels, target only
    cfg.Dataset.train = ""
    cfg.Dataset.target = video_txt
    cfg.Dataset.val = val_dir

    cfg.Dataset.train_label = None
    cfg.Dataset.train_unlabel = None
    cfg.Dataset.train_both = False

    cfg.SSOD.train_domain = False
    cfg.SSOD.epoch_adaptor = False
    cfg.SSOD.with_da_loss = False
    cfg.SSOD.pseudo_label_type = "FairPseudoLabel"
    cfg.SSOD.ssod_hyp.with_gt = False

    cfg.Dataset.nc = 4
    cfg.Dataset.names = ['person', 'car', 'truck', 'bus']

    if not cfg.device:
        cfg.device = str(base_device)

    cfg.weights = INITIAL_WEIGHTS_PATH

    cfg.freeze()
    dbg_cfg_brief(cfg, tag=f"CLIENT c{cid:02d}({WEATHER_NAMES[cid-1]})")
    return cfg


def build_cfg_for_server_update(device, round_idx):
    cfg = get_cfg()
    cfg.merge_from_file(ET_SERVER_CFG_PATH)
    cfg.defrost()

    cfg.role = "server"
    cfg.hyp.lr0 = 0.01
    cfg.epochs = int(SERVER_EPOCHS)
    cfg.hyp.burn_epochs = 0

    call_id = f"{int(time.time()*1000)%1_000_000_000:09d}"
    cfg.project = str(RUNS_FED)
    cfg.name = (
        f"server_update_r{round_idx:03d}_TEST_call{call_id}"
        if DEBUG_TEST else
        f"server_update_r{round_idx:03d}_call{call_id}"
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
    del trainer, callbacks
    torch.cuda.empty_cache()
    return new_global

def server_warmup(device, global_state):
    print(f"[SERVER-WARMUP] Warm up using labeled Didi server (urban_standard) data for {SERVER_WARMUP_EPOCHS} epoch(s)")
    cfg_s = build_cfg_for_server_update(device, 0)
    cfg_s.defrost()
    cfg_s.epochs = int(SERVER_WARMUP_EPOCHS)
    cfg_s.hyp.lr0 = 0.01
    cfg_s.freeze()

    callbacks = Callbacks()
    trainer = SSODTrainer(cfg_s, device, callbacks, -1, -1, 1)
    load_state_strict(trainer.model, global_state, tag="server_warmup_init")
    if getattr(trainer, "ema", None) is not None and getattr(trainer.ema, "ema", None) is not None:
        trainer.ema.ema.load_state_dict(global_state, strict=False)
    if getattr(trainer, "semi_ema", None) is not None and getattr(trainer.semi_ema, "ema", None) is not None:
        trainer.semi_ema.ema.load_state_dict(global_state, strict=False)

    trainer.train(callbacks, et_val)
    warmed = _merge_back(global_state, _clean_sd(trainer.model.state_dict()))
    del trainer, callbacks
    torch.cuda.empty_cache()
    return warmed


def global_eval_one_domain(global_state, device, val_dir, tag):
    dbg(f"[GLOBAL-EVAL] {tag} start")
    cfg_eval = get_cfg()
    cfg_eval.merge_from_file(ET_SERVER_CFG_PATH)
    cfg_eval.defrost()

    cfg_eval.role = "server"
    cfg_eval.Dataset.val = val_dir
    cfg_eval.Dataset.train = val_dir
    cfg_eval.Dataset.target = ""
    cfg_eval.Dataset.nc = 4
    cfg_eval.Dataset.names = ['person', 'car', 'truck', 'bus']
    cfg_eval.resume = False
    cfg_eval.weights = INITIAL_WEIGHTS_PATH

    if not cfg_eval.device:
        cfg_eval.device = str(device)

    cfg_eval.freeze()

    m = Model(cfg_eval).to(device)
    load_state_strict(m, global_state, tag=tag)

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

    if getattr(trainer, "semi_ema", None) is not None and getattr(trainer.semi_ema, "ema", None) is not None:
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

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        for p in ([EFFICIENT_TEACHER_PATH, FEDAVG_DIR,
                   ET_SERVER_CFG_PATH, ET_CLIENT_CFG_PATH,
                   INITIAL_WEIGHTS_PATH, SERVER_TRAIN, SERVER_VAL]
                  + CLIENT_TARGET_DIRS + CLIENT_VAL_DIRS):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[PATH ERROR] does not exist: {p}")
        print("[OK] All path checks passed")

        sanitize_labels_for_images_dir(SERVER_TRAIN, "server/train", nc=4)
        sanitize_labels_for_images_dir(SERVER_VAL, "server/val", nc=4)
        for i, val_dir in enumerate(CLIENT_VAL_DIRS, start=1):
            sanitize_labels_for_images_dir(val_dir, f"client{i}/val", nc=4)

        for i, (vtxt, vdir, weather) in enumerate(zip(CLIENT_TARGET_DIRS, CLIENT_VAL_DIRS, WEATHER_NAMES), start=1):
            print(f"[DATA] client{i}({weather}) video frame count = {count_lines(vtxt)}")
            print(f"[DATA] client{i}({weather}) validation image count = {count_lines(vdir)}")
        print(f"[DATA] server(urban_standard) training image count = {count_lines(SERVER_TRAIN)}")
        print(f"[DATA] server(urban_standard) validation image count   = {count_lines(SERVER_VAL)}")

        GLOBAL_PT_DIR.mkdir(parents=True, exist_ok=True)
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
        print(f"[SETUP] setting = Didi server(urban_standard,label) + {len(WEATHER_NAMES)} unlabeled clients")
        print("[SETUP] BDD personal split | all selected participants participate")
        print("[SETUP] paper-style update = best.pt init -> client EMA teacher -> FedAvg -> every-round server labeled update -> no global EMA")

        if FORCE_NO_RESUME:
            dbg("[INIT] FORCE_NO_RESUME=True, Initialize from INITIAL_WEIGHTS_PATH")
            start_round = 0
            global_state = _init_global_from_best_pt()
        else:
            start_round, global_state = _resume_last_round(GLOBAL_PT_DIR)

        sd_fingerprint(global_state, tag=f"global_init_r{start_round:03d}")

        # ==================== Main loop ====================
        for r in range(start_round + 1, ROUNDS + 1):
            phase = "Phase1:backbone-only" if r <= PHASE1_ROUNDS else "Phase2:full-network"
            print(f"\n{'='*60}")
            print(f"Round {r}/{ROUNDS}  [{phase}]")
            print(f"{'='*60}")

            client_states = []
            client_sizes = []

            client_indexes = list(range(NUM_CLIENTS))

            for idx in client_indexes:
                cid = idx + 1
                weather = WEATHER_NAMES[idx]
                video_txt = CLIENT_TARGET_DIRS[idx]
                val_dir = CLIENT_VAL_DIRS[idx]

                print(f"\n[r{r:03d}][c{cid:02d}({weather})] Start training")

                cfg_c = build_cfg_for_client(device, cid, r, video_txt, val_dir)
                callbacks = Callbacks()
                trainer = None
                try:
                    trainer = SSODTrainer(cfg_c, device, callbacks, -1, -1, 1)

                    dbg(f"[CLIENT BUILD] r{r:03d} c{cid:02d} role={trainer.role}")
                    dbg(f"[CLIENT BUILD] train_loader is None? {trainer.train_loader is None}")
                    dbg(f"[CLIENT BUILD] unlabeled_dataloader is None? {trainer.unlabeled_dataloader is None}")
                    dbg(f"[CLIENT BUILD] nb={trainer.nb}")

                    sync_global_to_trainer(trainer, global_state, cid, r)
                    set_client_train_stage(trainer.model, r, phase1_rounds=PHASE1_ROUNDS)

                    dbg(f"[CLIENT TRAIN] r{r:03d} c{cid:02d} begin")
                    res_cli = trainer.train(callbacks, et_val)
                    dbg(f"[CLIENT TRAIN] r{r:03d} c{cid:02d} done")

                    append_summary(r, cid, f"client_{weather}", res_cli)
                    print(f"[r{r:03d}][c{cid:02d}({weather})] "
                          f"P={res_cli[0]:.4f} R={res_cli[1]:.4f} "
                          f"mAP50={res_cli[2]:.4f} mAP50_95={res_cli[3]:.4f}")

                    state_to_save = _state_dict_for_fedavg(trainer)
                    client_states.append(state_to_save)
                    client_sizes.append(count_lines(video_txt))

                    sd_fingerprint(state_to_save, tag=f"client_r{r:03d}_c{cid:02d}")
                finally:
                    if CLEAN_CLIENT_RUN_DIRS and trainer is not None:
                        shutil.rmtree(trainer.save_dir, ignore_errors=True)
                    del trainer, callbacks
                    torch.cuda.empty_cache()

            dbg(f"[ROUND {r}] collected client_states={len(client_states)}")

            if not client_states:
                print(f"[Round {r}] No client completed successfully; terminating")
                break

            assert_same_keys_and_shapes(client_states, tag=f"r{r:03d}_before_FedAvg")

            dbg(f"[ROUND {r}] FedAvg start")
            avg_state = FedAvg(client_states, client_sizes)
            if r <= PHASE1_ROUNDS:
                avg_state = _merge_backbone_only(global_state, avg_state)
            dbg(f"[ROUND {r}] FedAvg done")
            sd_fingerprint(avg_state, tag=f"avg_r{r:03d}")

            print(f"[SERVER-UPDATE][r{r:03d}] Update using Didi server (urban_standard)")
            srv_state = server_update_one_epoch(device, r, avg_state)

            sd_fingerprint(srv_state, tag=f"srv_state_r{r:03d}")

            global_state = OrderedDict(
                (k.replace("module.", ""),
                 v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in srv_state.items()
            )
            sd_fingerprint(global_state, tag=f"global_after_server_update_r{r:03d}")

            ckpt_path = GLOBAL_PT_DIR / f"fed_global_r{r:03d}.pt"
            torch.save(global_state, ckpt_path)
            print(f"[Round {r}] global saved -> {ckpt_path}")

            res_server = global_eval_one_domain(
                global_state, device, SERVER_VAL,
                tag=f"global_on_server_r{r:03d}"
            )
            append_summary(r, None, "global_on_server", res_server)
            print(f"[r{r:03d}][global_on_server] "
                  f"P={res_server[0]:.4f} R={res_server[1]:.4f} "
                  f"mAP50={res_server[2]:.4f} mAP50_95={res_server[3]:.4f}")

            global_client_results = []
            for weather, val_dir in zip(WEATHER_NAMES, CLIENT_VAL_DIRS):
                res_glb = global_eval_one_domain(
                    global_state, device, val_dir,
                    tag=f"global_on_{weather}_r{r:03d}"
                )
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

        print("\n✅ global_navigation completed")
        print(f"Results summary: {SUMMARY_CSV}")
