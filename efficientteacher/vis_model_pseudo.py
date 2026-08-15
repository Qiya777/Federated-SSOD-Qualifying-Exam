import os
import glob
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


def load_ckpt_model(weights, device):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = ckpt["model"].float()
    model = model.fuse().eval() if hasattr(model, "fuse") else model.eval()
    return model


def preprocess_bgr(im0, img_size=640, stride=32):
    img = letterbox(im0, new_shape=img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    if img.ndim == 3:
        img = img.unsqueeze(0)
    return img


def draw_boxes(im0, det, names):
    out = im0.copy()
    n_det = 0 if det is None else len(det)

    if det is not None and len(det):
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            cls = int(cls)
            label = f"{names[cls]} {float(conf):.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)   # Orange bounding box
            cv2.putText(
                out,
                label,
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),   # Orange text
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        f"dets={n_det}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def images_to_video(img_dir, out_mp4, fps=10):
    """First write the raw video with mp4v, then convert it to a playable H.264 version using ffmpeg."""
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        print(f"[WARN] no jpg found in {img_dir}")
        return

    first = cv2.imread(img_paths[0])
    h, w = first.shape[:2]

    # Step 1: Write the raw video using mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for p in img_paths:
        im = cv2.imread(p)
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        writer.write(im)
    writer.release()
    print(f"[OK] raw video saved: {out_mp4}")

    # Step 2: Convert to H.264 + yuv420p with ffmpeg for broad browser/player compatibility
    playable = out_mp4.replace(".mp4", "_playable.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{out_mp4}" '
        f'-vcodec libx264 -crf 23 -pix_fmt yuv420p '
        f'"{playable}" -loglevel warning'
    )
    if ret == 0:
        print(f"[OK] playable video: {playable}")
    else:
        print(f"[WARN] ffmpeg failed (ret={ret}), use raw mp4 instead")


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    video_dir = os.path.dirname(args.out_video)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    device = select_device(args.device)
    model = load_ckpt_model(args.weights, device)

    model_stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    names = model.names if hasattr(model, "names") else [str(i) for i in range(100)]

    # Collect all images
    img_paths = sorted(glob.glob(os.path.join(args.source, "*.jpg")))
    img_paths += sorted(glob.glob(os.path.join(args.source, "*.png")))

    # Fix: sample frames using args.frame_stride to avoid conflict with the model stride variable
    img_paths = img_paths[::args.frame_stride]

    if args.max_frames > 0:
        img_paths = img_paths[:args.max_frames]

    print(f"[INFO] total frames after stride={args.frame_stride} sampling = {len(img_paths)}")
    print(f"[INFO] model_stride = {model_stride}")
    print(f"[INFO] img_size     = {args.imgsz}")
    print(f"[INFO] conf_thres   = {args.conf_thres}")
    print(f"[INFO] iou_thres    = {args.iou_thres}")

    for i, p in enumerate(img_paths, 1):
        im0 = cv2.imread(p)
        if im0 is None:
            print(f"[WARN] failed to read: {p}")
            continue

        img = preprocess_bgr(im0, img_size=args.imgsz, stride=model_stride).to(device)

        with torch.no_grad():
            raw = model(img)
            pred = raw[0] if isinstance(raw, (list, tuple)) else raw
            pred = non_max_suppression(
                pred,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                classes=None,
                agnostic=False,
                max_det=args.max_det,
            )

        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        vis = draw_boxes(im0, det, names)

        save_path = os.path.join(args.save_dir, Path(p).stem + ".jpg")
        cv2.imwrite(save_path, vis)

        if i == 1 or i % 50 == 0 or i == len(img_paths):
            n = 0 if det is None else len(det)
            print(f"[PROG] {i}/{len(img_paths)}  det={n}  {Path(p).name}")

    images_to_video(args.save_dir, args.out_video, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      type=str,   required=True)
    parser.add_argument("--source",       type=str,   required=True,
                        help="folder of extracted video frames (jpg/png)")
    parser.add_argument("--save-dir",     type=str,   required=True)
    parser.add_argument("--out-video",    type=str,   required=True)
    parser.add_argument("--imgsz",        type=int,   default=640)
    parser.add_argument("--conf-thres",   type=float, default=0.25)
    parser.add_argument("--iou-thres",    type=float, default=0.45)
    parser.add_argument("--max-det",      type=int,   default=300)
    parser.add_argument("--fps",          type=int,   default=10)
    parser.add_argument("--max-frames",   type=int,   default=-1,
                        help="<=0 means all frames")
    parser.add_argument("--device",       type=str,   default="0")
    # Fix: rename the argument to --frame-stride to avoid conflict with the model's internal stride variable
    parser.add_argument("--frame-stride", type=int,   default=1,
                        help="sample every N-th frame from source folder (e.g. 6 → ~400 frames from 2400)")
    args = parser.parse_args()
    main(args)