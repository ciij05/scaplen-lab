"""
Stage 3, two-layer cascade: whole-fly detection -> body-part segmentation -> heading.

Layer 1: the whole-fly segmentation model finds every fly in the full frame.
Layer 2: for each fly we crop+upsample exactly like the training crop step (so inference
         matches training), then run the body-part model to get head / thorax /
         abdomen masks.
Heading: vector (rear -> front) along the colinear body axis abdomen-thorax-head,
         using whichever >=2 parts are present.

FLIP-CORRECTION (built in)
  The raw per-frame heading can swap head<->abdomen for a frame or two (the two
  ends are near-identical blobs at ~15 px), flipping the arrow ~180deg. Because a
  fly cannot physically rotate 180deg in 1/30 s, we link detections into per-fly
  tracks and resolve that front/back SIGN with temporal continuity + a per-track
  motion vote (see heading_postprocess.correct_dataframe). This runs as a second
  inference pass: Pass 1 collects detections, we correct the headings, Pass 2
  re-renders with the CORRECTED heading arrow.

  Positions are never modified, so speed/velocity are untouched. The raw
  heading_deg is preserved; heading_corrected is what gets drawn.

Outputs (next to --out):
  <video>_stage3.mp4   optional QC video (upscaled), part masks + CORRECTED heading arrow
  <video>_stage3.csv   per-fly, per-frame parts + heading_deg + heading_corrected,
                       flipped, track_id, motion_speed, motion_agree

Usage:
  python inference.py --video test_1min_mid.mp4 \
      --fly-model stage1_wholefly_seg.pt \
      --parts-model stage2_bodyparts_seg.pt --out stage3_out
"""
import argparse, csv, math
import numpy as np, cv2, pandas as pd
from pathlib import Path
from ultralytics import YOLO
from heading_postprocess import correct_dataframe

PAD_FRAC = 0.35          # must match the training crops
OUT = 256               # crop size used in training
SCALE = 4               # upscale factor for the QC video (240x210 -> 960x840)
MIN_ARROW = 14          # px (full-frame): clamp heading arrow so it can't blow up
MAX_ARROW = 30          # px (full-frame)
DEDUP_FRAC = 0.5        # drop a Layer-1 box whose centre is within this*flysize of a stronger one

# Layer-2 class ids: 0=abdomen, 1=head, 2=thorax
PART_COLOR = {"head": (255, 90, 0), "thorax": (0, 210, 0), "abdomen": (40, 40, 255)}  # BGR

CSV_COLS = ["frame", "det", "fly_x", "fly_y", "head_x", "head_y",
            "thorax_x", "thorax_y", "abdomen_x", "abdomen_y", "heading_deg",
            "heading_src", "fly_conf", "box_w", "box_h", "edge", "n_near", "head_found"]


def crop_square(frame, box, W, H):
    """Crop a padded square around a box and upsample to OUTxOUT (mirrors the training crops)."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    half = side / 2 + PAD_FRAC * side
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1, sy1, sx2, sy2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    crop = np.zeros((sy2 - sy1, sx2 - sx1, 3), np.uint8)
    ix1, iy1, ix2, iy2 = max(0, sx1), max(0, sy1), min(W, sx2), min(H, sy2)
    crop[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = frame[iy1:iy2, ix1:ix2]
    cropr = cv2.resize(crop, (OUT, OUT), interpolation=cv2.INTER_CUBIC)
    return cropr, (sx1, sy1, sx2 - sx1, sy2 - sy1)   # origin x,y + width,height


def to_full(px, py, origin):
    sx1, sy1, sw, sh = origin
    return sx1 + px / OUT * sw, sy1 + py / OUT * sh


def paste_mask(overlay, mask256, origin, color):
    """Resize a 256x256 part mask back to full-frame (upscaled) and color it in."""
    sx1, sy1, sw, sh = origin
    tw, th = max(1, int(sw * SCALE)), max(1, int(sh * SCALE))
    m = cv2.resize(mask256.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
    ox, oy = int(sx1 * SCALE), int(sy1 * SCALE)
    H2, W2 = overlay.shape[:2]
    dx1, dy1 = max(0, ox), max(0, oy)
    dx2, dy2 = min(W2, ox + tw), min(H2, oy + th)
    if dx1 >= dx2 or dy1 >= dy2:
        return
    sub = m[dy1 - oy:dy2 - oy, dx1 - ox:dx2 - ox]
    region = overlay[dy1:dy2, dx1:dx2]
    region[sub] = color


def infer_frame(frame, fr, layer1, layer2, id2name, args, W, H):
    """Run the two-layer cascade on one frame. Returns a list of detection dicts
    holding everything needed to BOTH log a CSV row and (re)draw the QC overlay.
    Deterministic, so calling it again in Pass 2 reproduces Pass 1 exactly."""
    r1 = layer1.predict(frame, conf=args.fly_conf, imgsz=1280, verbose=False, max_det=30)[0]
    boxes = r1.boxes.xyxy.cpu().numpy() if r1.boxes is not None else np.empty((0, 4))
    confs = r1.boxes.conf.cpu().numpy() if r1.boxes is not None else np.empty((0,))

    
    if len(boxes):
        cen = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
        siz = np.array([max(b[2] - b[0], b[3] - b[1]) for b in boxes])
        keep = []
        for i in np.argsort(-confs):
            if all(np.hypot(*(cen[i] - cen[k])) >= DEDUP_FRAC * max(siz[i], siz[k]) for k in keep):
                keep.append(i)
        keep = sorted(keep)
        boxes, confs = boxes[keep], confs[keep]

    centers = (np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
               if len(boxes) else np.empty((0, 2)))

    dets = []
    if len(boxes) == 0:
        return dets
    crops, origins = [], []
    for b in boxes:
        crop, origin = crop_square(frame, b, W, H)
        crops.append(crop); origins.append(origin)

    r2_list = layer2.predict(crops, conf=args.part_conf, imgsz=OUT,
                             retina_masks=True, verbose=False)
    for di, (r2, origin, b) in enumerate(zip(r2_list, origins, boxes)):
        pts = {}                 # name -> (full_x, full_y)
        part_masks = []          # (name, mask256_bool, origin) for drawing
        if r2.masks is not None and len(r2.masks.data):
            masks = r2.masks.data.cpu().numpy()
            cls = r2.boxes.cls.cpu().numpy().astype(int)
            conf = r2.boxes.conf.cpu().numpy()
            # exactly ONE instance per class per fly: keep the highest-conf one
            for c in np.unique(cls):
                idxs = np.where(cls == c)[0]
                best = idxs[np.argmax(conf[idxs])]
                m = masks[best] > 0.5
                if m.sum() == 0:
                    continue
                ys, xs = np.where(m)
                name = id2name[int(c)]
                pts[name] = to_full(xs.mean(), ys.mean(), origin)
                part_masks.append((name, m, origin))

        fly_cx, fly_cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        fly_size = max(b[2] - b[0], b[3] - b[1])
        head, thorax, abdomen = pts.get("head"), pts.get("thorax"), pts.get("abdomen")
        heading, heading_src = None, ""

        def _ang(p, q):   # heading of vector q -> p, 0=right 90=up; None if degenerate
            dx, dy = p[0] - q[0], p[1] - q[1]
            return math.degrees(math.atan2(-dy, dx)) if math.hypot(dx, dy) > 1e-3 else None

        if abdomen is not None and thorax is not None:
            # PRIMARY: the stable body axis abdomen(rear) -> thorax, which points
            # toward the head. Both parts are detected ~99% and lie on the body axis.
            heading, heading_src = _ang(thorax, abdomen), "axis"
            # Upgrade to abdomen -> head for a longer, more precise baseline ONLY if
            # the head lines up with the body axis (<40 deg). This REJECTS off-axis
            # head mis-detections that were throwing the arrow sideways.
            if head is not None and heading is not None:
                ha = _ang(head, abdomen)
                if ha is not None and abs(((ha - heading + 180) % 360) - 180) < 40:
                    heading, heading_src = ha, "head"
        elif abdomen is not None and head is not None:
            heading, heading_src = _ang(head, abdomen), "head"            # no thorax
        elif abdomen is not None:
            heading, heading_src = _ang((fly_cx, fly_cy), abdomen), "abd_center"  # abdomen only
        elif head is not None and thorax is not None:
            heading, heading_src = _ang(head, thorax), "head"             # no abdomen (rare)

        edge = 1 if (b[0] <= 1 or b[1] <= 1 or b[2] >= W - 1 or b[3] >= H - 1) else 0
        if len(centers) > 1:
            d = np.hypot(centers[:, 0] - fly_cx, centers[:, 1] - fly_cy)
            n_near = int(((d > 1e-6) & (d < 1.5 * fly_size)).sum())
        else:
            n_near = 0

        dets.append({
            "di": di, "fly_cx": fly_cx, "fly_cy": fly_cy, "fly_size": fly_size,
            "thorax": thorax, "pts": pts, "part_masks": part_masks,
            "heading": heading, "heading_src": heading_src,
            "conf": float(confs[di]), "box_w": float(b[2] - b[0]),
            "box_h": float(b[3] - b[1]), "edge": edge, "n_near": n_near,
            "head_found": int(head is not None),
        })
    return dets


def det_to_row(fr, d):
    pts = d["pts"]
    return [
        fr, d["di"], round(d["fly_cx"], 2), round(d["fly_cy"], 2),
        *(round(v, 2) for v in (pts.get("head") or (np.nan, np.nan))),
        *(round(v, 2) for v in (pts.get("thorax") or (np.nan, np.nan))),
        *(round(v, 2) for v in (pts.get("abdomen") or (np.nan, np.nan))),
        round(d["heading"], 1) if d["heading"] is not None else "", d["heading_src"],
        round(d["conf"], 3), round(d["box_w"], 1), round(d["box_h"], 1),
        d["edge"], d["n_near"], d["head_found"],
    ]


def draw_frame(frame, dets, heading_lookup, fr):
    """Render the QC overlay for one frame, using the CORRECTED heading."""
    H, W = frame.shape[:2]
    canvas = cv2.resize(frame, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
    overlay = canvas.copy()
    labels, arrows = [], []
    for d in dets:
        for name, m, origin in d["part_masks"]:
            paste_mask(overlay, m, origin, PART_COLOR[name])
            cx, cy = d["pts"][name]
            labels.append((name, (int(cx * SCALE), int(cy * SCALE)), PART_COLOR[name]))

        hc = heading_lookup.get((fr, d["di"]))
        if hc is None or hc == "":
            continue
        a = math.radians(float(hc))
        ux, uy = math.cos(a), -math.sin(a)            # corrected heading direction
        fly_cx, fly_cy, fly_size = d["fly_cx"], d["fly_cy"], d["fly_size"]
        L = min(MAX_ARROW, max(MIN_ARROW, fly_size * 1.4))
        sx_, sy_ = fly_cx - ux * L * 0.3, fly_cy - uy * L * 0.3
        ex_, ey_ = fly_cx + ux * L * 0.7, fly_cy + uy * L * 0.7
        arrows.append(((int(sx_ * SCALE), int(sy_ * SCALE)), (int(ex_ * SCALE), int(ey_ * SCALE))))

    out = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    for s, e in arrows:
        cv2.arrowedLine(out, s, e, (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    for text, (lx, ly), color in labels:
        cv2.putText(out, text, (lx + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (lx + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fly-model", required=True, help="Layer-1 whole-fly seg model")
    ap.add_argument("--parts-model", required=True, help="Layer-2 body-part seg model")
    ap.add_argument("--out", default="stage3_out")
    ap.add_argument("--fly-conf", type=float, default=0.25)
    ap.add_argument("--part-conf", type=float, default=0.05)
    ap.add_argument("--cap-n", type=int, default=None,
                    help="known fly count for this dish: hold at most N persistent "
                         "tracks (re-acquire instead of spawning new ids)")
    ap.add_argument("--resume-raw", action="store_true",
                    help="reuse a completed <video>_stage3_raw.csv checkpoint if it exists")
    ap.add_argument("--raw-csv", default=None,
                    help="optional path for the raw Pass-1 detections checkpoint")
    ap.add_argument("--qc-video", action="store_true",
                    help="render the Pass-2 QC video after writing the stage3 CSV")
    ap.add_argument("--no-qc-video", action="store_true",
                    help="write the stage3 CSV and skip the Pass-2 QC video render")
    ap.add_argument("--qc-seconds", type=float, default=None,
                    help="optional QC video duration cap in seconds")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    raw_csv = Path(args.raw_csv) if args.raw_csv else out_dir / f"{stem}_stage3_raw.csv"
    raw_done = raw_csv.with_suffix(raw_csv.suffix + ".done")
    dst_vid = out_dir / f"{stem}_stage3.mp4"
    dst_csv = out_dir / f"{stem}_stage3.csv"

    layer1 = YOLO(args.fly_model)
    layer2 = YOLO(args.parts_model)
    id2name = layer2.names    # {0:abdomen,1:head,2:thorax}

    def open_video():
        cap = cv2.VideoCapture(args.video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap, W, H, fps

    if args.resume_raw and raw_csv.exists() and raw_done.exists():
        print(f"Pass 1/2: using completed raw checkpoint -> {raw_csv}")
        df_raw = pd.read_csv(raw_csv, keep_default_na=False, low_memory=False)
        n_frames = int(df_raw["frame"].max()) + 1 if len(df_raw) else 0
        cap_tmp, _, _, fps = open_video()
        cap_tmp.release()
    else:
        if args.resume_raw and raw_csv.exists() and not raw_done.exists():
            print(f"Pass 1/2: ignoring incomplete raw checkpoint -> {raw_csv}")
        # ----- PASS 1: inference, stream raw CSV rows to disk immediately -----
        # We do NOT cache the part masks across frames (that would be GBs on a
        # long video). infer_frame is deterministic, so Pass 2 reproduces them.
        print("Pass 1/2: inference + writing raw detections checkpoint ...")
        raw_done.unlink(missing_ok=True)
        cap, W, H, fps = open_video()
        fr = 0
        row_count = 0
        with raw_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLS)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rows = [det_to_row(fr, d)
                              for d in infer_frame(frame, fr, layer1, layer2, id2name, args, W, H)]
                writer.writerows(frame_rows)
                row_count += len(frame_rows)
                fr += 1
                if fr % 200 == 0:
                    fh.flush()
                    print(f"  frame {fr} ...")
        cap.release()
        n_frames = fr
        raw_done.write_text(f"frames={n_frames}\nrows={row_count}\n")
        print(f"  raw checkpoint: {row_count} detections -> {raw_csv.name}")
        df_raw = pd.read_csv(raw_csv, keep_default_na=False, low_memory=False)

    # ----- heading: abdomen-anchored geometry + SIGN-ONLY local de-jitter -----
    # heading_deg (raw, abdomen->thorax axis) already has the correct GLOBAL sign,
    # so we only need to fix isolated single-frame head/abdomen mislabels. We run
    # correct_dataframe with use_motion_vote=False: stage-1 local sign-consistency
    # ONLY (flips a frame iff it contradicts its immediate neighbours -- physically
    # impossible to be real), motion-vote OFF (it reversed ~12% of correct group
    # segments). Positions and the body AXIS are never touched (the flip is exactly
    # +/-180 deg); raw heading_deg is preserved; flipped is an audit flag.
    print("Building heading table (axis geometry + sign-only local de-jitter) ...")
    df = correct_dataframe(df_raw, cap_n=args.cap_n, use_motion_vote=False)
    df.to_csv(dst_csv, index=False)
    n_flip = int(df["flipped"].sum())
    headed = int((df["heading_corrected"].astype(str).str.strip() != "").sum())
    heading_lookup = {(int(r.frame), int(r.det)): r.heading_corrected
                      for r in df.itertuples()}

    render_qc = args.qc_video or args.qc_seconds is not None
    if args.no_qc_video:
        render_qc = False

    if not render_qc:
        print(f"done: {n_frames} frames, {len(df_raw)} detections -> {dst_csv.name} (QC video skipped)")
        print(f"  sign-flips corrected : {n_flip} / {headed} headed frames "
              f"({100*n_flip/max(1,headed):.2f}%)")
        return

    # ----- PASS 2: re-run inference (deterministic) and render with CORRECTED arrow -----
    print("Pass 2/2: rendering QC video with corrected heading ...")
    cap, W, H, fps = open_video()
    max_qc_frames = None
    if args.qc_seconds is not None:
        max_qc_frames = max(0, int(round(args.qc_seconds * fps)))
    writer = cv2.VideoWriter(str(dst_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * SCALE, H * SCALE))
    fr = 0
    while True:
        if max_qc_frames is not None and fr >= max_qc_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        dets = infer_frame(frame, fr, layer1, layer2, id2name, args, W, H)
        out = draw_frame(frame, dets, heading_lookup, fr)
        writer.write(out)
        fr += 1
        if fr % 200 == 0:
            print(f"  frame {fr} ...")
    cap.release(); writer.release()

    print(f"done: {fr} QC frames -> {dst_vid.name}, {len(df_raw)} detections -> {dst_csv.name}")
    print(f"  sign-flips corrected : {n_flip} / {headed} headed frames "
          f"({100*n_flip/max(1,headed):.2f}%)")


if __name__ == "__main__":
    main()
