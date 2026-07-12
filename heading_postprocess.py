"""
heading_postprocess.py -- fixes the 180deg heading flips (measurement-safe).

stage-3 reads each fly's heading per frame from the head/thorax/abdomen
segmentation. at ~15px the head and abdomen look almost identical, so layer 2
sometimes swaps them and the arrow flips ~180deg for a frame or two (about 5.8%
of frame-to-frame transitions). since a fly cant physically spin 180deg in
1/30s, i link each fly into a track and just flip the sign of any heading that
points backwards vs the track's recent heading. the axis from segmentation is
trusted -- only the front/back sign gets fixed.

positions (fly_x, fly_y) are never touched, so speed/velocity stay real. i keep
the raw heading_deg and add: track_id, heading_corrected, flipped (audit flag),
and motion_speed / motion_agree (diagnostics only -- heading is never snapped to
motion, so real backward/sideways walking is preserved).

usage:
  python heading_postprocess.py --csv <video>_stage3.csv [--out <video>_stage3_corrected.csv]
                                [--link-gate 25] [--max-gap 5] [--smooth 0.6]
                                [--turn-thresh 90]
"""
import argparse, math
import numpy as np
import pandas as pd

EMPTY = ""


def link_tracks(df, gate, max_gap, cap_n=None):
    
    df = df.sort_values(["frame", "det"]).reset_index()  # keep original index in 'index'
    track_of = {}                       # original_index -> track_id
    active = []                         # list of dicts: {tid, last_frame, x, y}
    next_tid = 0
    for fr, grp in df.groupby("frame", sort=True):
        dets = grp[["fly_x", "fly_y"]].to_numpy()
        oidx = grp["index"].to_numpy()
        used_d = set()
        if active:
            ax = np.array([a["x"] for a in active])
            ay = np.array([a["y"] for a in active])
            af = np.array([a["last_frame"] for a in active])
            # cost matrix (n_active, n_det)
            cost = np.hypot(ax[:, None] - dets[None, :, 0], ay[:, None] - dets[None, :, 1])
            gates = gate * np.clip(fr - af, 1, max_gap + 1)   # grow gate, but bounded
            used_a = set()
            pairs = sorted(((cost[a, d], a, d)
                            for a in range(len(active)) for d in range(len(oidx))),
                           key=lambda t: t[0])
            for c, a, d in pairs:
                if a in used_a or d in used_d:
                    continue
                if c > gates[a]:
                    continue
                used_a.add(a); used_d.add(d)
                tid = active[a]["tid"]
                track_of[oidx[d]] = tid
                active[a].update(last_frame=fr, x=dets[d, 0], y=dets[d, 1])
        # unmatched detections -> new tracks, unless we are at the cap
        for d in range(len(oidx)):
            if d in used_d:
                continue
            if cap_n is not None and len(active) >= cap_n:
                continue                                  # at cap: leave unlinked (-1)
            track_of[oidx[d]] = next_tid
            active.append({"tid": next_tid, "last_frame": fr,
                           "x": dets[d, 0], "y": dets[d, 1]})
            next_tid += 1
        # uncapped: retire tracks missing > max_gap. capped: keep them alive so the
        # fly can re-acquire its own slot.
        if cap_n is None:
            active = [a for a in active if fr - a["last_frame"] <= max_gap]
    return pd.Series(track_of)


def correct_track(sub, window, iters, motion_thresh, max_gap, use_motion_vote=False):
    """Sign-correct one track's headings.

    Stage (1), local temporal consistency:

      a real turn is gradual, a head/abdomen swap is an isolated outlier. Within a
      centered window we flip the few frames whose sign disagrees with the local
      majority direction. Not a forward chain, so a bad first frame can't poison
      the track, and genuine sustained turns/backward-walks (continuous) survive.

    Stage (2), global motion vote (disabled by default, use_motion_vote=False):

      assumes flies walk forward and flips a whole segment to agree with travel.
      On GROUP videos (flies stop/start/get jostled) this assumption fails often
      and REVERSES correct segments (~12% on our data), so we leave it off and
      rely on the abdomen-anchored axis for the global sign instead.

    Returns (corrected_deg list, flipped list) aligned to sub rows."""
    corrected = [EMPTY] * len(sub)
    flipped = [False] * len(sub)
    frames = sub["frame"].to_numpy()
    hx = sub["fly_x"].to_numpy(); hy = sub["fly_y"].to_numpy()
    hd = sub["heading_deg"].to_numpy()

    # rows that actually carry a heading
    have = np.array([i for i, h in enumerate(hd)
                     if not (h == "" or (isinstance(h, float) and math.isnan(h)))],
                    dtype=int)
    if len(have) == 0:
        return corrected, flipped

    # forward unit vector in IMAGE coords (y down): heading_deg = atan2(-dy, dx)
    ux = np.full(len(sub), np.nan, dtype=float)
    uy = np.full(len(sub), np.nan, dtype=float)
    for i in have:
        a = math.radians(float(hd[i]))
        ux[i] = math.cos(a)
        uy[i] = -math.sin(a)

    # split into segments separated by gaps > max_gap
    cuts = np.where(np.diff(frames[have]) > max_gap)[0] + 1
    segs = np.split(have, cuts)

    for seg in segs:
        s = np.ones(len(seg), dtype=np.int8)              # per-frame sign (+1/-1)
        sux = ux[seg]
        suy = uy[seg]
        # (1) local consistency by iterative majority within a centered window
        for _ in range(iters):
            changed = False
            for k in range(len(seg)):
                lo, hi = max(0, k - window), min(len(seg), k + window + 1)
                mx = float(np.dot(s[lo:hi], sux[lo:hi]) - s[k] * sux[k])
                my = float(np.dot(s[lo:hi], suy[lo:hi]) - s[k] * suy[k])
                if s[k] * (sux[k] * mx + suy[k] * my) < 0:
                    s[k] = -s[k]; changed = True
            if not changed:
                break
        # (2) global MOTION vote -- OFF by default (reverses correct group-video
        #     segments). Kept available for the single-fly footage it was built on.
        if use_motion_vote:
            vote = 0.0
            for k in range(1, len(seg)):
                i, p = int(seg[k]), int(seg[k - 1])
                dt = frames[i] - frames[p]
                if dt <= 0:
                    continue
                vx, vy = (hx[i] - hx[p]) / dt, (hy[i] - hy[p]) / dt
                spd = math.hypot(vx, vy)
                if spd < motion_thresh:
                    continue
                vote += s[k] * (sux[k] * vx + suy[k] * vy) / spd
            if vote < 0:                                 # whole segment is backward
                s *= -1
        # emit corrected angles (corrected RAW direction, no fabricated smoothing)
        for k, i in enumerate(seg):
            ucx = s[k] * sux[k]
            ucy = s[k] * suy[k]
            corrected[int(i)] = round(math.degrees(math.atan2(-ucy, ucx)), 1)
            flipped[int(i)] = bool(s[k] == -1)
    return corrected, flipped


def correct_dataframe(df, link_gate=25.0, max_gap=5, window=3, iters=8,
                      motion_thresh=0.8, cap_n=None, use_motion_vote=False):
    """Add track_id, heading_corrected, flipped, motion_speed, motion_agree to a
    stage-3 dataframe. Positions are never modified; heading_deg is preserved.
    Returns the same df with the new columns. (Shared by the CLI below and by
    inference.py so the live pipeline and the post-step stay identical.)"""
    df = df.copy()
    df["fly_x"] = df["fly_x"].astype(float)
    df["fly_y"] = df["fly_y"].astype(float)

    # 1) link into tracks
    tid = link_tracks(df, link_gate, max_gap, cap_n=cap_n)
    df["track_id"] = df.index.map(tid).fillna(-1).astype(int)

    # Build the new columns as plain Python lists (positional) and assign once as
    # object-dtype columns. Cell-by-cell writes into a column seeded with "" make
    # newer pandas infer a string dtype that then rejects floats -- so we avoid it.
    n = len(df)
    pos = {lbl: i for i, lbl in enumerate(df.index)}     # row label -> position
    corrected = [EMPTY] * n
    flipped = [False] * n
    motion_speed = [EMPTY] * n
    motion_agree = [EMPTY] * n

    # 2) per-track sign correction
    linked = df[df["track_id"] >= 0]
    for t, sub in linked.groupby("track_id"):
        sub = sub.sort_values("frame")
        corr, flip = correct_track(sub, window, iters, motion_thresh, max_gap, use_motion_vote)
        for lbl, c, fl in zip(sub.index, corr, flip):
            corrected[pos[lbl]] = c
            flipped[pos[lbl]] = fl

    # 3) DIAGNOSTIC kinematics from POSITION only (never alters heading)
    for t, sub in linked.groupby("track_id"):
        sub = sub.sort_values("frame")
        lbls = list(sub.index)
        x = sub["fly_x"].to_numpy(); y = sub["fly_y"].to_numpy()
        f = sub["frame"].to_numpy()
        for k in range(1, len(sub)):
            dt = f[k] - f[k - 1]
            if dt <= 0:
                continue
            vx, vy = (x[k] - x[k - 1]) / dt, (y[k] - y[k - 1]) / dt
            spd = math.hypot(vx, vy)
            p = pos[lbls[k]]
            motion_speed[p] = round(spd, 3)
            hc = corrected[p]
            if hc != EMPTY and spd > 1e-6:
                a = math.radians(float(hc))
                # heading uses screen-up = +deg (atan2(-dy,dx)); velocity dy is screen-down
                hx, hy = math.cos(a), -math.sin(a)
                motion_agree[p] = round((hx * vx + hy * vy) / spd, 3)

    df["heading_corrected"] = pd.array(corrected, dtype=object)
    df["flipped"] = flipped
    df["motion_speed"] = pd.array(motion_speed, dtype=object)
    df["motion_agree"] = pd.array(motion_agree, dtype=object)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="stage-3 heading CSV")
    ap.add_argument("--out", default=None)
    ap.add_argument("--link-gate", type=float, default=25.0,
                    help="px: max plausible movement/frame for NN linking")
    ap.add_argument("--max-gap", type=int, default=5,
                    help="frames: tolerate this gap before resetting a track/anchor")
    ap.add_argument("--window", type=int, default=3,
                    help="frames: half-width of the local sign-consistency window")
    ap.add_argument("--iters", type=int, default=8,
                    help="max passes of local sign-consistency")
    ap.add_argument("--motion-thresh", type=float, default=0.8,
                    help="px/frame: only frames moving faster than this vote on the "
                         "one global sign bit per segment")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)   # keep '' empties as ''
    df = correct_dataframe(df, args.link_gate, args.max_gap, args.window,
                           args.iters, args.motion_thresh)

    out = args.out or args.csv.replace(".csv", "_corrected.csv")
    df.to_csv(out, index=False)

    # --- report ---
    have = df[df["heading_corrected"] != EMPTY]
    n_flip = int(df["flipped"].sum())
    print(f"wrote {out}")
    print(f"  rows                : {len(df)}")
    print(f"  tracks linked       : {df['track_id'].nunique()}")
    print(f"  frames w/ heading   : {len(have)}")
    print(f"  sign-flips applied  : {n_flip}  ({100*n_flip/max(1,len(have)):.2f}% of headed frames)")


if __name__ == "__main__":
    main()
