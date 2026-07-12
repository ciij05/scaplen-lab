
import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import trajectorytools as tt
import trajectorytools.socialcontext as ttsocial


def finite_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def finite_median(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def finite_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def finite_min(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan


def frac_where(values, predicate):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    return float(np.mean(predicate(values[valid])))


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def circ_mean_deg(deg):
    deg = np.asarray(deg, dtype=float)
    rad = np.deg2rad(deg[np.isfinite(deg)])
    if len(rad) == 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))))


def circ_resultant(deg):
    deg = np.asarray(deg, dtype=float)
    rad = np.deg2rad(deg[np.isfinite(deg)])
    if len(rad) == 0:
        return np.nan
    return float(np.hypot(np.nanmean(np.cos(rad)), np.nanmean(np.sin(rad))))


def round_float(x, nd=4):
    if x is None or not np.isfinite(x):
        return np.nan
    return round(float(x), nd)


def build_arrays(csv_path):
    d = pd.read_csv(csv_path, keep_default_na=False, low_memory=False)
    for c in ("frame", "track_id", "fly_x", "fly_y", "heading_corrected", "flipped"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["track_id"] >= 0].dropna(subset=["frame", "track_id", "fly_x", "fly_y"])

    ids = sorted(d["track_id"].astype(int).unique())
    id2col = {tid: k for k, tid in enumerate(ids)}
    f0, f1 = int(d["frame"].min()), int(d["frame"].max())
    F, N = f1 - f0 + 1, len(ids)

    pos = np.full((F, N, 2), np.nan, dtype=float)
    hd = np.full((F, N), np.nan, dtype=float)
    flipped = np.zeros((F, N), dtype=bool)

    fr = d["frame"].astype(int).to_numpy() - f0
    col = d["track_id"].astype(int).map(id2col).to_numpy()
    pos[fr, col, 0] = d["fly_x"].to_numpy()
    pos[fr, col, 1] = d["fly_y"].to_numpy()
    if "heading_corrected" in d.columns:
        hd[fr, col] = d["heading_corrected"].to_numpy()
    if "flipped" in d.columns:
        flipped[fr, col] = d["flipped"].fillna(0).astype(bool).to_numpy()
    return d, ids, f0, pos, hd, flipped


def gap_safe_path_metrics(pos):
    
    N = pos.shape[1]
    total = np.full(N, np.nan, dtype=float)
    displacement = np.full(N, np.nan, dtype=float)
    straightness = np.full(N, np.nan, dtype=float)
    for k in range(N):
        p = pos[:, k, :]
        valid = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])
        if valid.sum() < 2:
            continue
        step_ok = valid[:-1] & valid[1:]
        if step_ok.any():
            total[k] = float(np.nansum(tt.norm(np.diff(p, axis=0)[step_ok])))
        first = p[np.where(valid)[0][0]]
        last = p[np.where(valid)[0][-1]]
        displacement[k] = float(tt.norm(last - first))
        if np.isfinite(total[k]) and total[k] > 0:
            straightness[k] = displacement[k] / total[k]
    return total, displacement, straightness


def social_arrays(s, hvec=None):
    T, N, _ = s.shape
    iid = np.full((T, N, N), np.nan, dtype=float)
    nnd = np.full((T, N), np.nan, dtype=float)
    nn_id = np.full((T, N), -1, dtype=int)
    convex_hull = np.zeros((T, N), dtype=bool)
    alpha_border = np.zeros((T, N), dtype=bool)
    facing_nn = np.full((T, N), np.nan, dtype=float)

    center_x, center_y, radius = tt.find_enclosing_circle(s[np.isfinite(s[..., 0])])
    center = np.array([center_x, center_y])
    radius = float(radius) if np.isfinite(radius) and radius > 0 else 1.0

    for t in range(T):
        valid = np.isfinite(s[t, :, 0]) & np.isfinite(s[t, :, 1])
        idx = np.where(valid)[0]
        if len(idx) < 2:
            continue
        p = s[t, idx]
        dist = ttsocial.adjacency_matrix_in_frame(p, num_neighbours=len(idx) - 1, mode="distance")
        dist = np.asarray(dist, dtype=float)
        np.fill_diagonal(dist, np.nan)
        iid[t, np.ix_(idx, idx)[0], np.ix_(idx, idx)[1]] = dist

        nearest = np.nanargmin(dist, axis=1)
        nearest_dist = dist[np.arange(len(idx)), nearest]
        nnd[t, idx] = nearest_dist
        nn_id[t, idx] = idx[nearest]

        if hvec is not None:
            ok = np.isfinite(hvec[t, idx, 0]) & np.isfinite(nearest_dist) & (nearest_dist > 0)
            if ok.any():
                delta = p[nearest] - p
                unit = delta[ok] / nearest_dist[ok, None]
                facing_nn[t, idx[ok]] = np.sum(hvec[t, idx[ok]] * unit, axis=1)

        if len(idx) >= 3:
            try:
                hull_local = ttsocial.in_convex_hull(p[np.newaxis, ...])[0]
                convex_hull[t, idx] = hull_local
            except Exception:
                pass
        if len(idx) >= 4:
            try:
                p_norm = (p - center) / radius
                alpha_local = ttsocial.in_alpha_border(p_norm[np.newaxis, ...], alpha=5)[0]
                alpha_border[t, idx] = alpha_local
            except Exception:
                pass

    return iid, nnd, nn_id, convex_hull, alpha_border, facing_nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="metrics")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--move-thresh", type=float, default=10.0)
    ap.add_argument("--text-out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    stem = csv_path.name.replace("_stage3.csv", "").replace(".csv", "")
    genotype = (re.search(r"(dSERT|W1118)", stem) or [None, "NA"])[1]

    _, ids, f0, pos, hd, flipped = build_arrays(csv_path)
    F, N = pos.shape[:2]

    # px -> mm per dish: dish radius from the enclosing circle vs the known 35 mm.
    # scale positions up front so all distances/speeds come out in mm.
    DISH_MM = 35.0
    finite_pts = pos[np.isfinite(pos[..., 0])]
    if len(finite_pts):
        _, _, R_px = tt.find_enclosing_circle(finite_pts)
        R_px = float(R_px) if np.isfinite(R_px) and R_px > 0 else np.nan
    else:
        R_px = np.nan
    if np.isfinite(R_px):
        mm_per_px = DISH_MM / (2.0 * R_px)
    else:
        mm_per_px = 1.0
        warnings.warn(f"{stem}: could not fit dish circle; leaving positions in pixels")
    pos = pos * mm_per_px
    move_thresh = args.move_thresh * mm_per_px   # keep 'moving' identical to the px threshold

    traj = tt.Trajectories.from_positions(pos.copy(), interpolate_nans=False)
    traj.new_time_unit(args.fps, "s")
    s = np.asarray(traj.s)
    v = np.asarray(traj.v)
    a = np.asarray(traj.a)
    speed = np.asarray(traj.speed)
    acceleration = np.asarray(traj.acceleration)
    tg_acceleration = np.asarray(traj.tg_acceleration)
    normal_acceleration = np.asarray(traj.normal_acceleration)
    curvature = np.asarray(traj.curvature)
    total_distance, displacement, straightness = gap_safe_path_metrics(pos)

    T = traj.number_of_frames
    frames_mid = np.arange(f0 + 1, f0 + 1 + T)
    hd_mid = hd[1:-1]
    flipped_mid = flipped[1:-1]

    hrad = np.deg2rad(hd)
    hvec_all = np.stack([np.cos(hrad), -np.sin(hrad)], axis=-1)
    hvec_all[~np.isfinite(hd)] = np.nan
    hvec_mid = hvec_all[1:-1]

    motion_agree = np.sum(hvec_mid * v, axis=-1) / speed
    motion_agree[~np.isfinite(motion_agree)] = np.nan
    moving = speed > move_thresh
    motion_agree_moving = np.where(moving, motion_agree, np.nan)

    angular_speed = np.full((F - 1, N), np.nan)
    for k in range(N):
        valid = np.isfinite(hd[:-1, k]) & np.isfinite(hd[1:, k])
        diffs = np.abs(angle_diff_deg(hd[1:, k], hd[:-1, k])) * args.fps
        angular_speed[valid, k] = diffs[valid]
    angular_speed_mid = angular_speed[1:]

    iid, nnd, nn_id, convex_hull, alpha_border, facing_nn = social_arrays(s, hvec_mid)
    mean_iid_per_individual = np.nanmean(iid, axis=2)

    pol_vec = tt.collective.polarization(v)
    pol_norm = tt.norm(pol_vec)
    com_s = np.asarray(traj.center_of_mass.s)
    com_v = np.asarray(traj.center_of_mass.v)
    com_speed = np.asarray(traj.center_of_mass.speed)
    angular_momentum = tt.collective.angular_momentum(v, s, center=com_s)

    hvx = hvec_mid[..., 0]
    hvy = hvec_mid[..., 1]
    group_heading_polarization = np.hypot(np.nanmean(hvx, axis=1), np.nanmean(hvy, axis=1))
    group_heading_nematic = np.hypot(
        np.nanmean(np.cos(2 * np.deg2rad(hd_mid)), axis=1),
        np.nanmean(np.sin(2 * np.deg2rad(hd_mid)), axis=1),
    )

    pairwise_heading_alignment = np.full(T, np.nan)
    for t in range(T):
        valid = np.isfinite(hvec_mid[t, :, 0])
        idx = np.where(valid)[0]
        if len(idx) < 2:
            continue
        pair = hvec_mid[t, idx] @ hvec_mid[t, idx].T
        pairwise_heading_alignment[t] = np.nanmean(pair[np.triu_indices(len(idx), 1)])

    perfly = []
    for k, tid in enumerate(ids):
        perfly.append(dict(
            track=int(tid),
            coverage=round_float(np.isfinite(pos[:, k, 0]).mean()),
            heading_coverage=round_float(np.isfinite(hd[:, k]).mean()),
            flip_rate=round_float(flipped[:, k].sum() / max(1, np.isfinite(hd[:, k]).sum())),
            mean_speed=round_float(finite_mean(speed[:, k]), 3),
            median_speed=round_float(finite_median(speed[:, k]), 3),
            max_speed=round_float(finite_max(speed[:, k]), 3),
            mean_acceleration=round_float(finite_mean(acceleration[:, k]), 3),
            mean_tangential_acceleration=round_float(finite_mean(tg_acceleration[:, k]), 3),
            mean_normal_acceleration=round_float(finite_mean(normal_acceleration[:, k]), 3),
            mean_abs_curvature=round_float(finite_mean(np.abs(curvature[:, k])), 5),
            total_distance=round_float(total_distance[k], 1),
            displacement=round_float(displacement[k], 3),
            straightness=round_float(straightness[k], 4),
            active_frac=round_float(frac_where(speed[:, k], lambda x: x > move_thresh)),
            mean_nnd=round_float(finite_mean(nnd[:, k]), 3),
            median_nnd=round_float(finite_median(nnd[:, k]), 3),
            min_nnd=round_float(finite_min(nnd[:, k]), 3),
            mean_iid=round_float(finite_mean(mean_iid_per_individual[:, k]), 3),
            convex_hull_frac=round_float(convex_hull[:, k].mean()),
            alpha_border_frac=round_float(alpha_border[:, k].mean()),
            circular_mean_heading_deg=round_float(circ_mean_deg(hd[:, k]), 2),
            circular_resultant=round_float(circ_resultant(hd[:, k])),
            mean_abs_angular_speed_deg_s=round_float(finite_mean(angular_speed[:, k]), 3),
            median_abs_angular_speed_deg_s=round_float(finite_median(angular_speed[:, k]), 3),
            mean_heading_motion_agree=round_float(finite_mean(motion_agree_moving[:, k])),
            forward_motion_frac=round_float(frac_where(motion_agree_moving[:, k], lambda x: x > 0.5)),
            sideways_motion_frac=round_float(frac_where(motion_agree_moving[:, k], lambda x: np.abs(x) <= 0.5)),
            backward_motion_frac=round_float(frac_where(motion_agree_moving[:, k], lambda x: x < -0.5)),
            mean_facing_nearest_neighbor=round_float(finite_mean(facing_nn[:, k])),
            facing_neighbor_frac=round_float(frac_where(facing_nn[:, k], lambda x: x > 0.5)),
            facing_away_neighbor_frac=round_float(frac_where(facing_nn[:, k], lambda x: x < -0.5)),
        ))

    perframe = pd.DataFrame(dict(
        frame=frames_mid,
        valid_flies=np.sum(np.isfinite(s[..., 0]), axis=1),
        mean_speed=np.nanmean(speed, axis=1),
        median_speed=np.nanmedian(speed, axis=1),
        max_speed=np.nanmax(speed, axis=1),
        mean_acceleration=np.nanmean(acceleration, axis=1),
        mean_nnd=np.nanmean(nnd, axis=1),
        mean_iid=np.nanmean(mean_iid_per_individual, axis=1),
        velocity_polarization=pol_norm,
        angular_momentum=angular_momentum,
        center_of_mass_x=com_s[:, 0],
        center_of_mass_y=com_s[:, 1],
        center_of_mass_speed=com_speed,
        convex_hull_count=np.sum(convex_hull, axis=1),
        alpha_border_count=np.sum(alpha_border, axis=1),
        heading_coverage=np.mean(np.isfinite(hd_mid), axis=1),
        group_heading_polarization=group_heading_polarization,
        group_heading_nematic_order=group_heading_nematic,
        pairwise_heading_alignment=pairwise_heading_alignment,
        mean_heading_motion_agree_moving=np.nanmean(motion_agree_moving, axis=1),
        mean_facing_nearest_neighbor=np.nanmean(facing_nn, axis=1),
    ))
    perframe.to_csv(out_dir / f"{stem}_tt_perframe.csv", index=False)
    pd.DataFrame(perfly).to_csv(out_dir / f"{stem}_tt_perfly.csv", index=False)

    summary = dict(
        dish=stem,
        genotype=genotype,
        n_flies=N,
        frames=F,
        trajectorytools_frames=T,
        duration_s=round_float(F / args.fps, 1),
        fps=args.fps,
        dish_radius_px=round_float(R_px, 1),
        mm_per_px=round_float(mm_per_px, 5),
        mean_speed=round_float(finite_mean(speed), 3),
        median_speed=round_float(finite_median(speed), 3),
        max_speed=round_float(finite_max(speed), 3),
        mean_acceleration=round_float(finite_mean(acceleration), 3),
        mean_tangential_acceleration=round_float(finite_mean(tg_acceleration), 3),
        mean_normal_acceleration=round_float(finite_mean(normal_acceleration), 3),
        mean_abs_curvature=round_float(finite_mean(np.abs(curvature)), 5),
        mean_total_distance=round_float(finite_mean(total_distance), 1),
        mean_straightness=round_float(finite_mean(straightness)),
        active_frac=round_float(frac_where(speed, lambda x: x > move_thresh)),
        mean_nnd=round_float(finite_mean(nnd), 3),
        median_nnd=round_float(finite_median(nnd), 3),
        min_nnd=round_float(finite_min(nnd), 3),
        mean_iid=round_float(finite_mean(mean_iid_per_individual), 3),
        mean_velocity_polarization=round_float(finite_mean(pol_norm)),
        median_velocity_polarization=round_float(finite_median(pol_norm)),
        mean_angular_momentum=round_float(finite_mean(angular_momentum), 3),
        mean_abs_angular_momentum=round_float(finite_mean(np.abs(angular_momentum)), 3),
        mean_center_of_mass_speed=round_float(finite_mean(com_speed), 3),
        mean_convex_hull_count=round_float(finite_mean(np.sum(convex_hull, axis=1)), 3),
        mean_alpha_border_count=round_float(finite_mean(np.sum(alpha_border, axis=1)), 3),
        heading_coverage=round_float(np.isfinite(hd).mean()),
        flip_rate=round_float(flipped.sum() / max(1, np.isfinite(hd).sum())),
        mean_abs_heading_angular_speed_deg_s=round_float(finite_mean(angular_speed), 3),
        median_abs_heading_angular_speed_deg_s=round_float(finite_median(angular_speed), 3),
        mean_heading_motion_agree_moving=round_float(finite_mean(motion_agree_moving)),
        forward_motion_frac=round_float(frac_where(motion_agree_moving, lambda x: x > 0.5)),
        sideways_motion_frac=round_float(frac_where(motion_agree_moving, lambda x: np.abs(x) <= 0.5)),
        backward_motion_frac=round_float(frac_where(motion_agree_moving, lambda x: x < -0.5)),
        mean_group_heading_polarization=round_float(finite_mean(group_heading_polarization)),
        mean_group_heading_nematic_order=round_float(finite_mean(group_heading_nematic)),
        mean_pairwise_heading_alignment=round_float(finite_mean(pairwise_heading_alignment)),
        mean_facing_nearest_neighbor=round_float(finite_mean(facing_nn)),
        facing_neighbor_frac=round_float(frac_where(facing_nn, lambda x: x > 0.5)),
        facing_away_neighbor_frac=round_float(frac_where(facing_nn, lambda x: x < -0.5)),
    )
    pd.DataFrame([summary]).to_csv(out_dir / f"{stem}_tt_summary.csv", index=False)

    lines = [
        f"=== {stem} ({genotype}, N={N}, {F / args.fps / 60:.1f} min) ===",
        "TRAJECTORYTOOLS KINEMATICS",
        f"  calibration                     R={summary['dish_radius_px']} px, {summary['mm_per_px']} mm/px (35 mm dish)",
        f"  speed mean / median / max       {summary['mean_speed']} / {summary['median_speed']} / {summary['max_speed']} mm/s",
        f"  acceleration mean               {summary['mean_acceleration']} mm/s^2",
        f"  tangential / normal accel       {summary['mean_tangential_acceleration']} / {summary['mean_normal_acceleration']} mm/s^2",
        f"  total distance / fly            {summary['mean_total_distance']} mm",
        f"  straightness                    {summary['mean_straightness']}",
        f"  active fraction                 {summary['active_frac']}",
        "TRAJECTORYTOOLS SOCIAL CONTEXT",
        f"  nearest-neighbor mean/median/min {summary['mean_nnd']} / {summary['median_nnd']} / {summary['min_nnd']} mm",
        f"  inter-individual distance        {summary['mean_iid']} mm",
        f"  convex hull count                {summary['mean_convex_hull_count']}",
        f"  alpha-border count               {summary['mean_alpha_border_count']}",
        "TRAJECTORYTOOLS COLLECTIVE",
        f"  velocity polarization mean       {summary['mean_velocity_polarization']}",
        f"  angular momentum mean / abs      {summary['mean_angular_momentum']} / {summary['mean_abs_angular_momentum']}",
        f"  center-of-mass speed             {summary['mean_center_of_mass_speed']} mm/s",
        "CORRECTED HEADING / ORIENTATION",
        f"  heading coverage                 {summary['heading_coverage']}",
        f"  sign-flip rate                   {summary['flip_rate']}",
        f"  heading angular speed mean/med   {summary['mean_abs_heading_angular_speed_deg_s']} / {summary['median_abs_heading_angular_speed_deg_s']} deg/s",
        f"  heading-motion agreement         {summary['mean_heading_motion_agree_moving']}",
        f"  forward / sideways / backward    {summary['forward_motion_frac']} / {summary['sideways_motion_frac']} / {summary['backward_motion_frac']}",
        f"  group heading polarization       {summary['mean_group_heading_polarization']}",
        f"  group heading nematic order      {summary['mean_group_heading_nematic_order']}",
        f"  pairwise heading alignment       {summary['mean_pairwise_heading_alignment']}",
        f"  facing nearest-neighbor cos      {summary['mean_facing_nearest_neighbor']}",
        f"  facing toward / away NN          {summary['facing_neighbor_frac']} / {summary['facing_away_neighbor_frac']}",
        f"wrote {stem}_tt_summary.csv, {stem}_tt_perfly.csv, {stem}_tt_perframe.csv -> {out_dir}/",
    ]
    text = "\n".join(lines) + "\n"
    if args.text_out:
        Path(args.text_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
