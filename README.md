here is the folder titled final upload 1 for the two stage fly segmentation pipeline made by Christian in the Scaplen Lab, 2026

how it works: both models were trained on datasets i collected from raw video frames. these datasets are available publicly on roboflow, and provided in this folder as well.

both datasets were annotated on roboflow using Meta's SAM3, and then verified by me.

stage1_wholefly_dataset is the wholefly dataset consisting of 1568 images total (1520 train / 28 val / 20 test). one class: fly.

stage2_bodyparts_dataset is the fly body parts: head, thorax and abdomen. this dataset is 220 images total (192 train / 28 val). these arent full frames, theyre single-fly crops: stage 1 finds each fly, the crop around it gets pulled and upscaled, and then thats what gets annotated. thats how stage 2 can resolve parts that are only a few pixels wide in the full frame.

both stage 1 and stage two are YOLO26m segmentation models (the medium variant). both weights are 52MB each and validation mAP50 is about 0.995 for both. the weights are on huggingface (link tbd), not in this repo.


whats in this folder (so far)
- train_stage1_wholefly_seg.py / train_stage2_bodyparts_seg.py  -- these are the training scripts
- stage1_wholefly_dataset / stage2_bodyparts_dataset            -- the roboflow datasets (wired into the training scripts)
- the weights (stage1_wholefly_seg.pt / stage2_bodyparts_seg.pt) are on huggingface, not in this repo (link tbd)
- inference.py: runs both models on a video, writes a csv (+ optional qc video)
- heading_postprocess.py: helper used by inference.py (the heading flip-correction) this gives us more accurate results, as sometimes the head and abdomen can get flipped- this is due to video quality/grayscale
- analyze_trajectorytools.py -- runs the trajectorytools toolbox on the csv
- plot_metrics.py         -- makes the matplotlib figures
- requirements.txt
- results/: the figures and the per-dish data from our analysis (see the results section at the bottom)


setup
everything runs on a vast.ai gpu instance (cuda). pick a pytorch/cuda template so torch is already installed, then:

    pip install -r requirements.txt


1. training
each script is wired to its dataset in this folder, so you just run:

    python train_stage1_wholefly_seg.py
    and/or (i recommend one at a time)
    python train_stage2_bodyparts_seg.py

they train YOLO26m-seg and drop a best.pt in runs/. the already-trained weights are on huggingface (link tbd), so you can skip training, download them, and go straight to inference.


2. inference (this is how we get the csv)
run both stages on a video at once:

    python inference.py --video myvideo.mp4 \
        --fly-model stage1_wholefly_seg.pt \
        --parts-model stage2_bodyparts_seg.pt \
        --out stage3_out --qc-video (to visually see your results)

this writes:
    stage3_out/myvideo_stage3.csv    this is the csv file
    stage3_out/myvideo_stage3.mp4    this is the video if you opt in for it

the csv is one row per fly per frame: the fly position (fly_x, fly_y), the head / thorax / abdomen points, heading_deg and heading_corrected (the flip-corrected heading), the track_id, and the motion columns. positions are never smoothed, so speed and velocity stay real measurements.


3. trajectorytools
feed that csv into the trajectorytools toolbox to get the locomotion metrics (velocity, acceleration, curvature, distance travelled, straightness, pairwise distances, polarization). all distances come out in mm, calibrated per dish from the 35mm dish:

    python analyze_trajectorytools.py --csv stage3_out/myvideo_stage3.csv --out metrics

this writes three csvs into metrics/
    myvideo_tt_summary.csv    <- one row, whole-video summary
    myvideo_tt_perfly.csv     <- one row per fly
    myvideo_tt_perframe.csv   <- one row per frame


4. plots (matplotlib)
make the figures from those three csvs:

    python plot_metrics.py \
        --summary  metrics/myvideo_tt_summary.csv \
        --perfly   metrics/myvideo_tt_perfly.csv \
        --perframe metrics/myvideo_tt_perframe.csv \
        --out figures

you get png figures plus a combined myvideo_metric_figures.pdf in figures/.


so the whole flow is:
    video -> inference.py -> _stage3.csv -> analyze_trajectorytools.py -> _tt_*.csv -> plot_metrics.py -> figures


whats in results/
these are the group-level figures for the paper (pooled across every video, separate from the single-video plot_metrics.py above) and the per-dish data behind them. dSERT is red, w1118 (control) is grey. all pooled across the 4 doses unless noted:
- timeline_speed_mm_s / timeline_ang_vel_deg_s: walking speed and turning rate over the 20 min, dSERT vs w1118, all 4 doses on one graph (solid = dSERT, dotted = w1118)
- zone_occupancy_bars: how much time flies spend in the center / middle / outer of the dish (thigmotaxis), across the 4 phases. dSERT hugs the wall more, strongest on ethanol.
- social_null_index: social spacing after removing the "where they are" confound with a within-dish null model. shows dSERT stays a bit more clustered than controls on ethanol.
- zone_perdish.csv / social_null_perdish.csv: the per-dish numbers (unit = dish, n=64/genotype)
- zone_stats.txt: the stats (mann-whitney / t-test per phase)
