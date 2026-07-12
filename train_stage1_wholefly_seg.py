"""
Stage 1 whole fly segmentation. built by christian 2026
"""
import subprocess, sys
from pathlib import Path

def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip("ultralytics==8.4.52")   # pinned for reproducibility; matches the verified run

from ultralytics import YOLO

HERE         = Path(__file__).resolve().parent
WORKSPACE    = "/workspace"                                   # Vast.ai run/output dir
DATASET_YAML = str(HERE / "stage1_wholefly_dataset" / "data.yaml")   # dataset ships next to this script
RUN_NAME     = "fly_wholefly_seg"

print("\n--- Training YOLO26m-seg (whole fly) ---")
model = YOLO("yolo26m-seg.pt")
 # model training pipeline
model.train(
    data        = DATASET_YAML,
    epochs      = 100,
    imgsz       = 1280,       
    batch       = 8,         
    device      = 0,
    name        = RUN_NAME,
    project     = f"{WORKSPACE}/runs",
#  augmentation pipeline
    flipud      = 0.5,
    fliplr      = 0.5,
    degrees     = 180,      
    translate   = 0.1,
    scale       = 0.5,
    mosaic      = 1.0,        
    close_mosaic = 10,        
    patience    = 50,
    save        = True,
    plots       = True,
)

best = f"{WORKSPACE}/runs/{RUN_NAME}/weights/best.pt"
print(f"\nTraining complete. Weights at: {best}")
print("\nDownload to your Mac (run locally):")
print(f'  scp -P <port> root@<ip>:{best} '
      f'"./layer1_wholefly_seg.pt"')

print("\n--- Evaluating on val split ---")
YOLO(best).val(data=DATASET_YAML, imgsz=1280, device=0)
