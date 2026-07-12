'''
this is the stage two build of fly bodyparts including head, throax and abdomen made by christian. 2026

'''
import subprocess, sys
from pathlib import Path

def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip("ultralytics==8.4.52")   # pinned for reproducibility; matches the verified run

from ultralytics import YOLO

HERE         = Path(__file__).resolve().parent
WORKSPACE    = "/workspace"                                     # Vast.ai run/output dir
DATASET_YAML = str(HERE / "stage2_bodyparts_dataset" / "dataset.yaml")  # dataset ships next to this script
RUN_NAME     = "fly_bodyparts_seg"

print("\n--- Training YOLO26m-seg (body parts) ---")
model = YOLO("yolo26m-seg.pt")

model.train(
    data        = DATASET_YAML,
    epochs      = 200,
    imgsz       = 256,       
    batch       = 16,       #depends on gpu used 
    device      = 0,
    name        = RUN_NAME,
    project     = f"{WORKSPACE}/runs",
    # augmentation
    flipud      = 0.5,
    fliplr      = 0.5,
    degrees     = 180,      
    translate   = 0.1,
    scale       = 0.3,
    mosaic      = 0.0,      
    patience    = 50,
    save        = True,
    plots       = True,
)

best = f"{WORKSPACE}/runs/{RUN_NAME}/weights/best.pt"
print(f"\nTraining complete. Weights at: {best}")
print("\nDownload to your Mac (run locally):")
print(f'  scp -P <port> root@<ip>:{best} '
      f'"./layer2_bodyparts_seg.pt"')

print("\n--- Evaluating on val split ---")
YOLO(best).val(data=DATASET_YAML, imgsz=256, device=0)
