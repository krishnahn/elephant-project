from ultralytics import YOLO
from pathlib import Path
import shutil

DATASET = '/content/dataset/elephant-dataset-yolov/dataset.yaml'
CHECKPOINT_DIR = Path('/content/elephant_models')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Train
model = YOLO('yolo11n.pt')
model.train(
    data=DATASET,
    epochs=30,
    batch=64,
    imgsz=640,
    device='cuda',
    project=str(CHECKPOINT_DIR),
    name='elephant_training'
)

# Copy best model
best_weights = CHECKPOINT_DIR / 'elephant_training/weights/best.pt'
final_model = CHECKPOINT_DIR / 'elephant_yolo11_best.pt'
shutil.copy(best_weights, final_model)

# Validate
best_model = YOLO(str(final_model))
metrics = best_model.val(data=DATASET)

print(f"\nmAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
print(f"\nModel saved: {final_model}")
