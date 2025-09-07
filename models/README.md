# Small Objects Detection Model Zoo

[Link to models stored on Kaggle](https://www.kaggle.com/models/jakubszpunar/small-objects-detection-benchmark-models)
---
This folder contains trained detectors used in the small objects detection benchmark trained on the [SkyFusion dataset](https://www.kaggle.com/datasets/kailaspsudheer/tiny-object-detection). These files are intended for use in Kaggle notebooks or locally. On Kaggle, place the files in your working directory or reference them from your Dataset path (e.g., `/kaggle/input/small-objects-detection-benchmark/models/<model-name>.pt`).

## Available Models (self-contained thresholds)

| Name                         | Type         | File                                       | Framework                         | Optimal params                                                                 |
|-----------------------------|--------------|--------------------------------------------|-----------------------------------|---------------------------------------------------------------------------------|
| yolov8m-aug-update_20250603 | yolo         | yolov8m-aug-update_20250603.pt             | Ultralytics (YOLOv8)              | conf_thr=0.06291186979342091, nms_iou=0.3358295483691517                        |
| yolov11m-p2-aug_20250603    | yolo         | yolov11m-p2-aug_20250603.pt                | Ultralytics (YOLOv11)             | conf_thr=0.052566007120515956, nms_iou=0.49317179138811856                      |
| rf-detr                     | rfdetr       | rfdetr_best_total.pth                      | PyTorch (RF-DETR impl)            | conf_thr=0.09616820140192325                                                    |
| faster-rcnn                 | faster-rcnn  | fasterrcnn-best-epoch=18-val_map=0.31.ckpt | PyTorch Lightning + torchvision   | conf_thr=0.07957236023833904, nms_iou=0.621230971215935                         |
| rt-detr                     | rtdetr       | rtdetr-aug_best.pt                          | Ultralytics (RT-DETR)             | conf_thr=0.2704984199324548                                                     |

Notes:
- The table includes the exact optimal thresholds; you don’t need an external YAML file.
- If a model doesn’t list `nms_iou`, use the framework’s default or your benchmark’s setting.

---

## Ultralytics (YOLOv8 / YOLOv11)

Install on Kaggle:
```bash
pip install -q ultralytics
```

YOLOv11 example:
```python
from ultralytics import YOLO

model = YOLO("yolov11m-p2-aug_20250603.pt")
conf_thr = 0.052566007120515956
nms_iou  = 0.49317179138811856

results = model("image.jpg", conf=conf_thr, iou=nms_iou)
# results[0].plot()  # visualize if needed
```

YOLOv8 example:
```python
from ultralytics import YOLO

model = YOLO("yolov8m-aug-update_20250603.pt")
conf_thr = 0.06291186979342091
nms_iou  = 0.3358295483691517

results = model("image.jpg", conf=conf_thr, iou=nms_iou)
```

---

## Ultralytics RT-DETR

```python
# Depending on ultralytics version:
from ultralytics import RTDETR, YOLO

# Preferred:
model = RTDETR("rtdetr-aug_best.pt")

# Fallback (unified loader):
# model = YOLO("rtdetr-aug_best.pt")

conf_thr = 0.2704984199324548
results = model("image.jpg", conf=conf_thr)  # iou optional for RT-DETR
```

---

## Faster R-CNN (PyTorch Lightning)

The checkpoint is a Lightning `.ckpt`. Replace `MyFasterRCNNModule` with your module class.

```bash
pip install -q lightning
```

```python
import torch
from lightning import LightningModule

class MyFasterRCNNModule(LightningModule):
    def __init__(self, ...):
        super().__init__()
        # define model here (e.g., torchvision.models.detection.fasterrcnn_resnet50_fpn)
        ...

ckpt_path = "fasterrcnn-best-epoch=18-val_map=0.31.ckpt"
model = MyFasterRCNNModule.load_from_checkpoint(ckpt_path, strict=False)
model.eval()

# Implement your preprocessing/postprocessing for inference
# outputs = model(images)
```

If `.ckpt` stores a plain `state_dict`:
```python
import torch

state = torch.load("fasterrcnn-best-epoch=18-val_map=0.31.ckpt", map_location="cpu")
model = build_faster_rcnn_model()  # your builder returning an nn.Module
model.load_state_dict(state.get("state_dict", state), strict=False)
model.eval()
```

Recommended inference thresholds:
- conf_thr: 0.07957236023833904
- nms_iou: 0.621230971215935

---

## RF-DETR (PyTorch)

```bash
pip install -q "rfdetr[metrics]"
```

```python
from rfdetr import RFDETRBase
from PIL import Image

# Initialize with pretrained weights at construction (matches adapter)
model = RFDETRBase(
    device="cuda",  # or "cpu"
    pretrain_weights="rfdetr_best_total.pth",
)

# Inference on a single image using the optimal threshold
threshold = 0.09616820140192325  # optimal conf_thr
pred = model.predict("image.jpg", threshold=threshold)

# Alternatively, pass a PIL image (adapter converts BGR->RGB then uses PIL)
# img = Image.open("image.jpg").convert("RGB")
# pred = model.predict(img, threshold=threshold)
```

