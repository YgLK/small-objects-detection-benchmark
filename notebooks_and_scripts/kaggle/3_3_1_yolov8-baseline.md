---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python papermill={"duration": 87.408198, "end_time": "2025-05-06T18:39:38.870416", "exception": false, "start_time": "2025-05-06T18:38:11.462218", "status": "completed"}
pip install --upgrade ultralytics==8.0.238 wandb
```

```python papermill={"duration": 11.778827, "end_time": "2025-05-06T18:39:50.671441", "exception": false, "start_time": "2025-05-06T18:39:38.892614", "status": "completed"}
pip install ray[tune]==2.9.3
```

```python papermill={"duration": 0.430001, "end_time": "2025-05-06T18:39:51.126154", "exception": false, "start_time": "2025-05-06T18:39:50.696153", "status": "completed"}
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
```

```python papermill={"duration": 2.592081, "end_time": "2025-05-06T18:39:53.742208", "exception": false, "start_time": "2025-05-06T18:39:51.150127", "status": "completed"}
import wandb
# in ENV variables api key for wandb is stored
wandb.login(key=wandb_api_key)
```

```python papermill={"duration": 8.859979, "end_time": "2025-05-06T18:40:02.626819", "exception": false, "start_time": "2025-05-06T18:39:53.766840", "status": "completed"}
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


def run_baseline_for_all_models(data_path: str, epochs: int = 5, img_size: int = 640) -> None:
    """
    Runs YOLOv8 baseline training for all model sizes and logs to Weights & Biases.

    :param data_path: path to the dataset config YAML file
    :param epochs: number of training epochs
    :param img_size: input image size
    """
    model_sizes = ["n", "s", "m", "l", "x"]  # YOLOv8 model sizes
    
    for size in model_sizes:
        model_name = f"yolov8{size}.pt"

        # Initialize a new W&B run for each model size
        wandb.init(project="object-detection-small", name=f"yolov8{size}-baseline", job_type="baseline")

        # load the model
        model = YOLO(model_name)

        # add wandb callback
        add_wandb_callback(model, enable_model_checkpointing=True)

        # train the model
        model.train(
            data=data_path, 
            epochs=epochs, 
            imgsz=img_size, 
            seed=42,
            batch=4
        )

        # # validate the model
        # model.val(data=data_path, split='test')

        # finish the W&B run
        wandb.finish()
```

<!-- #region papermill={"duration": 0.069425, "end_time": "2025-05-06T18:40:02.720635", "exception": false, "start_time": "2025-05-06T18:40:02.651210", "status": "completed"} -->
**Objective:** <br>
Quickly benchmark YOLOv8 sizes (n, s, m, l, x).

Dataset has ~3,000 images (a small to moderate size).

Identify the best size–performance trade-off.

**Recommendation:** <br>
Run the baseline for 5 epochs initially:
* Fast enough to highlight major differences in performance.
* With 3k images, models should start showing distinct patterns even with limited training.
* Saves compute and time, quick exploration.

**Then:** <br>
Pick the best-performing model (e.g., best mAP50 per size, or best mAP per inference time/size trade-off). Retrain that one for full training cycle with more tuning (e.g., learning rate, dataset augmentations).


<!-- #endregion -->

```python papermill={"duration": 2408.300937, "end_time": "2025-05-06T19:20:11.045280", "exception": false, "start_time": "2025-05-06T18:40:02.744343", "status": "completed"}
data_file_path = "/kaggle/input/skyfusion-yolo-v8/SkyFusion_yolo/dataset.yaml"
run_baseline_for_all_models(data_file_path, epochs=5)
```
