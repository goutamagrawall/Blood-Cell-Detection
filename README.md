# Blood Cell Detection using YOLOv8

Detect and count red blood cells (RBC), white blood cells (WBC), and platelets in microscope images. Includes a runnable command-line interface, a training notebook, the annotated dataset, seven trained checkpoints, and experiment results.

## Setup

Use Python 3.10 or newer (verified with Python 3.12 on Windows).

```powershell
git clone https://github.com/goutamagrawall/Blood-Cell-Detection.git
cd Blood-Cell-Detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

For a CPU-only PyTorch installation, install the following before the requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run detection and counting

The included balanced-loss checkpoint is used by default. No training or model download is required for inference.

```powershell
.\.venv\Scripts\python.exe main.py check
.\.venv\Scripts\python.exe main.py predict --source blood_cell_dataset/test/images
.\.venv\Scripts\python.exe main.py predict --source path/to/image.jpg --conf 0.25
```

Annotated images and per-image RBC/WBC/platelet counts are saved in `outputs/predict/`, including `counts.json`. Use `--output` to select a different directory. These paths are overwritten on a repeated run, so select a new output directory to preserve earlier results. Commands default to CPU; use `--device 0` with a CUDA-enabled PyTorch installation for GPU execution.

## Evaluate and train

```powershell
.\.venv\Scripts\python.exe main.py val --split test --batch 4
.\.venv\Scripts\python.exe main.py train --epochs 30 --batch 8
```

Evaluation writes plots and `metrics.json` to `outputs/val/`. Training fine-tunes the supplied balanced-loss model and writes checkpoints to `outputs/train/`. To train from general YOLOv8 weights instead, download `yolov8n.pt` and pass `--model path/to/yolov8n.pt`. Pass `--data path/to/data.yaml` to use another dataset. Relative paths in dataset YAML files are resolved from the YAML file's directory. Run `python main.py --help` for commands.

To reproduce the original loss-weight and model-size comparisons, install `requirements-notebook.txt` and open `Blood Cell detect and count.ipynb` using the virtual environment's kernel. Start the notebook in this repository's root directory. Running all cells retrains seven experiments and may take substantial time; the existing checkpoints are sufficient for detection and evaluation. The notebook downloads pretrained weights when needed.

## Dataset

The included Roboflow dataset has three classes in this order: `RBC`, `WBC`, `platelets`.

| Split | Images |
| --- | ---: |
| Training | 966 |
| Validation | 94 |
| Test | 46 |
| Total | 1,106 |

Source: [Roboflow blood cell detection, version 4](https://universe.roboflow.com/clg-vtj9f/blood-cell-detection-bsbvn/dataset/4). The dataset's supplied metadata identifies its license as CC BY 4.0. Attribution is retained in `blood_cell_dataset/README.dataset.txt`, `README.roboflow.txt`, and `data.yaml`.

## Results

Fresh CPU evaluation of `blood_cell_experiments/balanced_loss/weights/best.pt` on all 46 test images (934 annotated cells), at image size 640:

| Metric | Value |
| --- | ---: |
| mAP@0.5 | 0.9318 |
| mAP@0.5:0.95 | 0.6295 |
| Precision | 0.8722 |
| Recall | 0.9309 |

See [VERIFICATION.md](VERIFICATION.md) for checks and limitations. These are dataset benchmark results; the project has not established clinical suitability. Speed depends on hardware and configuration.

The original experiment comparisons are retained in `blood_cell_experiments/results_comparison.csv` and `model_size_comparison.csv`. Historical timing measurements in those files are from the original training environment.

## Project layout

```text
main.py                              # check, predict, val, train commands
requirements.txt                     # runtime dependencies
requirements-notebook.txt            # additional notebook dependencies
Blood Cell detect and count.ipynb    # original experiment workflow, with portable setup
blood_cell_dataset/                  # images, labels, and portable data.yaml
blood_cell_experiments/              # best checkpoints, training logs, plots, comparisons
tests/test_main.py                   # dataset and counting regression tests
VERIFICATION.md                     # local verification report
```

Workstation settings, caches, duplicate ZIP data, downloadable root-level pretrained weights, and duplicate `last.pt` checkpoints are excluded from Git. Each experiment's `best.pt` remains included.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Acknowledgments

This project is based on the original [Blood-Cell-Detection project by Aaditya Rao](https://github.com/aadi883/Blood-Cell-Detection), with a portable runtime interface and verification added in this repository. It uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and the Roboflow dataset credited above. Refer to upstream projects and dataset metadata for their license terms.
