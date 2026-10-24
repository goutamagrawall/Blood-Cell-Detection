# Verification report

Verified on 24 October 2026 using Windows, Python 3.12.14, Ultralytics 8.3.221, PyTorch 2.14.1+cpu, and an AMD Ryzen 7 5700U CPU.

## Repairs

- Replaced the greeting-only main.py with commands for dataset verification, detection/counting, evaluation, and training.
- Replaced hardcoded Lightning AI dataset paths with portable paths. Runtime YAML files are written under outputs/.
- Added runtime and notebook dependency files, seven regression tests, and Git exclusions for copied workstation files and redundant artifacts.
- Made the notebook reuse the included dataset and write its runtime configuration outside the tracked dataset. Cleared previous notebook outputs.
- Updated the README to describe actual commands and benchmark results, retaining original author and dataset attribution.

## Executed checks

- Dataset verification passed: all 1,106 images decoded, each had a label file, and all annotations had valid class IDs and normalized finite box values. Split sizes: 966 training, 94 validation, 46 test.
- All seven regression tests passed: relative-path resolution, runtime configuration generation, valid dataset loading, invalid annotations, missing labels, corrupt images, model class mapping, and empty detections.
- CPU prediction completed on all 46 test images at image size 640, producing annotated images and counts.json.
- All seven supplied best.pt checkpoints loaded and completed sample inference with the expected three class names. All notebook code cells compiled successfully.
- Full test-set evaluation completed on 46 images with 934 annotated objects: 857 RBC, 47 WBC, and 30 platelets.
- Training smoke check completed: one epoch on four training images, with four validation images, image size 128, batch size 2, and CPU. New best.pt and last.pt files were written under outputs/training-smoke/. Original checkpoints were preserved.

## Fresh balanced-loss test metrics

| Metric | Value |
| --- | ---: |
| mAP@0.5 | 0.9317883326 |
| mAP@0.5:0.95 | 0.6295277689 |
| Precision | 0.8722265080 |
| Recall | 0.9308829249 |

Full local outputs are under outputs/predict/, outputs/val/, and outputs/training-smoke/ and are excluded from Git. Historical experiment plots and result tables remain under blood_cell_experiments/.

## Scope

The complete seven-experiment training notebook was not rerun. The training smoke check verifies the training pipeline; it does not reproduce a full 30-epoch experiment. GPU execution and clinical suitability were not evaluated.
