# Custom Object Detection training, exporting, inference pipeline
This is a custom project to work with D-FINE - state of the art object detection model based on RT-DETR.

## Configuration
Check config.yaml for configs

## Usage
To run the scripts, use the following commands:
```bash
python -m src.etl.preprocess   # Converts images and PDFs to JPG format
python -m src.etl.split        # Creates train, validation, and test CSVs with image paths
python -m src.dl.train         # Runs the training pipeline
python -m src.dl.export        # Exports weights in various formats after training
python -m src.dl.bench         # Runs all exported models on the test set

python -m src.dl.subm          # Creates a submition file for DL Enigma Kaggle challenge
```

## Inference
Use inference classes in `src/infer`. Currently available:
- Torch
- TensorRT

## Outputs
- **Models**: Saved during the training process and export at `output/models/exp_name_date`. Includes training logs, table with main metrics, confusion matrics, f1-score_vs_threshold and precisino_recall_vs_threshold.
- **Debug images**: Preprocessed images (including augmentations) are saved at `output/debug_images/split` as they are fed into the model (except for normalization).
- **Evaluation predicts**: Visualised model's predictions on val set. Includes GT as green and preds as blue.
- **Bench images**: Visualised model's predictions with inference class. Uses all exported models

## Results examples
**Train**

![image](assets/train.png)

**Benchmarking**

![image](assets/bench.png)

**WandB**

![image](assets/wandb.png)


## Features
- Training pipeline from SoTA D-FINE model
- Export to ONNX, OpenVino, TensorRT. Half precision included
- Inference class for Torch and TensorRT

- Augs based on the [albumentations](https://albumentations.ai) lib
- Mosaic augmentation
- Metrics: mAPs, Precision, Recall, F1-score, Confusion matrix, IoU, plots
- After training is done - runs a test to calculate the optimal conf threshold
- Exponential moving average model
- Batch accumulation
- Automatic mixed precision (40% less vRAM used and 15% faster training)
- Gradient clipping
- Keep ratio of the image and use paddings or use simple resize
- When ratio is kept, inference can be sped up with removal of grey paddings
- Visualisation of preprocessed images, model predictions and ground truth
- Warmup epochs to ignore background images for easier start of convirsion
- OneCycler used as scheduler, AdamW as optimizer
- Unified configuration file for all scrips
- Annotations in YOLO format, splits in csv format
- ETA displayed during training, precise strating epoch 2
- Logging file with training process
- WandB integration

## TODO
- Batch inference
- Add other model sizes
- Add ONNX and OpenVino inference classes
- Fix visualizations when ratio is kept
- Finetune with layers freeze
- Add support for cashing in dataset
- Add support for multi GPU training
- Instance segmentation
- Smart dataset preprocessing. Detect small objects. Detect near duplicates (remove from val/test)


## Acknowledgement
This project is built upon original [D-FINE repo](https://github.com/Peterande/D-FINE). Thank you to the D-FINE team for an awesome model!

``` bibtex
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
