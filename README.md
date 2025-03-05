# Custom SoTA Object Detection D-FINE model training, exporting, inferencing pipeline
This is a custom project to work with [D-FINE](https://arxiv.org/abs/2410.13842) - state of the art object detection transformer based model. Model author's repo: [D-FINE](https://github.com/Peterande/D-FINE).

## Main scripts
To run the scripts, use the following commands:
```bash
python -m src.etl.preprocess    # Converts images and PDFs to JPG format
python -m src.etl.split         # Creates train, validation, and test CSVs with image paths
python -m src.dl.train          # Runs the training pipeline
python -m src.dl.export         # Exports weights in various formats after training
python -m src.dl.bench          # Runs all exported models on the test set
python -m src.dl.infer          # Runs model ontest folder, saves visualisations and txt preds
```

## Usage example
0. `git clone https://github.com/ArgoHA/custom_d_fine.git`
1. Prepare your data: `images` folder and `labels` folder (txt file per image in YOLO format).
2. Customize `config.yaml`, minimal example:
      - `exp_name`. This is experiment name which is used in model's output folder. After you train a model, you can run export/bench/infer and it will use the model under this name + current date.
      - `root`. Path to the directory where you store your dataset and where model outputs will be saved
      - `data_path`. Path to the folder with `images` and `labels`
      - `label_to_name`. Your custom dataset classes
      - `model_name`. Choose from n/s/m/l/x model sizes.
      - and usual things like: epochs, batch_size, num_workers. Check out config.yaml for all configs.
3. Run `preprocess` and split `scripts` from custom_d_fine repo.
4. Run `train` script, changing confurations, iterating, untill you get desired results.
5. Run `export`script to create ONNX, TensorRT, OpenVINO models.

## Inference
Use inference classes in `src/infer`. Currently available:
- Torch
- TensorRT
- OpenVINO

## Outputs
- **Models**: Saved during the training process and export at `output/models/exp_name_date`. Includes training logs, table with main metrics, confusion matrics, f1-score_vs_threshold and precisino_recall_vs_threshold.
- **Debug images**: Preprocessed images (including augmentations) are saved at `output/debug_images/split` as they are fed into the model (except for normalization).
- **Evaluation predicts**: Visualised model's predictions on val set. Includes GT as green and preds as blue.
- **Bench images**: Visualised model's predictions with inference class. Uses all exported models
- **Infer**: Visualised model's predictions and predicted annotations in yolo txt format

## Results examples
**Train**

![image](assets/train.png)

**Benchmarking**

![image](assets/bench.png)

**WandB**

![image](assets/wandb.png)

**Infer**

![image](assets/infer_high.jpg)

![image](assets/infer_water.jpg)


## Features
- Training pipeline from SoTA D-FINE model
- Export to ONNX, OpenVino, TensorRT.
- Inference class for Torch, TensorRT, OpenVINO

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
- Batch inference
- Early stopping

## TODO
- Implement multiscale aug
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
