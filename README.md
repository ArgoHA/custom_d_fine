# D-FINE-seg Object Detection and Segmentation Framework (Train, Export, Inference)

This is a framework that implements [D-FINE](https://arxiv.org/abs/2410.13842) architecture for object detection and adds a segmentation head, so you can train an object detection task or instance segmentation task. Detection architecture and loss used from the original [repo](https://github.com/Peterande/D-FINE), everything else was developed from scratch, this is not a fork.

Check out [the video tutorial](https://youtu.be/_uEyRRw4miY) to get familiar with this framework.

## Introducing Instance Segmentation

This goes beyond the original paper and is developed specifically for this framework. Segmentation task is still an early feature and there are no pretrained weights for segentation head yet. Mosaic augmentation is not recommended for segmentation at this moment.

MaskDecoder takes FPN (preserving fine spatial details for segmentation) and PAN (enriched with both local and global context) embeddings from HybridEncoder. Outputs H/4 masks. Number of masks = number of detected objects. Postprocessing filters out pixels outside of the corresponding detection bounding box.

Note: mAP values calculated are lower then the real ones. It's done to reduce RAM usage and remove low confidence masks (which affect mAP scores).

## Main scripts

To run the scripts, use the following commands:

```bash
make split          # Creates train, validation, and test CSVs with image paths
make train          # Runs the training pipeline, including DDP version
make export         # Exports weights in various formats after training

make bench          # Runs all exported models on the test set
make infer          # Runs model ontest folder, saves visualisations and txt preds
make check_errors   # Runs model on train and val sets, saves only missmatched boxes with GT
make test_batching  # Gets stats to find the optimal batch size for your model and GPU
make ov_int8        # Runs int8 accuracy aware quantization for OpenVINO. Can take several hours
```

Note: if you want to pass parameters, you can run any of these scripts with `python -m src.dl script_name` (use `etl` instead of `dl` for `preprocess` and `split`), You can also just run `make` to run `preprocess, split, train, export, bench` scripts as 1 sequence.

For **DDP training** just set train.ddp.enabled to True, pick number of GPUs and run `make train` as usual.

## Usage example

0. `git clone https://github.com/ArgoHA/D-FINE-seg.git`
1. For bigger models (l, x) download from [gdrive](https://drive.google.com/drive/folders/1cjfMS_YV5LcoJsYi-fy0HWBZQU6eeP-7?usp=share_link) andput into `pretrained` folder
2. Prepare your data: `images` folder and `labels` folder - txt file per image in YOLO format.
3. Customize `config.yaml`, minimal example:
      - `task`. Set to `segment` to enable Segmentation head.
      - `exp_name`. This is experiment name which is used in model's output folder. After you train a model, you can run export/bench/infer and it will use the model under this name + current date.
      - `root`. Path to the directory where you store your dataset and where model outputs will be saved
      - `data_path`. Path to the folder with `images` and `labels`
      - `label_to_name`. Your custom dataset classes
      - `model_name`. Choose from n/s/m/l/x model sizes.
      - and usual things like: epochs, batch_size, num_workers. Check out config.yaml for all configs.
4. Run `preprocess` and `split` scripts from d_fine_seg repo.
5. Run `train` script, changing confurations, iterating, untill you get desired results.
6. Run `export`script to create ONNX, TensorRT, OpenVINO models.

[Training example with Colab](https://colab.research.google.com/drive/1ZV12qnUQMpC0g3j-0G-tYhmmdM98a41X?usp=sharing)

If you run train script passing the args in the command and not changing them in the config file - you should also pass changed args to other scripts like `export` or `infer`. Example:

```bash
python -m src.dl.train exp_name=my_experiment
python -m src.dl.export exp_name=my_experiment
```

## Labels format

We use YOLO labels format. One txt file per image (with the same stem). One row = one object.

```
📂 data/dataset
├── 📁 images
├── 📁 labels
```

**Detection**: [class_id, xc, yc, w, h], coords normalized

**Segmentation**: [class_id, xy, xy, ...], coords normalized. Length = number of points + 1

## Exporting tips

TensorRT export must be done on the GPU that you are going to use for inferencing.

Half precision:

- usually makes inference faster with minimum accuracy suffering
- works best with TensorRT and OpenVINO (when running on GPU cores). OpenVINO can be exported ones and then can be inferenced in both fp32 or fp16. Note on Apple Silicon right now OpenVINO version of D-FINE works only in full precision.
- Not used for ONNX and Torch at the moment.

Dynamic input means that during inference, we cut black paddings from letterbox. I don't recommend using it with D-FINE as accuracy degrades too much (probably because absolute Positional Encoding of patches)

## Inference

Use inference classes in `src/infer`. Currently available:

- Torch
- TensorRT
- OpenVINO
- ONNX

You can run inference on a folder (path_to_test_data) of images or on a folder of videos. Crops will be created automatically. You can control it and paddings from config.yaml in the `infer` section.

## Performace benchmarks

All benchmarks below are on the same **custom dataset** with **D-FINEm** at **640×640**.
Latency numbers include image preprocessing -> model inference -> postprocessing.

### Desktop: Intel i5-12400F + RTX 5070 Ti

```
+----------------------+--------------+--------------+
| Format               |   F1 score   | Latency (ms) |
+----------------------+--------------+--------------+
| Torch, FP32, GPU     |    0.9161    |    16.6      |
| TensorRT, FP32, GPU  |    0.9166    |    7.5       |
| TensorRT, FP16, GPU  |    0.9167    |    5.5       |
| OpenVINO, FP32, CPU  |    0.9165    |    115.4     |
| OpenVINO, FP16, CPU  |    0.9165    |    115.4     |
| OpenVINO, INT8, CPU  |    0.9139    |    44.1      |
| ONNX, FP32, CPU      |    0.9165    |    150.6     |
+----------------------+--------------+--------------+
```

**Notes (desktop):**

- TensorRT FP16 gives ~**3x speedup** vs Torch FP32 GPU with **no meaningful F1 drop**.
- On the CPU, OpenVINO seems to ignore FP16 - it's identical to FP32.
- OpenVINO INT8 on CPU gives ~**2.6x speedup** vs FP32 with a **small F1 drop** on this particular dataset.

---

### Edge device: Intel N150 (CPU with iGPU cores)

```
+----------------------+--------------+--------------+
| Format               |   F1 score   | Latency (ms) |
+----------------------+--------------+--------------+
| OpenVINO, FP32, iGPU |    0.9165    |     350.8    |
| OpenVINO, FP16, iGPU |    0.9157    |     209.6    |
| OpenVINO, INT8, iGPU |    0.9116    |     123.1    |
| OpenVINO, FP32, CPU  |    0.9165    |     505.2    |
| OpenVINO, FP16, CPU  |    0.9165    |     505.2    |
| OpenVINO, INT8, CPU  |    0.9139    |     252.7    |
+----------------------+--------------+--------------+
```

**Notes (edge / N150):**

- On the iGPU, FP16 and INT8 both give **significant latency reductions** with **minor F1 degradation**.
- On the CPU, FP16 again seems to be ignored, while INT8 still gives a solid speedup.

### How to interpret these numbers

- FP16 is often a great sweet spot on GPUs: same accuracy, noticeably faster inference.
- On CPUs, FP16 may or may not be accelerated, depending on the hardware.
- INT8 can give big speedups on both CPU and GPU, but the accuracy drop is highly data- and model-dependent.

I recommend always benchmarking on your own hardware and dataset.

## Batched inference

Another thing to check on your hardware and model is batch size when you run batched inference (to get higher throughput, losing overall service latency). For that you can simpli run `make test_batching`, it will run torch model with different batch sizes and calculate **throughput** (proccesed images per second) and **average latency (per image). For example, with Intel i5-12400F + RTX 5070 Ti and D-FINEm, ~4 is the optimal batch size to inference with Torch.

```
+------+------------+-------------------+
|  bs  | throughput | latency_per_image |
+------+------------+-------------------+
| 1.0  |    76.4    |       13.1        |
| 2.0  |   113.4    |        8.8        |
| 4.0  |   138.1    |        7.2        |
| 8.0  |   122.7    |        8.1        |
| 16.0 |   119.7    |        8.4        |
| 32.0 |   117.8    |        8.5        |
+------+------------+-------------------+
```

## Outputs

- **Models**: Saved during the training process and export at `output/models/exp_name_date`. Includes training logs, table with main metrics, confusion matrics, f1-score_vs_threshold and precisino_recall_vs_threshold. In extended_metrics you can file per class metrics (saved during final eval after all epochs)
- **Debug images**: Preprocessed images (including augmentations) are saved at `output/debug_images/split` as they are fed into the model (except for normalization).
- **Evaluation predicts**: Visualised model's predictions on val set. Includes GT as green and preds as blue.
- **Bench images**: Visualised model's predictions with inference class. Uses all exported models
- **Infer**: Visualised model's predictions and predicted annotations in yolo txt format
- **Check errors**: Creats a folder check_errors with FP and FN bboxes only. Used to check model's errors on training and val sets and to find mislabelled samples.
- **Test batching**: Csv file with all tested batch sizes and latency

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
- Instance Segmentation task.
- Export to ONNX, OpenVino, TensorRT.
- Inference class for Torch, TensorRT, OpenVINO on images or videos
- Label smoothing in Focal loss
- Augs based on the [albumentations](https://albumentations.ai) lib
- Mosaic augmentation, multiscale aug
- Metrics: mAPs, Precision, Recall, F1-score, Confusion matrix, IoU, plots
- Distributed Data Parallel (DDP) training
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
- Gradio UI demo

## TODO

- Finetune with layers freeze
- Add support for cashing in dataset
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
