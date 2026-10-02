# Explicit Modeling of Multi-View Observation Consistency for Supervised Multi-View Stereo

**Liangliang Li, Guihua Liu, Feng Xu**

**Official implementation of MVOCO-MVSNet and MVOCO-MVSNet+**

We propose **MVOCO (Multi-View Observation Consistency Optimization)**, a Bayesian framework for supervised Multi-View Stereo (MVS) based on the Maximum A Posteriori (MAP) principle. Furthermore, we introduce **Differentiable Spatial Encoding (DSE)** to enhance spatial-aware feature representation for MVS. Based on these designs, we develop **MVOCO-MVSNet** and **MVOCO-MVSNet+** for accurate and complete multi-view 3D reconstruction.

---

## 🔨 Setup

### 1.1 Requirements

Use the following commands to create the `conda` environment:

```bash
conda create -n mvocomvsnet python=3.12
conda activate mvocomvsnet
pip install -r requirements.txt
```

### 1.2 Datasets

Download the following datasets and modify the corresponding local paths in `scripts/data_path.sh`.

#### DTU Dataset

**Training data.** We use the same DTU training data as MVSNet and CasMVSNet. Please refer to the [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).

After downloading, unzip and organize the dataset as follows:

```text
dtu/
├── Cameras
├── Depths
├── Depths_raw
└── Rectified
```

**Testing data.** For convenience, we use the [DTU testing data](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view?usp=sharing) processed by CVP-MVSNet.

After downloading, unzip and organize the dataset as follows:

```text
dtu-test/
├── Cameras
├── Depths
└── Rectified
```

> **Note:** The images and lighting conditions are consistent with those of the original DTU dataset.

#### BlendedMVS Dataset

Download the low-resolution version of the [BlendedMVS dataset](https://drive.google.com/file/d/1ilxls-VJNvB7IaFj7P0ehMPr7ikRCb/view) and unzip it as follows:

```text
blendedmvs/
└── dataset_low_res
    ├── ...
    └── 5c34529873a8df509ae57b58
```

#### Tanks and Temples Dataset

Download the Intermediate and Advanced subsets of the [Tanks and Temples dataset](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and unzip them.

For the Intermediate subset, if you want to use the short-range version of the camera parameters, unzip `short_range_caemeras_for_mvsnet.zip` and move the corresponding `cam_[]` files to their respective scenarios.

The expected directory structure is:

```text
tnt/
├── advanced
│   ├── ...
│   └── Temple
│       ├── cams
│       ├── images
│       ├── pair.txt
│       └── Temple.log
└── intermediate
    ├── ...
    └── Train
        ├── cams
        ├── cams_train
        ├── images
        ├── pair.txt
        └── Train.log
```

---

## 🚂 Training

> **⚠️ Checkpoint Selection for Reproducibility**
>
> During training, checkpoint selection should be based on a joint consideration of **absolute depth error (`abs_depth_err`)**, **reprojection error (`epe`)**, **2 mm error (`2mm_err`)**, **4 mm error (`4mm_err`)**, and **8 mm error (`8mm_err`)**. We do not recommend selecting a checkpoint based on any single metric alone.
>
> As a practical reference, for the **DTU dataset**, we generally select a checkpoint around **epoch 13**. For the **Tanks & Temples dataset**, we generally use the **last training epoch**.
>
> The exact optimal epoch may vary slightly with the training configuration and random seed.

You can train **MVOCO-MVSNet and MVOCO-MVSNet+** from scratch on the DTU and BlendedMVS datasets.

After training, the generated checkpoints and logs will be saved in the `checkpoints` directory. The main output files include:

* `events.out.tfevents*`: TensorBoard logs for monitoring the training process.
* `model_[epoch].ckpt`: Model checkpoints saved according to `--save_freq`.
* `train-[TIME].log`: Detailed training logs for monitoring and evaluating the training process.

### 2.1 DTU

To train **MVOCO-MVSNet** on the DTU dataset, refer to:

```text
scripts/dtu/train_dtu.sh
scripts/dtu/train_dtu_plus.sh
```

Modify `THISNAME`, `batch_size`, and other parameters according to your requirements.

Then run the corresponding script:

```bash
bash scripts/dtu/train_dtu.sh
```

or

```bash
bash scripts/dtu/train_dtu_plus.sh
```

You can configure the model parameters and other hyperparameters in `models/utils/opts` according to your specific requirements.

Then, run the corresponding training script in the terminal:

```bash
python train.py
```

or

```bash
python train_reg.py
```

> **Note:** When the weight of the geometric observation term is below **3.0**, we recommend using `train_reg.py` to achieve better performance.

### 2.2 BlendedMVS

To train **MVOCO-MVSNet** on the BlendedMVS dataset, refer to:

```text
scripts/blend/train_blend.sh
```

Modify `THISNAME`, `batch_size`, and other parameters according to your requirements.

Then run:

```bash
bash scripts/blend/train_blend.sh
```

By default, we use **9 viewpoints** as input during BlendedMVS training.

Similarly, you can configure the model parameters and other hyperparameters in `models/utils/opts` according to your specific requirements.

Then, run:

```bash
python train.py
```

or

```bash
python train_reg.py
```

> **Note:** When the weight of the geometric observation term is below **3.0**, we recommend using `train_reg.py` to achieve better performance.

---

## ⚗️ Testing

### 3.1 DTU

For DTU testing, we use models trained on the DTU training dataset. We provide several pre-trained checkpoints corresponding to the models reported in our paper:

| Checkpoint                       | Model                           | Testing Script          | fusions—DTU                      |
| -------------------------------- | ------------------------------- | ----------------------- | ---------------------------------|
| `dtu_mvoco.ckpt`                 | MVOCO-MVSNet                    | `test.py`               | `fusions/dtu/mvoco.py`           |
| `dtu_mvoco_plus.ckpt`            | MVOCO-MVSNet+                   | `test.py`               | `fusions/dtu/mvocoplus.py`       |
| `model_A_paper.ckpt`             | Ablation Model A                | `test_reg.py`           | `fusions/dtu/mvoco_modelA.py`    |
| `model_B_paper.ckpt`             | Ablation Model B                | `test_reg.py`           | `fusions/dtu/mvoco_modelB.py`    |
| `dtu_real_depth_mvoco_plus.ckpt` | Ablation Model S                | `test.py`               | `fusions/dtu/mvoco_modelS.py`    |
| `dtu_Casmvsnet_mvoco.ckpt`       | CasMVSNet integrated with MVOCO | CasMVSNet official code | `fusions/dtu/mvoco_casmvsnet.py` |

Please specify the corresponding checkpoint in the configuration file before testing.

The testing process consists of **depth map estimation, point cloud fusion, and result evaluation**, as described below.

#### Step 1: Depth Map Estimation

Run:

```bash
bash scripts/dtu/test_dtu.sh
```

The estimated depth maps and confidence maps will be stored in:

```text
outputs/dtu/[THISNAME]/
```

Each scan folder contains files such as:

```text
depth_est/
confidence/
...
```

#### Step 2: Point Cloud Fusion

Run:

```bash
bash scripts/dtu/fusion_dtu.sh
```

We provide three different point cloud fusion methods. The `open3d` option is recommended by default.

After fusion, the resulting point clouds will be stored in:

```text
[FUSION_METHOD]_fusion_plys/
```

under the corresponding experiment output directory. The point cloud of each testing scan is stored in this directory.

<details>
<summary>(Optional) Using the "Gipuma" fusion method</summary>

1. Clone the [edited Fusibile repository](https://github.com/YoYo000/fusibile).

2. Refer to the [Fusibile configuration guide (Chinese)](https://zhuanlan.zhihu.com/p/460212787) for compilation details.

3. Create a Python 2.7 conda environment:

```bash
conda create -n fusibile python=2.7
conda install scipy matplotlib
conda install tensorflow==1.14.0
conda install -c https://conda.anaconda.org/menpo opencv
```

4. Activate the `fusibile` environment when using the `gipuma` fusion method.

</details>

#### Step 3: DTU Evaluation

Download the DTU ground-truth point clouds, including [ObsMask](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip), from the official website.

Organize the evaluation data as follows:

```text
dtu-evaluation/
├── ObsMask
└── Points
```

> **Note:** The checkpoint `dtu_Casmvsnet_mvoco.ckpt` is obtained by integrating MVOCO into the original CasMVSNet framework. To evaluate this checkpoint, please download the official [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet) source code and follow its original testing pipeline.

### 3.2 Tanks and Temples

For testing on the [Tanks and Temples benchmark](https://www.tanksandtemples.org/leaderboard/), you can use any of the following training configurations:

* Train only on the DTU training dataset.
* Train only on the BlendedMVS dataset.
* Pre-train on the DTU training dataset and fine-tune on the BlendedMVS dataset. **(Recommended)**

After training, follow the steps below.

#### Step 1: Depth Map Estimation

Run:

```bash
bash scripts/tnt/test_tnt.sh
```

The estimated results will be stored in:

```bash
python ./fusions/tnt/dypcd_dtnt.py   [**or other fusions method ./fusions/tnt/*******.py]
```

```text
outputs/[TRAINING_DATASET]/[THISNAME]/
```

You can use `outputs/visual.ipynb` for depth map visualization.

#### Step 2: Point Cloud Fusion

Run:

```bash
bash scripts/tnt/fusion_tnt.sh
```

We provide the commonly used dynamic fusion strategy. The fusion threshold can be adjusted in:

```text
fusions/tnt/dypcd.py
```

#### Step 3: Online Evaluation

Follow the *Upload Instructions* provided on the [official Tanks and Temples website](https://www.tanksandtemples.org/submit/) to submit the reconstructed point clouds for online evaluation.

### 3.3 Custom Data (TODO)

**MVOCO-MVSNet** can also be used for reconstruction on custom datasets.

Currently, you can refer to [MVSNet](https://github.com/YoYo000/MVSNet#file-formats) for the required data organization and follow the same procedures described above for **depth estimation** and **point cloud fusion**.

---

## 💡 Results

Our quantitative results on the DTU and Tanks and Temples datasets are shown below.

### DTU Dataset

| Method        | Acc. ↓ | Comp. ↓ | Overall ↓ |
| ------------- | ------ | ------- | --------- |
| MVOCO-MVSNet  | 0.344  | 0.243   | 0.294     |
| MVOCO-MVSNet+ | 0.328  | 0.244   | 0.286     |

### Tanks and Temples — Intermediate Set

| Method        | Mean ↑ | Family | Francis | Horse | Lighthouse | M60   | Panther | Playground | Train |
| ------------- | ------ | ------ | ------- | ----- | ---------- | ----- | ------- | ---------- | ----- |
| MVOCO-MVSNet+ | 65.30  | 82.29  | 68.37   | 55.61 | 67.34      | 64.03 | 63.47   | 61.52      | 59.78 |

### Tanks and Temples — Advanced Set

| Method        | Mean ↑ | Auditorium | Ballroom | Courtroom | Museum | Palace | Temple |
| ------------- | ------ | ---------- | -------- | --------- | ------ | ------ | ------ |
| MVOCO-MVSNet+ | 41.84  | 30.96      | 46.02    | 40.30     | 51.98  | 35.97  | 45.80  |

---

## 🖼️ Visualization Results on the Tanks and Temples Dataset

### Qualitative Results on the Intermediate Set

<p align="center">
  <img src="tnt_intermediate_visualization.png" width="900">
  <br>
  <em>Figure 1: MVOCO-MVSNet+ reconstruction results on the Intermediate Set of the Tanks and Temples dataset.</em>
</p>

### Qualitative Results on the Advanced Set

<p align="center">
  <img src="tnt_advanced_visualization.png" width="900">
  <br>
  <em>Figure 2: MVOCO-MVSNet+ reconstruction results on the Advanced Set of the Tanks and Temples dataset.</em>
</p>

---

## 👩 Acknowledgements

We would like to thank the authors of the following open-source projects for their valuable contributions:

* [MVSNet](https://github.com/YoYo000/MVSNet)
* [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch)
* [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet)
* [ET-MVSNet](https://github.com/TQTQliu/ET-MVSNet)
* [CL-MVSNet](https://KaiqiangXiong.github.io/CL-MVSNet)
