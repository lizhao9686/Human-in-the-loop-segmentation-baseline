# GazeMedSeg 基础复现与技术审计报告

审计日期：2026-09-01  
审计对象：[med-air/GazeMedSeg](https://github.com/med-air/GazeMedSeg)，`main` commit [`12d02590a4b343a3b52de4fa7bd76cda659d4736`](https://github.com/med-air/GazeMedSeg/commit/12d02590a4b343a3b52de4fa7bd76cda659d4736)（2024-07-30）  
论文：[MICCAI 2024 main paper](https://papers.miccai.org/miccai-2024/paper/1675_paper.pdf)，[appendix](https://papers.miccai.org/miccai-2024/supp/1675_supp.pdf)

## 结论摘要

1. 发布代码实现的是“多个完整 2D MONAI BasicUNet + 不同 CRF-map 阈值 pseudo-mask + 跨 level consistency”。它没有 Vision-Language、text encoder、Mamba 或新的 loss；本次也未添加或修改这些内容。
2. 代码只有 `train` 和 `test`。训练期间每 `val_step=1000` 次迭代直接在 test set 上评估，并用 test Dice 选择 `model_best.pth`；这是明确的 test-set model-selection leakage。
3. 发布脚本/附录使用 SGD、lr `1e-2`、224×224、batch 8；论文正文却写 Adam、lr `4e-4`。README 将论文结果与 released result 分列。两套配置不能混称为同一次复现。
4. 两层 Kvasir 模型共有 62,768,770 个可训练参数。官方 batch 8、224²、FP32 时，仅 LPP 两层反向图中的两个 `unfold` 邻域张量静态下界约 19.14 GiB，尚未计两套 UNet 激活、梯度、optimizer state 和 workspace；RTX 3090 24GB 上有很高 OOM 风险。
5. 代码提供 `--fp16`，但 `trainers/gaze_sup.py::GazeSupTrainer._update` 在 `scaler.step()` 后漏掉 `scaler.update()`；该路径很可能在第二个 optimization step 失败。因此不能把“打开 FP16”视为当前原代码可直接使用的解决方案。
6. 官方源文件 29/29 与下载 archive 的 SHA-256 内容一致，本次未修改任何官方源文件。已完成 synthetic Kvasir 的 dataset、pseudo-label、dataloader、forward、loss、backward、SGD step、evaluation metric 和 checkpoint serialization 烟雾测试；真实 Kvasir/Gaze 数据未提供，因此没有、也不会伪造 Dice 复现结果。

---

## A. Repository Map

### 顶层入口

| 路径 | 职责 | 关键符号 |
|---|---|---|
| `run.py` | 唯一训练入口；初始化 run id/输出路径和 `pseudo_root`，设 seed，创建 train/test loader、model、optimizer、criterion、trainer，再执行训练 | `init_arguments`（23–45），`main`（48–86） |
| `parse_args.py` | CLI 参数定义 | `args_parser`（4–98） |
| `environment.yaml` | 官方 Linux Conda 环境锁定；Python 3.9.16、PyTorch 1.13.1、CUDA 11.6、MONAI 1.1.0 等 | 无代码符号 |
| `README.md` | 数据、gaze 预处理、实验脚本和 checkpoint 说明 | 注意第 28 行仍写旧仓库名 `GazeSup.git` |

### `configs/`

- `configs/static.py`：只保存 hierarchical level 配置。
  - `KVASIR_LEVEL_CONFIGS`：支持 1–5 levels；两层为 threshold `0.5`、`0.3`，两者 `compat=1`。
  - `PROSTATE_LEVEL_CONFIGS`：只定义两层；threshold `0.2`、`0.4`，`compat=1`。
- 没有 YAML/JSON experiment config；绝大多数实验设置来自 CLI/scripts。

### `datasets/`

- `datasets/__init__.py::get_dataloader(args, split)`：按 dataset、method、split 构造 dataset/DataLoader；训练 batch 为 CLI batch size、shuffle；非训练 batch 固定 1、不 shuffle。
- `datasets/base_dataset.py::BaseImageDataset`：公共 `Dataset` 基类。
  - `__getitem__`：`_fetch_data → transform → _transform_custom`，返回 `idx/subject_id/path` 与 tensor 字典。
  - `get_transform`：channel-first、normalization、resize、train-only horizontal/vertical random flip、tensor conversion。
- `datasets/kvasir_seg.py`
  - `KvasirSegDataset`：读取 JPEG RGB image 与 JPEG grayscale mask。
  - `KvasirGazeDataset`：额外读取 `gaze/crf_compat*/<id>.jpg`，resize 后按 level threshold 二值化。
- `datasets/nci_isbi.py`
  - `NCIISBIProstateDataset`：读取 DICOM pixel array 与 PNG mask。
  - `NCIISBIProstateGazeDataset`：额外读取 PNG CRF map 并按 level threshold 二值化。
- `datasets/transform.py::GaussianBlur/get_additional_transform`：当前主训练路径未调用；定义了额外 grayscale/blur augmentation，但 `get_additional_transform` 也没有被 dataset 使用。

### `models/`

- `models/__init__.py::get_model_opt`：只支持 `unet`；根据 method 创建单 UNet 或 `MultiLevelModel`，并创建 Adam/SGD。
- `models/unet.py::UNet`：包装 MONAI `BasicUNet`，返回 final decoder feature 与 logits。
- `models/multi_level.py`
  - `unfold_w_center`：用 `F.unfold` 创建 5×5、dilation 2 的局部邻域。
  - `MultiLevelModel`：保存多个完整 `UNet`；计算每层普通输出、LPP propagated feature/logits 和平均 logits。
  - `feature_propogate`：LPP 实现（函数名中的 `propogate` 是原代码拼写）。
  - `ensemble_weights/forward_ensemble`：权重平均模型，但训练/验证主路径未调用。

### `trainers/`

- `trainers/__init__.py::get_trainer_class`：`full_sup → SegTrainer`，`gaze_sup → GazeSupTrainer`。
- `trainers/base.py::BaseTrainer`：通用 iteration loop、日志、test-time validation 调度、checkpoint save/load/resume、wandb。
- `trainers/seg.py::SegTrainer`：全监督 update、IoU/Dice validation、cosine LR hook、prediction 保存。
- `trainers/gaze_sup.py::GazeSupTrainer`：每 level pseudo-mask BCE、cross-level consistency、level/ensemble validation。

### `utils/`

- `utils/__init__.py::get_criterion`：Kvasir/Prostate 均返回 `BCEWithLogitsMaskLoss`。
- `utils/losses.py`
  - `BCEWithLogitsMaskLoss`：逐像素 BCEWithLogits 后做 spatial/batch reduction。
  - `multi_level_consistency_loss`：普通 prediction 间的 stop-gradient agreement。
  - `multi_level_propgation_consistency_loss`：propagated prediction 对 peer non-propagated prediction 的 stop-gradient agreement。
- `utils/metric.py::compute_iou/compute_dice`：sigmoid，可选 0.5 threshold，逐样本计算。
- `utils/util.py::setup_logger/mkdirs/adjust_learning_rate`：日志、目录与 epoch-level cosine schedule。

### `scripts/`

- `scripts/gazesup_kvasir_2_levels.sh`：Kvasir released recipe；SGD、batch 8、224²、lr `1e-2`、cos、15k iterations、2 levels、prop consistency、weight 3、seed 0。
- `scripts/gazesup_prostate_2_levels.sh`：同上，输入通道改为 1。
- 两脚本的 `EXP_PATH`、`ROOT` 都是 `PUT YOUR PATH HERE`，没有 path quoting 或环境检查。

### `GazeMedSeg/` 与 notebooks

- `GazeMedSeg/README.md`：这是 gaze CSV 下载位置/数据字典说明，不是 Python package。`.gitignore` 忽略该目录中除 README 以外的内容，因此原始 fixation CSV 不在仓库中。
- `notebooks/gaze_annotation/generate_gaze_annotation_kvasir.ipynb`：Kvasir fixation → heatmap → DenseCRF map。
- `notebooks/gaze_annotation/generate_gaze_annotation_nci-isbi.ipynb`：NCI fixation → heatmap → DenseCRF map。
- `notebooks/preprocess/process_nci-isbi.ipynb`：从官方 volume/challenge mask 中保留 prostate-positive axial slices，生成 DICOM/PNG 文件和 `train.txt/test.txt`。

---

## B. Training Pipeline

以下以 released 两层 gaze supervision 为主。

### Kvasir 单样本数据流

| 阶段 | 文件与符号 | 输入 | 输出/shape |
|---|---|---|---|
| split/index | `datasets/kvasir_seg.py::KvasirSegDataset.__init__` | `<root>/train.txt` 中 image id | `images/<id>.jpg`、`masks/<id>.jpg` 路径；训练取前 `int(N*data_size_rate)` 个 |
| raw image/GT | `KvasirSegDataset._fetch_data` | RGB JPEG、grayscale mask | image `[H0,W0,3] float32`，label `[H0,W0] int16` |
| gaze pseudo map | `KvasirGazeDataset._fetch_data` | `gaze/crf_compat1/<id>.jpg`，每 level 实际读取同一 CRF 文件 | `pseudo_label_i [H0,W0] float32`，值约 `[0,1]` |
| preprocessing/augmentation | `KvasirGazeDataset.get_transform` | 上述数组 | channel-first；训练 resize 至 `[3,224,224]`、`[1,224,224]`；image/label/pseudo-label 同步随机 flip |
| normalization/binarization | `KvasirGazeDataset._transform_custom` | resized tensors | image 做 ImageNet normalization；GT 变 long binary；pseudo 1 用 `>=0.5`、pseudo 2 用 `>=0.3`，均为 `[1,224,224] int32` |
| dataloader | `datasets/__init__.py::get_dataloader` | dataset | 默认 batch：image `[8,3,224,224]`，label `[8,1,224,224]`，两个 pseudo `[8,1,224,224]` |
| model | `models/multi_level.py::MultiLevelModel.forward` | image `[B,3,H,W]` | 每层 `feature_i [B,128,H,W]`、`logits_i [B,1,H,W]`、`feature_prop_i [B,128,H,W]`、`logits_prop_i [B,1,H,W]`；`logits` 为 raw-logit mean `[B,1,H,W]` |
| supervised loss | `trainers/gaze_sup.py::GazeSupTrainer._update` + `BCEWithLogitsMaskLoss.forward` | `logits_i`、`pseudo_label_i` | 每层 scalar BCE；两层相加 |
| consistency | `GazeSupTrainer._update` + `multi_level_propgation_consistency_loss` | 每层普通/propagated logits | length-`m` loss vector；peer probability detach；训练加 `cons_weight * loss_consistency.sum()` |
| backward/update | `GazeSupTrainer._update` | total scalar loss | `zero_grad → backward → SGD.step`；released SGD momentum 0.99、Nesterov、weight decay 0.0004 |
| evaluation | `GazeSupTrainer.validate` | test image、原始尺寸 GT | test image `[1,3,224,224]`；GT 保持 `[1,1,H0,W0]`；各 level logits bilinear interpolate 至 GT size，mean logits，再算 per-image IoU/Dice |

重要实现事实：gaze-supervised training 仍由 `KvasirSegDataset._fetch_data` 读取完整训练 GT mask，并在 augmentation 中处理它，但 `GazeSupTrainer._update` 完全不读取 `minibatch["label"]`。因此 GT 没进入训练 loss，却是当前 data pipeline 的硬依赖；删除训练 masks 会导致 gaze training 不能启动。

### NCI-ISBI 单样本差异

- `notebooks/preprocess/process_nci-isbi.ipynb` 先在 volume 中仅保留 `mask.max()>0` 的 axial slices。
- `NCIISBIProstateDataset._fetch_data` 产生 DICOM image `[H0,W0] float32`、PNG mask `[H0,W0] int16`。
- `NCIISBIProstateGazeDataset.get_transform` 用 `EnsureChannelFirstd(channel_dim="no_channel")`，所以 train batch image 为 `[8,1,224,224]`。
- intensity 使用 MONAI `NormalizeIntensityd`，不是 Kvasir 的 ImageNet mean/std。
- 两层 thresholds 为 `0.2`、`0.4`；CRF 文件为 PNG。
- test 同样只 resize image，不 resize label；validation 将 logits 插回原 mask spatial size。

### 训练循环

`run.py::main → BaseTrainer.run`。iteration 从 1 到 `max_ite`（含 15000）；iterator 耗尽时递增 epoch。`SegTrainer._epoch_begin_hook` 每个 epoch 更新一次 cosine LR，不是每 iteration。每 100 steps log、每 5000 保存 `model_ite*.pth`、每 1000 在 test 上 evaluation/save latest/select best。

---

## C. Gaze Pipeline

### 1. 原始 gaze 数据

`GazeMedSeg/README.md` 说明 CSV 已先过滤 duration <50 ms 的 fixation 和 image 外 outlier；仓库没有实现这一步，也没有原始未过滤记录。主要字段为 image id、fixation order、normalized X/Y、duration、pupil、timestamp 和原图宽高。

### 2. Heatmap 生成

- Kvasir notebook cell 3 `draw_heatmap`、cell 4 主循环；NCI notebook cell 3/4 同名逻辑。
- 在固定 768×768 canvas 上，把每个 normalized fixation 映射到 pixel，累加 `CURRENT_FIX_DURATION`。
- Kvasir Gaussian sigma/kernel 参数为 70，NCI 为 50；min-max normalize 后以 bicubic resize 回原图大小，保存 8-bit heatmap。
- `gaussian` 函数虽定义，但主路径实际调用 `scipy.ndimage.filters.gaussian_filter`。
- 坐标公式是 `round(norm*size)-1`；如果合法 fixation 恰好为 0，会索引 `-1` 并绕到末行/列，这是潜在边界 bug。

### 3. CRF map

- 两个 gaze notebook cell 7 `crf_inference`：把 continuous heatmap 组成 `[background=1-p, foreground=p]` unary；DenseCRF2D 迭代 10 次。
- 只添加 bilateral pairwise term：`sxy=80`、`srgb=13`、`compat=1`；Gaussian pairwise term 被注释。
- Kvasir用 RGB 原图；NCI 先把 DICOM min-max 映射到 0–255 并复制成 RGB。
- 训练不会读 `gaze/heatmap`；只读 `gaze/crf_compat1`。所以 CRF map 确实是进入训练的 continuous pseudo source。

### 4. Hierarchical threshold 与 pseudo-mask

threshold 不在 notebook 中批量生成多个 mask 文件，而是在 dataset runtime 产生：

1. `configs/static.py` 选择每 level threshold/compat；
2. `KvasirGazeDataset/NCIISBIProstateGazeDataset._fetch_data` 读 continuous CRF map；
3. `get_transform` 用 bilinear resize continuous pseudo map；
4. `_transform_custom` 再执行 `>= threshold` 二值化。

因此“hierarchical thresholds on gaze heatmaps”在发布实现中更精确地说是“hierarchical thresholds on DenseCRF-refined gaze maps，且在 resize 后 threshold”。

### 5. Multi-level framework

- `MultiLevelModel.__init__` 保存 `num_levels` 个完整 `UNet` 于 `nn.ModuleList`。
- 每个 level 看到同一 image，但 `GazeSupTrainer._update` 用对应 `pseudo_label_i` 做独立 BCE。
- inference/validation 对每层 raw logits 求均值；没有 voting 或 weight learning。
- 所有网络参数、optimizer state 同时驻留；所有 level forward graph 在一次 total loss backward 前保留。

### 6. Cross-level consistency

`feature_propogate` 对 final `[B,128,H,W]` feature 做 5×5 dilation-2 neighborhood unfold；用 query-neighbor similarity、ReLU clamp 和 softmax 加权邻域 feature，再复用各 level 的 `classifier` 得到 `logits_prop_i`。

`multi_level_propgation_consistency_loss` 将 binary sigmoid probability 扩成 `[1-p,p]` 两通道。对 level `i`，最大化 propagated probability 与所有 peer `j` 的 non-propagated probability 点积；peer 使用 `.detach()`。当所有 `i` 项求和时，每个网络只从自己的 propagated branch 收到 consistency gradient，符合“peer frozen”的优化意图。

### 7. 论文—代码差异（不在本阶段修改）

- 论文称 level networks independently initialized；代码 `deepcopy(model_list[0])`，实测初始参数逐元素完全相同。它们只因不同 pseudo-label gradient 在训练后分化。
- 论文 Eq. (1) 写 cosine similarity；代码 `torch.einsum` 使用未做 L2 normalization 的 raw dot product。
- 论文举例 3×3 dilation 1；代码固定 5×5 dilation 2。
- 论文正文写 Adam/lr `4e-4`；附录和 released scripts 写 SGD/lr `1e-2`。
- notebook 在 gaze/CRF 生成中读取 GT mask，但主生成路径的 `gt/gt_array` 没用于 heatmap/CRF；GT 只在 optional quality-assessment cell 中参与 Dice。

---

## D. Model Architecture

### 支持的 backbone

只有一个：`parse_args.py` 把 `--model` choices 限制为 `unet`；`models/__init__.py::get_model_opt` 也只有 `unet` 分支。没有 ResNet、Transformer、SAM、Mamba 或 pretrained backbone。

### 默认/released backbone

`models/unet.py::UNet` 内部实例化 MONAI `BasicUNet(spatial_dims=2, features=(64,128,256,512,1024,128), GroupNorm(num_groups=4))`，从 scratch 训练。

命名上 `self.encoder` 实际是完整 BasicUNet encoder-decoder，而不仅是 encoder。其内部 children 是：`conv_0`、`down_1..4`、`upcat_4..1`、`final_conv`。BasicUNet 最终输出 128-channel full-resolution feature，再由 `UNet.classifier` 1×1 conv 输出 binary logit。

### 参数量

| 模型 | 参数数 | FP32 weights |
|---|---:|---:|
| Kvasir single UNet（3-channel） | 31,384,385 | 119.72 MiB |
| Kvasir two-level | 62,768,770 | 239.44 MiB |
| Prostate single UNet（1-channel） | 31,383,233 | 119.72 MiB |
| Prostate two-level | 62,766,466 | 239.43 MiB |

每个 `UNet.head` 含 33,024 参数，但 `UNet.forward` 从不调用它；实测 backward 后 `head` 没有 gradient。这是 dead parameters，不应误认为当前方法的 projection head。CLI `--feat_dim` 也没有传入模型；代码硬编码 128。

### Intermediate visual features

- 容易取得：`UNet.forward` 已返回 final decoder feature `feature [B,128,H,W]`；`MultiLevelModel.forward` 返回 `feature_1...feature_m` 和 propagated feature。
- 不容易取得：BasicUNet 的多尺度 down/up intermediate features 没被 wrapper 返回。未来若需要多尺度视觉特征，应使用显式 adapter/hooks 或最小扩展 BasicUNet forward；当前不应重构整个网络。

---

## E. Experimental Protocol

### 当前 split 与模型选择

- `run.py` 只调用 `get_dataloader(split="train")` 和 `get_dataloader(split="test")`；没有 validation loader、`val.txt` 或 val-only method。
- `BaseTrainer.run` 每 `val_step` 直接调用 `validate(self.test_dataloader)`。
- gaze 方法把 `main_metric` 设为 `mdice`；若当前 test Dice 高于历史最好值就保存 `model_best.pth`。
- full supervision 则按 `miou` 选 best，虽然论文主表报告 Dice；selection criterion 不一致。
- 训练结束又评估 latest 与报告 best，但不重新加载 best 做一次独立 test。
- `--test` 只影响 run id 字符串；不会跳过训练，也不会加载 `ckpt_path`。当前没有独立 inference/test mode。

结论：发布流程使用 test set 做模型选择。公开 test 指标已参与训练过程中的 checkpoint selection，不再是无偏最终泛化估计；也存在依据 test 曲线调整超参数/停止点的风险。

### 最小化加入 validation 的方案（本阶段不修改）

保留官方 reproduction protocol 完全不动，并新增一条明确命名的 clean protocol：

1. 增加固定、版本控制的 `val.txt`，但不要覆盖官方 `train.txt/test.txt`。
2. Kvasir 从 900 training images 中按固定 seed 划出例如 90 val；官方 100 test 保持封存。
3. NCI 必须按 volume/case 划分 60 training volumes，例如固定 54 train/6 val，再展开 slice；禁止随机 slice split，否则同一 volume 的相邻 slice 会泄漏。
4. `run.py` 构造 train/val/test 三个 loader；trainer 用 val `mdice` 保存 best。
5. 训练完加载 best，test 只运行一次。超参数、阈值、iteration、checkpoint selection 都只看 val。
6. 保存 split manifest、case ids、生成 seed 和 SHA-256；启动时 assert train/val/test id disjoint。
7. 报告中分开列出：`Official reproduction (test-selected)` 与 `Clean validation protocol`，不要把二者数值直接混合比较，因为训练样本数和 selection protocol 已不同。

---

## F. Reproducibility Risks

| 风险 | 证据/影响 |
|---|---|
| seed 不完整 | `run.py` 只设 Python/NumPy/Torch/CUDA seeds；未设 cuDNN deterministic/benchmark、`torch.use_deterministic_algorithms`、MONAI `set_determinism`、DataLoader generator/worker init。多 worker random transform 不保证跨版本/平台逐位复现。 |
| split 不完整 | Kvasir split 生成逻辑未提供；NCI train list 在 notebook 中先 `np.random.shuffle` 且未设 seed。`data_size_rate<1` 取 list 前缀，因此排序会影响子集。 |
| paper/released recipe 冲突 | main paper 写 Adam `4e-4`；appendix/scripts 写 SGD `1e-2`。README paper/released Dice 也不同。必须记录运行的是哪套 recipe。 |
| environment 难以直接重建 | Linux-specific Conda lock 含大量 build strings 和混合 CUDA toolkit entries；当前 Windows/Python 3.12 上 MONAI 1.1 还遇到已移除 import API 与新版 NumPy incompatibility。 |
| FP16 路径缺陷 | `GazeSupTrainer._update` 缺 `self.scaler.update()`；与 `BaseTrainer/SegTrainer` 不一致。现状不应声称支持稳定 mixed precision training。 |
| checkpoint/test 语义不完整 | `--test` 未实现；`--ckpt_path` 只在 `--resume` 时使用。`resume_configure` 不恢复 `best_performance_dict`，也不保存/恢复 RNG、scheduler/scaler state。 |
| evaluation std 含义不同 | 代码 `mdice_std` 是 test images 间标准差；论文表中的 ± 是 three seeds/runs 的标准差。不能直接对齐。 |
| save prediction 失败 | `SegTrainer.save_pred_batch` 对 Torch Tensor 调 `.astype(np.float32)`；实测抛出 `AttributeError`。 |
| GT 是弱监督 loader 的隐性依赖 | gaze train loss 不用 GT，但 dataset 必须读取完整 mask；只持有 image+gaze 的真实弱标注场景无法直接运行。 |
| bare exception | `BaseTrainer.run` 用裸 `except` 判断 dataloader epoch end，可能把非 `StopIteration` 数据错误误分类一次，降低诊断性。 |
| CLI 默认/未使用参数 | `--opt` 默认值 `a` 不受 choices 约束，会造成 optimizer 未定义；method/data/root/exp_path 不是 argparse required；`feat_dim/finalize/test` 未进入实际算法。 |
| hard-coded/path assumptions | notebooks 和 scripts 有绝对 `/home/...` 示例/占位符；notebook gaze CSV 用相对 `../../GazeMedSeg/...`；代码全程 `.cuda()`，没有 CPU/device abstraction。 |
| dependency surface 偏大 | baseline 未用的 sklearn clustering/logistic regression 与 wandb 仍被顶层 import；即使关闭 wandb 也必须安装它，增加环境失败点。 |
| metric corner case | empty GT + empty prediction 的 Dice/IoU 返回 0 而不是常见定义的 1；当前 datasets 主要保留有目标样本，但应明确。 |
| source/论文实现差异 | identical deepcopy initialization、dot product vs cosine、5×5 dilation-2 vs 论文示例；复现时必须以 code commit 为准并单独披露。 |

固定 released 主要参数：batch 8、224²、15k iterations、SGD momentum 0.99/Nesterov、lr `1e-2`→`1e-4` epoch-level cosine、weight decay `4e-4`、random horizontal/vertical flip、2 levels、prop consistency weight 3、seed 0、FP16 off。论文结果声称三 seeds，仓库只提供 seed-0 scripts；其余 seeds 需显式运行并由外部聚合。

---

## G. Data Leakage Risks

### 确定存在

1. **Test-driven checkpoint selection**：test 每 1000 iterations 被读取，best 直接由 test metric 决定。
2. **重复观察 test 曲线**：日志/W&B 可持续显示 test metrics，研究人员能够据此调整 training decisions；代码没有防护。

### 可能存在或需要约束

1. `GazeMedSeg/README.md` 称 Kvasir fixation 覆盖全部 1000 images，而训练集只有 900。gaze notebook 会为 CSV 中所有 images 生成 heatmap/CRF，且 optional quality cell 可在全部 masks 上算 Dice。训练 dataset 只按 `train.txt` 读取 pseudo maps，所以没有直接把 test pseudo-label 喂入训练；但 threshold 的来源/选择过程未记录，无法排除观察 test annotation quality 后选全局阈值。应视为“无法证伪的 protocol risk”，不是已证实的 tensor-level leakage。
2. NCI preprocessing 使用 test ground-truth mask 来筛出 prostate-positive slices。这定义了一个依赖 test label 的条件评估 cohort；模型没有用 mask 训练，但结果不代表全 volume/all-slice performance。
3. gaze training loader 读取 GT mask 但 loss 不使用。当前代码没有 tensor-level GT loss leakage；然而未来修改 trainer 时很容易误用，建议 baseline tests 显式断言 gaze update 不访问 `label`。
4. 没有 train/test filename 或 patient-id overlap assertion。NCI 使用官方 volume train/test 原则上分离，但代码本身不验证；未来 validation 必须 case-level split。

---

## H. RTX3090 Feasibility

### 计算/显存结构

- 两个完整 UNet 的 weights 约 239.44 MiB FP32；再加 gradients 与 SGD momentum，纯模型状态约 0.7 GiB 量级，参数本身不是主瓶颈。
- 主瓶颈是 `MultiLevelModel.feature_propogate`。对单层 feature `[B,128,224,224]`，一次 5×5 `unfold` 为 `[B,128,25,224,224]`。
- 官方 `B=8` 时一次 unfold 为 1,284,505,600 elements，即 4.785 GiB FP32。代码对相同 feature 做两次 unfold，因此每 level 约 9.57 GiB；两 level 的 training graphs 在统一 backward 前保留，两个 level 合计约 19.14 GiB FP32，仅为核心邻域 tensor 下界。
- UNet multiscale activations、softmax/correlation、loss graph、CUDA allocator fragmentation 与 workspace 会继续增加峰值。论文使用 NVIDIA A40；其显存条件不能直接外推到 24GB 3090。
- model weights 同时驻留；level forward 虽顺序执行，但反向图必须同时保留。validation 也无条件计算 `logits_prop_i`，即使最终 metric 只用普通 logits，因此 inference 的 LPP 峰值仍显著。

### 3090 判断

- **batch 8 + 224² + FP32：高度可能 OOM。** 建议先做单 iteration memory probe，而不要开始 15k full run。
- **FP16 数值上很有价值**：仅上述两层 unfold 下界可降到约 9.57 GiB；但当前 gaze FP16 training 路径有缺失 `scaler.update()` 的实现缺陷，不能未经修复就依赖。
- **建议优先级**：
  1. 保持两 levels、UNet、224²、loss 和 released hyperparameters，先测 batch 8 FP32 峰值并保留 OOM 证据。
  2. 经明确批准后，只修复 `GradScaler.update()` 这一基础设施 bug，使用已有 `--fp16` 选项；这不改变方法结构，但属于原代码修改，必须单独 commit/记录。
  3. 若仍 OOM，再把 physical batch 降至 4/2/1，并明确这是 protocol deviation。若需要保持 effective batch 8，后续可另行批准 gradient accumulation；当前不实现。
  4. spatial size 最后调整，因为 224² 是附录明确实验分辨率，改变它比降 physical batch 更直接改变任务信息。
- 不建议为省显存减少 `num_levels`、删除 LPP、换 backbone 或改 loss；这些会改变原方法。

一个代数上明显的工程观察是 `feat_value = feat` 后对二者重复 `unfold`；复用一次结果可能节省大量显存，但这是源代码优化，必须在 baseline reproduction 后单独验证数值等价，本阶段未实施。

---

## I. Reproduction Status

### 已完成

- 下载并锁定官方 commit；archive SHA-256：`7DE477072B585F94FEC57140DCC4DBCA071C00C4377C5FD9ED0BD8E6B6F3CC0F`。
- 官方 29 个文件逐文件 hash 与 archive 比对：29/29 PASS；未修改官方源文件/算法。
- Python syntax `compileall`：PASS。
- 在隔离审计环境执行结构测试：PyTorch 2.13 CPU、MONAI 1.1.0、NumPy 1.26.4。由于主机只有 Python 3.12，对 MONAI 1.1 import 使用了仅限进程内的 importlib compatibility shim；没有写回仓库。
- 参数量、模块 child structure、level 初始权重相等、dead `head` 无梯度：已实测。
- official spatial size 的 two-level Kvasir `batch=1, 224×224` CPU forward：PASS，ensemble logits `[1,1,224,224]`。
- synthetic Kvasir：image/mask/CRF-map 读取、两 threshold pseudo-mask、同步 transform、DataLoader shapes：PASS。
- synthetic end-to-end：multi-level forward、每层 BCE、propagation consistency、finite loss、backward、SGD step：PASS。
- evaluation：test image resize、原尺寸 label 保留、logits interpolate、IoU/Dice 函数：代码路径 PASS。synthetic/untrained 数字不作为实验结果报告。
- checkpoint `BaseTrainer.save` payload、optimizer/model state serialization 与 state reload：PASS。
- `save_pred_batch`：FAIL，已稳定复现 `Tensor` 无 `.astype`。

### 尚未完成及原因

- 真实 Kvasir-SEG/GazeMedSeg 数据读取：未运行；工作区没有 Kvasir images/masks、`train.txt/test.txt`、fixation CSV 或预生成 CRF maps。
- gaze heatmap/DenseCRF notebook 全流程：未运行；缺原始 fixation/data，且未安装/验证 official `pydensecrf` build。
- 原版 CUDA trainer/run.py：未运行；官方 Conda 环境未提供，本机是 Windows 且现有隔离 PyTorch 为 CPU build；代码又硬编码 `.cuda()`。
- RTX 3090 实测：未运行；当前主机实际 GPU 为 16GB RTX 4070 系列，不是 3090 24GB。
- 15k training、best/latest checkpoint selection、released Dice：未运行；缺真实数据和匹配官方 CUDA 环境。
- resume 全路径：仅静态审计与 checkpoint serialization；未在 CUDA trainer 中恢复并续训。

因此当前状态是“官方代码完整静态审计 + 核心可执行性烟雾测试通过”，不是“论文数值复现完成”。

---

## J. Future Extension Points（仅定位，不实现）

为避免污染 baseline，未来组件应作为新 method/model/trainer 分支存在，保留 `gaze_sup` 原路径和 checkpoint compatibility。

### 1. Structured Text

- 数据入口：在 `BaseImageDataset.__getitem__` 返回字典中加入与 `subject_id` 对齐的结构化字段，或在 `Kvasir*/NCI*Dataset._fetch_data` 旁增加独立 metadata loader。
- 推荐保持 image/gaze/split manifest 的一一映射，文本缺失策略在 dataset 层处理。
- 当前代码没有任何文本字段、tokenizer 或 text encoder。

### 2. Vision-Language Interaction

- 最小视觉接口：`models/unet.py::UNet.forward` 中 `feat = self.encoder(x)` 与 `logits = self.classifier(feat)` 之间；现有 `feat [B,128,H,W]` 已直接暴露。
- multi-level 接口：`MultiLevelModel.forward` 取得每层 `feature_i` 后、调用 classifier/LPP 前。
- 如需多尺度 encoder features，应用 wrapper/hooks 显式暴露 BasicUNet `down_* / upcat_*` 输出，不应重构整个 baseline。

### 3. Human Attention Calibration

- 连续 attention 数据入口：`KvasirGazeDataset._fetch_data` / `NCIISBIProstateGazeDataset._fetch_data` 当前只把 CRF map 作为待阈值 pseudo-label；未来新 method 可额外读取 `gaze/heatmap` 或 continuous CRF map 为独立 tensor key，但不改 baseline key 语义。
- calibration 模块位置：在新模型分支中，接收 continuous attention、visual feature 和 language representation，在 `UNet.classifier` 前产生 calibrated visual representation；或在新 trainer 中产生明确的 calibrated supervision target。
- 不应把 calibration 逻辑塞入 `BCEWithLogitsMaskLoss` 或覆盖 `GazeSupTrainer._update`，否则无法保持原方法可复现对照。

---

## 建议的下一阶段进入条件

1. 固定并归档 official preprocessed Kvasir root，记录 `train.txt/test.txt` 与所有文件 checksum。
2. 在 Linux/Python 3.9 上重建 official environment，先运行 `batch=1, max_ite=1` 的真实数据 CUDA smoke。
3. 在目标 RTX 3090 测 `batch=8, 224², FP32` peak memory；若 OOM，先提交独立的 FP16 scaler bugfix 说明与审批，不触碰算法。
4. 完成 released seed-0 reproduction 后，再运行多个 seeds；将 image-wise std 与 run-wise std 分开。
5. baseline 锁定后再引入独立 validation protocol；在此之前不实现 Structured Text、Vision-Language Interaction 或 Human Attention Calibration。
