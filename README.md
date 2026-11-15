# DEF-YOLO

DEF-YOLO 是一个面向无人机航拍交通场景的小目标检测模型，基于 Ultralytics YOLO11n 实现。本仓库在 YOLO11 的特征提取与检测框架上构建细节增强特征融合网络，并使用 Focal-EIoU 边界框回归损失，以提升航拍图像中小型、密集交通目标的特征表达和定位能力。

本 README 介绍 DEF-YOLO 主模型、训练和推理流程。模型配置为 `ultralytics/cfg/models/11/improve1/FPN-DEConv/yolov11-FPN-DEConv.yaml`。

## 项目特点

- 基于 YOLO11n，保留骨干网络、特征融合颈部和检测头的整体框架。
- 在 FPN/PAN 双向融合路径中加入浅层辅助特征，补充高分辨率细节。
- 在五个特征融合节点后加入 DEConv，增强目标边缘、纹理和局部差异特征。
- 使用 Focal-EIoU 改进边界框回归，对定位误差较大的样本赋予更大的优化权重。
- 检测头输出 P2/4、P3/8、P4/16 三个尺度，重点保留 P2 高分辨率分支以适应航拍小目标。
- 支持自建 UAV 三类交通目标数据集和 VisDrone2019 十类目标数据集的训练与评估。

## 模型结构

DEF-YOLO 的主干网络沿用 YOLO11n 的 Conv、C3k2、SPPF 和 C2PSA 结构。除主干特征外，模型从浅层网络引出三个辅助特征分支，并将其接入颈部进行多尺度融合。颈部采用自顶向下和自底向上的特征传递路径，在每次关键融合后通过 DEConv 增强融合特征，再送入 C3k2 进行特征整合。

三个检测输出分别对应 P2/4、P3/8 和 P4/16。与仅使用较低分辨率检测层相比，P2/4 分支保留了更细的空间信息，有助于发现占图像像素较少的车辆和非机动车目标。

DEConv 由普通卷积以及中心差分、水平差分、垂直差分和角度差分卷积分支组成。各分支的等效卷积核在前向过程中合并，提供强度与方向细节信息。实现位于 `ultralytics/nn/modules/DEConv.py`，模型解析器在 `ultralytics/nn/tasks.py` 中注册该模块。

边界框损失位于 `ultralytics/utils/loss.py` 和 `ultralytics/utils/metrics.py`。实现先计算 EIoU 相似度，再使用 `(1 - EIoU)^gamma` 调节回归项，当前 `gamma` 默认值为 1.5；分类损失和分布焦点损失仍按训练器的其余实现计算。

## 目录结构

```text
DEF-YOLO/
├── ultralytics/
│   ├── cfg/models/11/improve1/FPN-DEConv/
│   │   └── yolov11-FPN-DEConv.yaml  # DEF-YOLO 模型结构
│   ├── nn/modules/DEConv.py         # DEConv 实现
│   ├── nn/tasks.py                  # 自定义模块的模型解析支持
│   └── utils/loss.py                # Focal-EIoU 损失实现
├── UAV/
│   ├── UAV2_3/                      # 自建数据集目录（需自行准备）
│   ├── UAV2_3.yaml                  # 自建数据集配置
│   ├── train.py                     # UAV 训练入口
│   ├── test.py                      # UAV 验证入口
│   └── predict.py                   # UAV 推理入口
├── VisDrone2019/
│   ├── visdrone2019.yaml            # VisDrone 数据集配置
│   ├── train.py                     # VisDrone 训练入口
│   ├── test.py                      # VisDrone 验证入口
│   └── predict.py                   # VisDrone 推理入口
├── requirements.txt
└── README.md
```

仓库同时包含 Ultralytics 其他模型配置、文档、示例和测试；上面列出的是 DEF-YOLO 相关的主要文件。

## 环境安装

建议使用 Python 3.10 或 3.11，并根据目标电脑的操作系统、显卡驱动和 CUDA 版本安装匹配的 PyTorch 与 TorchVision。然后在仓库根目录运行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目依赖和当前仓库中的 Ultralytics 包：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 指定了 PyTorch 2.2.2 和 TorchVision 0.17.2。若需要 GPU 训练，请先确认这组版本适用于本机 CUDA 环境；如需安装特定 CUDA 版本的 PyTorch，可先安装对应官方 wheel，再根据本机环境调整依赖安装命令。

确认 Python 导入的是仓库内的源码：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

## 数据集准备

数据集采用 YOLO 检测格式。图像和标签按相同文件名一一对应，标签为每行一个目标：

```text
class_id x_center y_center width height
```

坐标为相对于图像宽高归一化到 `[0, 1]` 的数值。推荐目录结构如下：

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

自建 UAV 数据集配置文件 `UAV/UAV2_3.yaml` 定义三类目标：

| ID | 类别 |
|---:|---|
| 0 | small-vehicle（小型车辆） |
| 1 | large-vehicle（大型车辆） |
| 2 | non-motor-vehicle（非机动车） |

VisDrone 配置定义十类目标，类别列表见 `VisDrone2019/visdrone2019.yaml`。两个数据集的类别 ID 和语义不同，训练、验证时必须使用与标签相匹配的数据 YAML。

编辑相应 YAML 中的 `path` 字段，使其指向本机数据集根目录。例如：

```yaml
path: D:/datasets/UAV2_3
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: small-vehicle
  1: large-vehicle
  2: non-motor-vehicle
```

当前仓库中的数据配置和部分辅助脚本记录了其他电脑上的数据绝对路径；在新环境中使用前应按上例调整。数据集本身不包含在代码安装依赖中。

## 训练

可以使用 Python API 直接训练，避免依赖训练脚本中已有的机器专属路径。以下示例假设已修改 `UAV/UAV2_3.yaml`：

```python
from pathlib import Path
from ultralytics import YOLO

model_yaml = Path("ultralytics/cfg/models/11/improve1/FPN-DEConv/yolov11-FPN-DEConv.yaml")
model = YOLO(model_yaml)
model.load("UAV/yolo11n.pt")  # 使用预训练权重初始化；没有该文件时可换成本机 yolo11n.pt 路径

model.train(
    data="UAV/UAV2_3.yaml",
    epochs=200,
    batch=4,
    imgsz=1024,
    device=0,                 # 无可用 CUDA GPU 时可设为 "cpu"
    workers=0,                # Windows 环境可设为 0
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    amp=True,
    project="runs/DEF-YOLO/UAV",
    name="train",
)
```

训练 VisDrone2019 时，将数据配置改为 `VisDrone2019/visdrone2019.yaml`，并按数据规模和显存调整输入尺寸、batch 和 epoch。仓库中的 `UAV/train.py` 与 `VisDrone2019/train.py` 是相应入口脚本；在其他电脑运行前，应检查其中的数据路径、预训练权重路径和输出目录。

训练结果默认包含 `weights/best.pt`、`weights/last.pt`、`results.csv`、训练参数和指标曲线。Ultralytics 训练目录由 `project` 与 `name` 参数决定。

## 验证

使用训练得到的权重在验证集或测试集上评估：

```python
from ultralytics import YOLO

model = YOLO("runs/DEF-YOLO/UAV/train/weights/best.pt")
metrics = model.val(
    data="UAV/UAV2_3.yaml",
    split="test",             # 可改成 "val"
    imgsz=1024,
    batch=4,
    device=0,
    project="runs/DEF-YOLO/UAV",
    name="test",
)

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

常用检测指标包括 Precision、Recall、mAP@0.5 和 mAP@0.5:0.95。比较不同实验时，应保持数据划分、类别定义、输入尺寸和置信度/NMS 评估设置一致。

## 推理

单张图片、图片目录或视频均可作为预测源：

```python
from ultralytics import YOLO

model = YOLO("runs/DEF-YOLO/UAV/train/weights/best.pt")
results = model.predict(
    source="path/to/image-or-video",
    imgsz=1024,
    conf=0.25,
    iou=0.5,
    device=0,
    save=True,
    project="runs/DEF-YOLO/predict",
    name="exp",
)
```

也可在仓库根目录使用 CLI：

```bash
yolo predict model=runs/DEF-YOLO/UAV/train/weights/best.pt source=path/to/image.jpg imgsz=1024 conf=0.25 save=True
```

仓库中的 `UAV/predict.py` 和 `VisDrone2019/predict.py` 是示例脚本，使用前请更新其中的权重、数据源和结果保存路径。

## 其他工具

- `UAV/heatmap.py`：基于预测框置信度批量生成可视化热力图。需修改脚本末尾的输入图片目录、权重文件和输出目录。
- `UAV/export_deconv_fp16.py`：将已训练 DEConv 分支融合为等效卷积，并导出 FP16 检查点。它适用于部署转换，不会改变训练时的 DEF-YOLO 架构定义；请先查看脚本中的默认权重目录是否与当前仓库位置相符。
- `VisDrone2019/VisDrone_sample_2400_split_721.py`：从有对应标签的图像中按配置抽样，并划分训练、验证和测试集。运行前需修改脚本顶部的源图像、标签和输出目录。
- `VisDrone2019/every_label_count.py`：统计数据集标签类别数量。

## 复现实验说明

仓库保存了已有训练运行目录和日志，可用于查看历史实验参数与结果。自建 UAV 运行记录中一次训练日志的最佳验证指标为 mAP@0.5 **0.88156**、mAP@0.5:0.95 **0.63118**、Precision **0.87998**、Recall **0.83196**。这是对应历史运行的验证结果，不代表所有数据划分或硬件环境下都能得到相同表现。

可通过每个运行目录中的 `args.yaml` 和 `results.csv` 查看训练设置与逐轮指标。用于正式报告时，请重新确认数据划分，并在独立测试集上评估。

## 常见问题

**找不到数据集或图像。** 检查 `UAV/UAV2_3.yaml` 或 `VisDrone2019/visdrone2019.yaml` 中的 `path` 是否存在，且 `train`、`val`、`test` 相对路径下有 `images` 和对应 `labels`。

**模型 YAML 中提示找不到 DEConv。** 确认当前工作目录是仓库根目录，并且通过 `python -m pip install -e .` 安装/加载本仓库源码；不要让环境导入其他位置安装的 Ultralytics 包。

**预训练权重未加载或路径不存在。** `yolo11n.pt` 需要单独准备。传入模型结构 YAML 后，可使用 `model.load(...)` 迁移兼容的 YOLO11n 预训练权重；新增加的结构层会由当前模型初始化。

**CUDA 不可用。** 安装与 GPU 驱动匹配的 PyTorch/CUDA 版本，并用 `python -c "import torch; print(torch.cuda.is_available())"` 检查。CPU 可用于调试，但训练速度通常较慢。

## 许可证

本项目基于 Ultralytics 代码库。仓库根目录的 `LICENSE` 文件包含适用的 AGPL-3.0 许可证文本；使用和分发前请阅读该文件，并确认第三方依赖、模型权重及数据集各自的许可要求。
