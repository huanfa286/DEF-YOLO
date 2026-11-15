# -----------YOLOv8train-----------------

# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
# if __name__ == '__main__':
#   model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
#   model.load('yolo-parameter/v8/yolov8x.pt')  #注释则不加载
#   results = model.train(
#     data='UAV/UAV.yaml',  #数据集配置文件的路径
#     epochs=200,  #训练轮次总数
#     batch=8,  #批量大小，即单次输入多少图片训练
#     imgsz=1024,  #训练图像尺寸
#     workers=8,  #加载数据的工作线程数
#     device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
#     optimizer='SGD',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
#     amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
#     cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
#     resume = True,  # 关键参数：启用恢复训练模式
#     project='UAV/runs',
#     name='1-1024sunny'
# )

# -----------YOLOv11train-----------------
# from ultralytics.nn.modules import register_module
# # 导入你封装后的LAMA模块
# from ultralytics.nn.modules.LAMA import LAMA

# # 注册自定义模块
# register_module(LAMA)

import csv
import os
import sys
from pathlib import Path

# Some Windows Python environments load two OpenMP runtimes through different
# numerical packages. Allow the process to continue with the already selected
# runtime so the training entry point can start.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ultralytics.utils.files import increment_path

# Keep the original GPU settings when CUDA is available, but make the entry point
# runnable on CPU-only machines as well.
HAS_CUDA = torch.cuda.is_available()
DEVICE = 0 if HAS_CUDA else "cpu"
BATCH_SIZE = 4 if HAS_CUDA else 1
# Windows multiprocessing workers can exit during long validation runs.
WORKERS = 0 if sys.platform == "win32" else (8 if HAS_CUDA else 0)
USE_AMP = HAS_CUDA
RUN_PROJECT = ROOT / "UAV/UAV2_3/runs/Yolov11n/Yolov11n-FPN-DEConv-FocalEIoULoss"
RUN_NAME = "train-gamma15-map50"
TARGET_EPOCHS = 200
# Always allocate a new run directory. Existing results are never treated as a
# completed run and are never mixed into the next training invocation.
RUN_DIR = increment_path(RUN_PROJECT / RUN_NAME, exist_ok=False)
RUN_NAME = RUN_DIR.name
RUN_WEIGHTS = RUN_DIR / "weights" / "best.pt"
BEST_MAP50_WEIGHTS = RUN_DIR / "weights" / "best_mAP50.pt"
RESULTS_CSV = RUN_DIR / "results.csv"


def report_best_map50(results_path):
    """Write and print the row with the highest validation mAP50."""
    if not results_path.exists():
        return None
    with results_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("metrics/mAP50(B)")]
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    checkpoint_path = (
        str(BEST_MAP50_WEIGHTS)
        if BEST_MAP50_WEIGHTS.exists()
        else "not created yet; rerun training with the updated script"
    )
    curves_path = (
        str(RUN_DIR / "best_mAP50_eval")
        if (RUN_DIR / "best_mAP50_eval").exists()
        else "created after the updated training finishes"
    )
    summary = (
        f"Best mAP50 epoch: {int(float(best['epoch']))}\n"
        f"mAP50: {float(best['metrics/mAP50(B)']):.5f}\n"
        f"mAP50-95: {float(best['metrics/mAP50-95(B)']):.5f}\n"
        f"Precision: {float(best['metrics/precision(B)']):.5f}\n"
        f"Recall: {float(best['metrics/recall(B)']):.5f}\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Curves: {curves_path}\n"
    )
    summary_path = RUN_DIR / "best_mAP50.log"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Best mAP50 log: {summary_path}")
    return best


import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO('ultralytics/cfg/models/11/improve1/yolo11-DEConv.yaml')
    model = YOLO(ROOT / "ultralytics/cfg/models/11/improve1/FPN-DEConv/yolov11-FPN-DEConv.yaml")
    # model = RTDETR('rtdetr-l.pt')

    model.load(ROOT / "yolo11n.pt")  # 注释则不加载
    results = model.train(
        data=ROOT / "UAV/UAV2_3/UAV2_3.yaml",  # 数据集配置文件的路径
        epochs=TARGET_EPOCHS,  # 训练轮次总数
        batch=BATCH_SIZE,  # CUDA使用4，CPU使用1
        imgsz=1024,  # 训练图像尺寸
        workers=WORKERS,  # Windows使用0，避免长时间验证时子进程异常退出
        device=DEVICE,  # CUDA优先，无CUDA时自动使用CPU
        optimizer="AdamW",  # 训练使用优化器，可选 auto,SGD,Adam,AdamW 等
        amp=USE_AMP,  # 仅CUDA环境启用自动混合精度
        cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
        # resume = True,  # 关键参数：启用恢复训练模式
        project=RUN_PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        lr0=0.001,  # 初始学习率，AdamW搭配0.001最优，SGD是0.01，适配改进模型
        lrf=0.01,  # 学习率衰减因子，余弦退火到初始的1%，防止后期震荡
        warmup_epochs=3,  # 热身训练3轮，防止初始学习率太大导致梯度爆炸
        weight_decay=0.0005,  # L2正则化，轻微权重衰减，防止过拟合（无人机数据集易过拟合）
        patience=100,  # 早停机制：100轮精度不涨则自动停止训练，避免无效训练+过拟合
        val=True,  # 训练中验证，实时监控精度，必开
        save=True,  # 保存最优模型，必开
        # save_period=10,  # 每10轮保存一次权重，防止意外中断
        cos_lr=True,  # ✅ 开启余弦退火学习率：核心！学习率平滑下降，收敛更稳定，精度更高
        label_smoothing=0.05,  # ✅ 标签平滑：轻微平滑标签，防止过拟合，对小目标检测精度提升明显
        mosaic=0.8,  # ✅ 马赛克增强：从1.0→0.8，降低增强强度，无人机小目标避免被过度切割导致特征丢失
        mixup=0.0,  # 关闭mixup，避免小目标特征被混合增强削弱
        copy_paste=0.1,  # ✅ 开启轻微复制粘贴增强：针对无人机小目标密集场景，提升小目标召回率
        single_cls=False,  # 数据集是多类则False，单类则True，根据你的UAV数据集修改
        rect=False,  # 1024 尺寸关闭矩形训练，避免小目标特征变形
    )
    report_best_map50(RESULTS_CSV)
