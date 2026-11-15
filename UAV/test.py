import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的模型权重（替换为实际训练生成的best.pt路径）
    # 训练权重默认保存路径：project/name/weights/best.pt（对应训练脚本中的project和name）
    model = YOLO(ROOT / 'UAV/UAV2_3/runs/Yolov11n/Yolov11n-FPN-DEConv-FocalEIoULoss/train-test3/weights/best.pt')
    
    # 模型验证
    results = model.val(
        data=ROOT / 'UAV/UAV2_3/UAV2_3.yaml',  # 与训练一致的数据集配置文件
        imgsz=1024,  # 与训练一致的图像尺寸
        batch=4,  # 验证时的批量大小（可根据显存调整）
        device=0,  # 计算设备（与训练一致，cpu或gpu编号）
        split='val',  # 验证集划分（默认是val，无需修改）
        save_json=False,  # 是否保存COCO格式的json结果（如需进一步分析可设为True）
        save=True,  # 是否保存验证时的预测可视化结果
        project=ROOT / 'UAV/UAV2_3/runs/Yolov11n/val-and-test/FPN-DEConv-FocalEIoU',  # 验证结果保存的项目目录
        name='val'  # 本次验证的结果文件夹名称
    )
    
    # 打印关键指标（mAP@0.5、mAP@0.5:0.95等）
    print(f"mAP@0.5: {results.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.box.map:.3f}")
