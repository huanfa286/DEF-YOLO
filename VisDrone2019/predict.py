import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载训练好的模型权重（同上，使用best.pt）
    model = YOLO("VisDrone2019/runs/our/train1/weights/best.pt")

    # 模型预测
    results = model.predict(
        source="E:/Dingzhenhao/python_workplace/Datasets/VisDrone2019/val/images",  # 输入源：可以是单张图片路径（如'test.jpg'）、图片目录或视频路径
        imgsz=800,  # 与训练一致的图像尺寸
        visualize=False,  # 激活推理期间模型特征的可视化，从而深入了解模型正在“看到”的内容。这对于调试和模型解释非常有用。
        conf=0.5,  # 置信度阈值（过滤低置信度预测，0-1之间，可根据需求调整）
        iou=0.5,  # IOU阈值（NMS非极大值抑制阈值，0-1之间）
        device=0,  # 计算设备（与训练一致）
        save=True,  # 是否保存预测结果图像/视频
        save_txt=False,  # 是否保存预测结果的txt文件（格式：class_id x1 y1 x2 y2）
        show=False,  # 是否实时显示预测结果（服务器环境建议设为False）
        show_labels=True,  # 在可视化输出中显示每个检测的标签。 能够立即理解检测到的对象。
        show_conf=True,  # 在标签旁边显示每个检测的置信度分数。 可以深入了解模型对每次检测的确定性。
        show_boxes=True,  # 在检测到的对象周围绘制边界框。 这对于在图像或视频帧中以可视方式识别和定位对象至关重要。
        line_width=1,  # 指定边界框的线条宽度。 如果 None，则线条宽度会根据图像大小自动调整。 提供视觉自定义以提高清晰度。
        project="E:/Dingzhenhao/python_workplace/UAV/UAV2_3/use/VisDrone2019predict/our",  # 预测结果保存的项目目录
        name="pred",  # 本次预测的结果文件夹名称
    )
