import os  # 用于文件夹遍历、创建

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


def batch_draw_yolo_detection_heatmap(input_img_dir, weight_path, output_heatmap_dir="output_heatmaps"):
    """
    批量绘制YOLO目标检测热力图（原始图像+热力图叠加），保存到指定输出文件夹
    :param input_img_dir: 输入图片文件夹路径（存放待处理的所有图片）
    :param weight_path: YOLO权重文件路径（.pt）
    :param output_heatmap_dir: 输出热力图文件夹路径（自动创建，不存在则新建）.
    """
    # 1. 加载YOLO模型
    model = YOLO(weight_path)

    # 2. 创建输出文件夹（若不存在则自动创建，exist_ok=True避免文件夹已存在时报错）
    os.makedirs(output_heatmap_dir, exist_ok=True)

    # 3. 定义支持的图片格式（避免处理非图片文件）
    supported_img_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    # 4. 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_img_dir):
        # 过滤出仅支持的图片格式文件（忽略大小写）
        if not file_name.lower().endswith(supported_img_formats):
            continue

        # 拼接单张图片的完整输入路径
        img_full_path = os.path.join(input_img_dir, file_name)
        try:
            # 5. 推理单张图片，获取预测结果
            results = model(img_full_path)
            result = results[0]  # 取单张图像的推理结果
            img_ori = cv2.imread(img_full_path)
            if img_ori is None:  # 跳过无法读取的损坏图片
                print(f"警告：无法读取图片 {file_name}，已跳过")
                continue
            # 转换为matplotlib支持的RGB格式（cv2默认读取为BGR）
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            h, w, _ = img_ori.shape

            # 6. 初始化空白热力图画布（与原始图像同尺寸，全0初始化）
            heatmap = np.zeros((h, w), dtype=np.float32)

            # 7. 遍历所有预测框，填充热力图
            if result.boxes is not None:  # 避免无检测框时报错
                for box in result.boxes:
                    # 提取预测框坐标（xyxy格式：左上x、左上y、右下x、右下y）
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # 提取该框的置信度
                    conf = box.conf[0].cpu().numpy()

                    # 给预测框覆盖的区域赋值（置信度，重叠区域取最大值）
                    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], conf)

            # 8. 热力图平滑处理（让效果更美观，减少框的锯齿感）
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), sigmaX=10, sigmaY=10)

            # 9. 绘制【原始图像+热力图叠加】（满足观察需求）
            fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)  # 适配图片原始尺寸
            # 先显示原始图像（底层）
            ax.imshow(img_ori)
            # 再叠加热力图（上层，alpha=0.5调节透明度，方便同时观察原图和热力区域）
            ax.imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            # 隐藏所有坐标轴、边框（仅保留叠加后的图像，无多余元素）
            ax.axis("off")
            plt.axis("off")
            fig.patch.set_visible(False)  # 隐藏画布背景

            # 10. 拼接单张热力图的完整输出路径（保持原文件名，方便对应）
            heatmap_full_path = os.path.join(output_heatmap_dir, file_name)
            # 保存叠加后的热力图（去除白边，高清保存）
            plt.savefig(heatmap_full_path, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig)  # 关闭画布，释放内存（批量处理关键，避免内存溢出）

            print(f"已处理并保存：{heatmap_full_path}")

        except Exception as e:
            print(f"错误：处理图片 {file_name} 时失败，错误信息：{e!s}")
            continue

    print(f"\n批量处理完成！所有叠加热力图已保存至：{os.path.abspath(output_heatmap_dir)}")


# 调用函数（修改为你的输入文件夹、权重文件路径）
if __name__ == "__main__":
    batch_draw_yolo_detection_heatmap(
        input_img_dir="E:/Dingzhenhao/python_workplace/UAV/UAV2_3/use/UAV2-3images",  # 你的输入图片文件夹（存放所有待处理图片）
        weight_path="E:/Dingzhenhao/python_workplace/ultralytics-8.3.135(2)-FocalEIoULoss/UAV/UAV2_3/runs/Yolov11n/Yolov11-FocalEIoULoss/train/weights/best.pt",  # 你的YOLO权重文件路径
        output_heatmap_dir="E:/Dingzhenhao/python_workplace/UAV/UAV2_3/use/heatmap/loss/Focal-EIOU",  # 输出热力图文件夹（自动创建）
    )
