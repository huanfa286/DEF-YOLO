import os
import random
import shutil

# ===================== 【核心可配置参数区】 直接修改这里即可 =====================
# 原始数据集路径
IMG_ROOT = "E:/Dingzhenhao/python_workplace/Datasets/VisDrone2019/train/images"
LABEL_ROOT = "E:/Dingzhenhao/python_workplace/Datasets/VisDrone2019/train/labels"

# 抽取的总样本数量、划分比例
SAMPLE_NUM = 2400  # 抽取2400张样本
TRAIN_RATIO = 0.7  # 训练集70%
VAL_RATIO = 0.2  # 验证集20%
TEST_RATIO = 0.1  # 测试集10%

# 划分后的子数据集保存根路径（自动创建，无需手动建文件夹）
SAVE_ROOT = "E:/Dingzhenhao/python_workplace/Datasets/VisDrone2019_subset"

# 随机种子（固定种子保证每次划分结果一致，如需随机结果可修改数字或注释此行）
RANDOM_SEED = 42
# ==============================================================================

# 初始化随机种子
random.seed(RANDOM_SEED)


def main():
    # 1. 获取所有图片文件名，并过滤出【有对应标签】的有效样本
    img_files = [f for f in os.listdir(IMG_ROOT) if f.endswith((".jpg", ".png", ".jpeg"))]
    valid_samples = []
    for img_name in img_files:
        # 核心逻辑：图片和标签的【文件名前缀一致】，后缀分别为图片格式和txt
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(LABEL_ROOT, label_name)
        if os.path.exists(label_path):
            valid_samples.append(img_name)

    # 校验有效样本数量是否足够
    if len(valid_samples) < SAMPLE_NUM:
        print(f"❌ 错误：有效样本数量({len(valid_samples)}) 小于 要抽取的数量({SAMPLE_NUM})")
        return

    # 2. 随机抽取指定数量的样本
    sample_samples = random.sample(valid_samples, SAMPLE_NUM)
    print(f"✅ 成功随机抽取 {len(sample_samples)} 个有效样本")

    # 3. 打乱样本顺序（二次随机，保证划分更均匀）
    random.shuffle(sample_samples)

    # 4. 按比例计算各数据集的样本数量
    train_num = int(SAMPLE_NUM * TRAIN_RATIO)
    val_num = int(SAMPLE_NUM * VAL_RATIO)
    SAMPLE_NUM - train_num - val_num  # 兜底，避免浮点计算误差导致总数不对

    # 5. 划分数据集
    train_samples = sample_samples[:train_num]
    val_samples = sample_samples[train_num : train_num + val_num]
    test_samples = sample_samples[train_num + val_num :]

    print(f"📊 划分结果：训练集 {len(train_samples)} 张 | 验证集 {len(val_samples)} 张 | 测试集 {len(test_samples)} 张")

    # 6. 定义数据集类型和对应的样本列表
    datasets = [("train", train_samples), ("val", val_samples), ("test", test_samples)]

    # 7. 遍历创建目录 + 复制图片和标签文件
    for set_name, samples in datasets:
        # 创建子数据集的图片/标签目录
        img_save_dir = os.path.join(SAVE_ROOT, set_name, "images")
        label_save_dir = os.path.join(SAVE_ROOT, set_name, "labels")
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        # 复制文件
        for img_name in samples:
            # 复制图片
            src_img = os.path.join(IMG_ROOT, img_name)
            dst_img = os.path.join(img_save_dir, img_name)
            shutil.copyfile(src_img, dst_img)

            # 复制对应标签
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(LABEL_ROOT, label_name)
            dst_label = os.path.join(label_save_dir, label_name)
            shutil.copyfile(src_label, dst_label)

        print(f"✅ {set_name} 集已完成：{len(samples)} 个样本")

    print("\n🎉 所有操作完成！子数据集保存路径：", SAVE_ROOT)


if __name__ == "__main__":
    main()
