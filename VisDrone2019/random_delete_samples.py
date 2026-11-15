import os
import random
import sys

# ===================== 配置区域（请确认路径正确性）=====================
# 图片文件夹路径
IMAGE_DIR = r"E:\Dingzhenhao\python_workplace\UAV\DOTA\images"
# 标签文件夹路径
LABEL_DIR = r"E:\Dingzhenhao\python_workplace\UAV\DOTA\labels"
# classes.txt文件路径
CLASSES_PATH = r"E:\Dingzhenhao\python_workplace\UAV\DOTA\labels\classes.txt"
# 保护类别（包含该类别的样本不允许删除）
PROTECT_CLASS = "ship"
# 计划删除的样本数量（目标值）
PLANNED_DELETE_COUNT = 2000
# 随机种子（可选，设置后可复现抽样结果，None则随机）
RANDOM_SEED = 42


# =====================================================================

def load_classes(classes_path):
    """读取classes.txt，返回{类别名:索引, 索引:类别名}的双向映射"""
    class2idx = {}
    idx2class = {}
    try:
        with open(classes_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            for idx, cls_name in enumerate(lines):
                class2idx[cls_name] = idx
                idx2class[idx] = cls_name
        return class2idx, idx2class
    except Exception as e:
        print(f"❌ 读取classes.txt失败：{e}")
        sys.exit(1)


def get_valid_samples(image_dir, label_dir):
    """
    获取图片-标签对应的有效样本字典
    返回：{样本基础名: (图片路径, 标签路径)}
    """
    valid_samples = {}
    # 支持的图片扩展名
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')

    # 先收集所有标签文件
    label_files = {}
    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            base_name = os.path.splitext(fname)[0]
            label_files[base_name] = os.path.join(label_dir, fname)

    # 匹配图片文件
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(img_exts):
            base_name = os.path.splitext(fname)[0]
            if base_name in label_files:
                img_path = os.path.join(image_dir, fname)
                label_path = label_files[base_name]
                valid_samples[base_name] = (img_path, label_path)

    return valid_samples


def has_protect_class(label_path, protect_idx):
    """检查标签文件是否包含保护类别（ship）"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        for line in lines:
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                cls_idx = int(parts[0])
                if cls_idx == protect_idx:
                    return True
            except ValueError:
                print(f"⚠️  警告：标签行格式错误 → {label_path} | 行：{line}")
        return False
    except Exception as e:
        print(f"⚠️  警告：读取标签文件失败（视为含保护类，不删除）→ {label_path} | 错误：{e}")
        return True


def main():
    # 安全提醒
    print("⚠️  重要提醒：请先备份你的数据集，避免误删！")
    input("按下 Enter 键继续（或 Ctrl+C 终止）...\n")

    # 1. 加载类别映射并验证保护类别
    class2idx, idx2class = load_classes(CLASSES_PATH)
    if PROTECT_CLASS not in class2idx:
        print(f"❌ 错误：classes.txt中未找到保护类别「{PROTECT_CLASS}」")
        print(f"当前classes.txt包含的类别：{list(class2idx.keys())}")
        sys.exit(1)
    protect_idx = class2idx[PROTECT_CLASS]
    print(f"✅ 成功加载类别映射，{PROTECT_CLASS} 对应的索引为：{protect_idx}")

    # 2. 获取所有有效样本（图片+标签对应）
    valid_samples = get_valid_samples(IMAGE_DIR, LABEL_DIR)
    print(f"✅ 找到有效样本（图片+标签对应）数量：{len(valid_samples)}")

    if len(valid_samples) == 0:
        print(f"❌ 错误：未找到任何有效样本（图片+标签对应）")
        sys.exit(1)

    # 3. 筛选出可删除的样本（无ship标签）
    deletable_samples = {}
    protected_samples = {}
    for base_name, (img_path, label_path) in valid_samples.items():
        if has_protect_class(label_path, protect_idx):
            protected_samples[base_name] = (img_path, label_path)
        else:
            deletable_samples[base_name] = (img_path, label_path)

    print(f"✅ 含「{PROTECT_CLASS}」标签的受保护样本数量：{len(protected_samples)}")
    print(f"✅ 无「{PROTECT_CLASS}」标签的可删除样本数量：{len(deletable_samples)}")

    # 4. 确定实际删除数量（核心调整：不足2000则删除全部可删除样本）
    actual_delete_count = min(PLANNED_DELETE_COUNT, len(deletable_samples))
    if actual_delete_count < PLANNED_DELETE_COUNT:
        print(f"\n⚠️  提示：可删除样本数量（{len(deletable_samples)}）小于计划删除数量（{PLANNED_DELETE_COUNT}）")
        print(f"👉 将调整为删除所有可删除样本，实际删除数量：{actual_delete_count}")
    else:
        print(f"\n✅ 可删除样本数量充足，将按计划删除 {PLANNED_DELETE_COUNT} 个样本")

    # 无可用删除样本的情况
    if actual_delete_count == 0:
        print(f"\n✅ 未找到可删除的样本（所有样本均含{PROTECT_CLASS}标签），无需删除")
        sys.exit(0)

    # 5. 随机抽样待删除的样本
    random.seed(RANDOM_SEED)  # 设置随机种子确保可复现
    delete_base_names = random.sample(list(deletable_samples.keys()), actual_delete_count)
    to_delete = [deletable_samples[base_name] for base_name in delete_base_names]

    # 6. 输出统计信息并确认删除
    print("\n" + "=" * 60)
    print(f"📊 最终抽样结果：")
    print(f"   计划删除样本数：{PLANNED_DELETE_COUNT}")
    print(f"   实际删除样本数：{actual_delete_count}")
    print(f"   可删除样本总数：{len(deletable_samples)}")
    print(f"   受保护样本数（含{PROTECT_CLASS}）：{len(protected_samples)}")
    print(f"   剩余样本数（删除后）：{len(valid_samples) - actual_delete_count}")
    print("=" * 60)

    # 展示待删除文件列表（前10个示例，避免输出过长）
    print("\n🔍 待删除文件示例（前10个）：")
    for idx, (img_path, label_path) in enumerate(to_delete[:10], 1):
        print(f"   {idx}. 图片：{img_path}")
        print(f"      标签：{label_path}")
    if len(to_delete) > 10:
        print(f"   ... 还有 {len(to_delete) - 10} 个文件待删除")

    # 安全确认
    confirm = input(f"\n⚠️  确认删除以上选中的 {actual_delete_count} 个样本？(输入 y 确认，其他键取消)：")
    if confirm.lower() != 'y':
        print("\n🚫 用户取消了删除操作")
        sys.exit(0)

    # 7. 执行删除操作
    deleted_img_count = 0
    deleted_label_count = 0
    failed = []

    print("\n🚀 开始执行删除操作...")
    for img_path, label_path in to_delete:
        # 删除图片
        try:
            os.remove(img_path)
            deleted_img_count += 1
        except Exception as e:
            failed.append(f"图片删除失败：{img_path} | 错误：{e}")

        # 删除标签
        try:
            os.remove(label_path)
            deleted_label_count += 1
        except Exception as e:
            failed.append(f"标签删除失败：{label_path} | 错误：{e}")

    # 8. 输出最终结果
    print("\n" + "=" * 60)
    print(f"🎉 删除操作完成！")
    print(f"   计划删除样本数：{PLANNED_DELETE_COUNT}")
    print(f"   实际删除样本数：{actual_delete_count}")
    print(f"   成功删除图片数：{deleted_img_count}")
    print(f"   成功删除标签数：{deleted_label_count}")
    print(f"   失败数：{len(failed)}")
    if failed:
        print("❌ 失败详情（前10个）：")
        for err in failed[:10]:
            print(f"   - {err}")
        if len(failed) > 10:
            print(f"   - ... 还有 {len(failed) - 10} 个失败项")
    print("=" * 60)


if __name__ == "__main__":
    main()