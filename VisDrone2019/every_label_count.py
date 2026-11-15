import os
import yaml
from collections import defaultdict

# ========================== 配置参数 ==========================
# YAML配置文件路径（替换为你的实际路径）
YAML_PATH = "UAV/UAV3/UAV3.yaml"
# =============================================================

def load_yaml_config(yaml_path):
    """加载YAML配置文件，获取数据集根路径和类别名称"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取关键信息
    dataset_root = config.get('path')  # 数据集根路径（dataset目录）
    class_names = config.get('names')  # 类别名称字典 {0: 'small-vehicle', ...}
    nc = config.get('nc')  # 类别数量
    
    # 校验必要字段
    if not dataset_root or not class_names or not nc:
        raise ValueError("YAML文件缺少必要字段（path/names/nc）")
    
    # 转换类别名称为有序列表（按ID排序）
    class_names_sorted = [class_names[i] for i in range(nc)]
    return dataset_root, class_names_sorted, nc

def count_labels_in_subset(subset_name, dataset_root, class_names):
    """统计单个子集（train/val/test）的类别标签数量（仅依赖YAML类别配置）"""
    # 构建子集labels目录路径
    labels_dir = os.path.join(dataset_root, subset_name, "labels")
    if not os.path.exists(labels_dir):
        print(f"警告：{subset_name}的labels目录不存在 → {labels_dir}")
        return defaultdict(int)
    
    # 初始化类别计数
    label_count = defaultdict(int)
    # 遍历所有标签文件（排除可能存在的classes.txt，避免误读）
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != "classes.txt"]
    
    for file_name in label_files:
        file_path = os.path.join(labels_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # 统计每行的类别ID（仅基于YAML的类别映射）
            for line in lines:
                parts = line.split()
                if len(parts) < 1:
                    continue  # 跳过空行
                try:
                    class_id = int(parts[0])  # 第一个字段是类别ID
                    # 仅当类别ID在YAML定义的范围内时统计
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                        label_count[class_name] += 1
                    else:
                        print(f"警告：{file_name} 中存在超出范围的类别ID → {class_id}（YAML定义类别数：{len(class_names)}）")
                except ValueError:
                    # 跳过非数字的类别ID
                    print(f"警告：{file_name} 中存在非数字类别ID → {parts[0]}")
                    continue
        except Exception as e:
            print(f"读取{file_name}失败：{e}")
            continue
    
    return label_count

def main():
    # 1. 加载YAML配置
    print("="*60)
    print("开始统计DOTA数据集类别标签数量（仅依赖YAML配置）")
    print("="*60)
    try:
        dataset_root, class_names, nc = load_yaml_config(YAML_PATH)
        print(f"数据集根路径：{dataset_root}")
        print(f"类别数量：{nc}")
        print(f"类别列表（按ID排序）：{class_names}")
    except Exception as e:
        print(f"加载YAML失败：{e}")
        return
    
    # 2. 定义要统计的子集
    subsets = ['train', 'val', 'test']
    
    # 3. 初始化总统计字典
    total_count = defaultdict(int)
    subset_stats = {}
    
    # 4. 逐个统计子集
    print("\n" + "-"*60)
    for subset in subsets:
        print(f"统计{subset}集标签...")
        count = count_labels_in_subset(subset, dataset_root, class_names)
        subset_stats[subset] = count
        
        # 累加到总数
        for cls in count:
            total_count[cls] += count[cls]
    
    # 5. 补全所有类别的计数（包括数量为0的类别）
    for subset in subsets:
        for cls in class_names:
            if cls not in subset_stats[subset]:
                subset_stats[subset][cls] = 0
    for cls in class_names:
        if cls not in total_count:
            total_count[cls] = 0
    
    # 6. 格式化输出统计结果
    print("\n" + "="*60)
    print("数据集类别标签统计结果")
    print("="*60)
    
    # 打印表头
    header = f"{'类别名称':<15} {'训练集':<8} {'验证集':<8} {'测试集':<8} {'总计':<8}"
    print(header)
    print("-"*60)
    
    # 打印每个类别的统计
    for cls in class_names:
        train_num = subset_stats['train'][cls]
        val_num = subset_stats['val'][cls]
        test_num = subset_stats['test'][cls]
        total_num = total_count[cls]
        row = f"{cls:<15} {train_num:<8} {val_num:<8} {test_num:<8} {total_num:<8}"
        print(row)
    
    # 打印子集总数
    print("-"*60)
    train_total = sum(subset_stats['train'].values())
    val_total = sum(subset_stats['val'].values())
    test_total = sum(subset_stats['test'].values())
    grand_total = sum(total_count.values())
    total_row = f"{'总数':<15} {train_total:<8} {val_total:<8} {test_total:<8} {grand_total:<8}"
    print(total_row)
    
    # 打印比例
    print("-"*60)
    print("各子集标签占比：")
    print(f"训练集：{train_total/grand_total:.2%} ({train_total}/{grand_total})")
    print(f"验证集：{val_total/grand_total:.2%} ({val_total}/{grand_total})")
    print(f"测试集：{test_total/grand_total:.2%} ({test_total}/{grand_total})")

if __name__ == '__main__':
    main()