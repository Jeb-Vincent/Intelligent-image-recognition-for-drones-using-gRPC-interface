#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测结果准确率统计工具

功能:
- 遍历指定目录下所有算法文件夹（以编号开头，跳过13号）
- 对比 labels（真实标签）和 testlabels（预测标签）
- 按类别统计：真实标签数、预测正确数、正确率
- 支持从 algo_classes.yaml 读取类别名称

目录结构:
    DATA_ROOT/
    ├── 01_松线虫害识别/
    │   ├── images/
    │   ├── labels/        (真实标签)
    │   └── testlabels/    (预测标签)
    ├── 02_河道淤积识别/
    │   ├── images/
    │   ├── labels/
    │   └── testlabels/
    └── ...

使用方法:
    python accuracy_stats.py <数据目录路径> [--yaml <yaml配置文件路径>] [--iou <IoU阈值>]
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import yaml


# ============================================================
# 配置
# ============================================================

# IoU阈值，用于判断预测是否正确
DEFAULT_IOU_THRESHOLD = 0.5

# 支持的图片格式（用于统计图片数量）
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ============================================================
# 工具函数
# ============================================================

def load_algo_config_from_yaml(yaml_path: str) -> Dict:
    """从yaml文件加载算法配置"""
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        algorithms = config.get('algorithms', {})
        algo_config = {}
        
        for algo_id, algo_info in algorithms.items():
            algo_id = int(algo_id)
            
            classes = {}
            for cls_id, cls_info in algo_info.get('classes', {}).items():
                cls_id = int(cls_id)
                classes[cls_id] = cls_info.get('name_cn', cls_info.get('name', f'class_{cls_id}'))
            
            algo_config[algo_id] = {
                'name': algo_info.get('name', f'算法{algo_id}'),
                'classes': classes
            }
        
        return algo_config
    
    except Exception as e:
        print(f"[警告] 加载YAML配置失败: {e}")
        return {}


def extract_algo_id_from_folder(folder_name: str) -> Optional[int]:
    """从文件夹名称提取算法编号"""
    match = re.match(r'^(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def parse_yolo_label(label_path: str) -> List[Dict]:
    """解析YOLO格式的标签文件，返回归一化坐标"""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # 转换为 xyxy 格式（归一化坐标）
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    labels.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2]
                    })
    except Exception as e:
        pass
    
    return labels


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个框的IoU（归一化坐标）"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def match_predictions_to_ground_truth(
    predictions: List[Dict], 
    ground_truths: List[Dict], 
    iou_threshold: float = 0.5
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    将预测框与真实框进行匹配
    
    返回:
        gt_count: 每个类别的真实标签数
        tp_count: 每个类别的正确预测数（True Positive）
        pred_count: 每个类别的预测标签数
    """
    gt_count = defaultdict(int)
    tp_count = defaultdict(int)
    pred_count = defaultdict(int)
    
    # 统计真实标签数
    for gt in ground_truths:
        gt_count[gt['class_id']] += 1
    
    # 统计预测标签数
    for pred in predictions:
        pred_count[pred['class_id']] += 1
    
    if len(ground_truths) == 0 or len(predictions) == 0:
        return dict(gt_count), dict(tp_count), dict(pred_count)
    
    # 计算所有可能的匹配
    matches = []
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            if pred['class_id'] == gt['class_id']:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold:
                    matches.append((iou, i, j, pred['class_id']))
    
    # 按IoU降序排序，贪心匹配
    matches.sort(reverse=True, key=lambda x: x[0])
    
    matched_pred = set()
    matched_gt = set()
    
    for iou, pred_idx, gt_idx, class_id in matches:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            tp_count[class_id] += 1
    
    return dict(gt_count), dict(tp_count), dict(pred_count)


def scan_data_folders(data_root: str) -> List[Dict]:
    """扫描数据目录，找出所有算法文件夹"""
    if not os.path.exists(data_root):
        print(f"[错误] 数据目录不存在: {data_root}")
        return []
    
    folders = []
    
    for item in sorted(os.listdir(data_root)):
        item_path = os.path.join(data_root, item)
        
        if not os.path.isdir(item_path):
            continue
        
        algo_id = extract_algo_id_from_folder(item)
        if algo_id is None:
            continue
        
        # 跳过13号变化检测
        if algo_id == 13:
            continue
        
        # 检查必要的子目录
        labels_dir = os.path.join(item_path, 'labels')
        testlabels_dir = os.path.join(item_path, 'testlabels')
        
        if not os.path.exists(labels_dir):
            print(f"[跳过] 算法 {algo_id}: labels目录不存在")
            continue
        
        if not os.path.exists(testlabels_dir):
            print(f"[跳过] 算法 {algo_id}: testlabels目录不存在")
            continue
        
        folders.append({
            'algo_id': algo_id,
            'folder_name': item,
            'folder_path': item_path,
            'labels_dir': labels_dir,
            'testlabels_dir': testlabels_dir
        })
    
    return folders


def analyze_algorithm(
    algo_id: int,
    labels_dir: str,
    testlabels_dir: str,
    iou_threshold: float,
    class_names: Dict[int, str]
) -> Dict:
    """分析单个算法的检测结果"""
    
    # 获取所有标签文件
    gt_files = set(f for f in os.listdir(labels_dir) if f.endswith('.txt'))
    pred_files = set(f for f in os.listdir(testlabels_dir) if f.endswith('.txt'))
    
    # 合并所有文件名
    all_files = gt_files | pred_files
    
    # 统计
    total_gt_count = defaultdict(int)
    total_tp_count = defaultdict(int)
    total_pred_count = defaultdict(int)
    
    image_count = 0
    
    for filename in all_files:
        gt_path = os.path.join(labels_dir, filename)
        pred_path = os.path.join(testlabels_dir, filename)
        
        ground_truths = parse_yolo_label(gt_path)
        predictions = parse_yolo_label(pred_path)
        
        gt_count, tp_count, pred_count = match_predictions_to_ground_truth(
            predictions, ground_truths, iou_threshold
        )
        
        for cls_id, count in gt_count.items():
            total_gt_count[cls_id] += count
        for cls_id, count in tp_count.items():
            total_tp_count[cls_id] += count
        for cls_id, count in pred_count.items():
            total_pred_count[cls_id] += count
        
        image_count += 1
    
    # 整理结果
    all_classes = set(total_gt_count.keys()) | set(total_pred_count.keys())
    
    class_stats = {}
    for cls_id in sorted(all_classes):
        gt = total_gt_count.get(cls_id, 0)
        tp = total_tp_count.get(cls_id, 0)
        pred = total_pred_count.get(cls_id, 0)
        
        # 召回率 = TP / GT（预测正确数 / 真实标签数）
        recall = tp / gt if gt > 0 else 0
        # 精确率 = TP / Pred（预测正确数 / 预测总数）
        precision = tp / pred if pred > 0 else 0
        
        class_stats[cls_id] = {
            'name': class_names.get(cls_id, f'class_{cls_id}'),
            'gt_count': gt,
            'pred_count': pred,
            'tp_count': tp,
            'recall': recall,
            'precision': precision
        }
    
    return {
        'image_count': image_count,
        'class_stats': class_stats,
        'total_gt': sum(total_gt_count.values()),
        'total_pred': sum(total_pred_count.values()),
        'total_tp': sum(total_tp_count.values())
    }


def print_algorithm_stats(algo_id: int, algo_name: str, stats: Dict):
    """打印单个算法的统计结果"""
    print(f"\n{'='*80}")
    print(f"算法 {algo_id}: {algo_name}")
    print(f"{'='*80}")
    print(f"  分析图片数: {stats['image_count']}")
    print(f"  总真实标签: {stats['total_gt']}")
    print(f"  总预测标签: {stats['total_pred']}")
    print(f"  总正确预测: {stats['total_tp']}")
    
    if stats['total_gt'] > 0:
        overall_recall = stats['total_tp'] / stats['total_gt'] * 100
        print(f"  总体召回率: {overall_recall:.2f}%")
    if stats['total_pred'] > 0:
        overall_precision = stats['total_tp'] / stats['total_pred'] * 100
        print(f"  总体精确率: {overall_precision:.2f}%")
    
    print(f"\n  {'类别ID':<8} {'类别名称':<20} {'真实数':<10} {'预测数':<10} {'正确数':<10} {'召回率':<12} {'精确率':<12}")
    print(f"  {'-'*90}")
    
    for cls_id, cls_stats in stats['class_stats'].items():
        recall_str = f"{cls_stats['recall']*100:.2f}%" if cls_stats['gt_count'] > 0 else "N/A"
        precision_str = f"{cls_stats['precision']*100:.2f}%" if cls_stats['pred_count'] > 0 else "N/A"
        
        print(f"  {cls_id:<8} {cls_stats['name']:<20} {cls_stats['gt_count']:<10} "
              f"{cls_stats['pred_count']:<10} {cls_stats['tp_count']:<10} "
              f"{recall_str:<12} {precision_str:<12}")


def main():
    parser = argparse.ArgumentParser(description='YOLO检测结果准确率统计工具')
    parser.add_argument('path', help='数据目录路径（包含算法文件夹）')
    parser.add_argument('--yaml', '-y', default='', help='algo_classes.yaml配置文件路径')
    parser.add_argument('--iou', '-i', type=float, default=DEFAULT_IOU_THRESHOLD, 
                        help=f'IoU阈值 (默认: {DEFAULT_IOU_THRESHOLD})')
    args = parser.parse_args()
    
    data_root = args.path.strip('"').strip("'")
    yaml_path = args.yaml.strip('"').strip("'") if args.yaml else ''
    iou_threshold = args.iou
    
    print("=" * 80)
    print("  YOLO 检测结果准确率统计工具")
    print("=" * 80)
    print(f"  数据目录: {data_root}")
    print(f"  配置文件: {yaml_path if yaml_path else '未指定'}")
    print(f"  IoU阈值: {iou_threshold}")
    print("=" * 80)
    
    # 加载算法配置
    algo_config = load_algo_config_from_yaml(yaml_path)
    if algo_config:
        print(f"\n✓ 已加载 {len(algo_config)} 个算法配置")
    
    # 扫描文件夹
    print("\n扫描数据目录...")
    folders = scan_data_folders(data_root)
    
    if not folders:
        print("\n没有找到可分析的算法文件夹。")
        print("请确保目录结构正确：每个算法文件夹下需要有 labels/ 和 testlabels/ 子目录")
        return
    
    print(f"\n找到 {len(folders)} 个算法文件夹待分析")
    
    # 汇总统计
    all_results = []
    
    # 分析每个算法
    for folder in folders:
        algo_id = folder['algo_id']
        algo_info = algo_config.get(algo_id, {})
        algo_name = algo_info.get('name', folder['folder_name'])
        class_names = algo_info.get('classes', {})
        
        stats = analyze_algorithm(
            algo_id,
            folder['labels_dir'],
            folder['testlabels_dir'],
            iou_threshold,
            class_names
        )
        
        print_algorithm_stats(algo_id, algo_name, stats)
        
        all_results.append({
            'algo_id': algo_id,
            'algo_name': algo_name,
            'stats': stats
        })
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("  汇总报告")
    print("=" * 80)
    print(f"\n  {'算法ID':<8} {'算法名称':<25} {'真实数':<10} {'预测数':<10} {'正确数':<10} {'召回率':<12} {'精确率':<12}")
    print(f"  {'-'*95}")
    
    total_gt_all = 0
    total_pred_all = 0
    total_tp_all = 0
    
    for result in all_results:
        stats = result['stats']
        total_gt_all += stats['total_gt']
        total_pred_all += stats['total_pred']
        total_tp_all += stats['total_tp']
        
        recall = stats['total_tp'] / stats['total_gt'] * 100 if stats['total_gt'] > 0 else 0
        precision = stats['total_tp'] / stats['total_pred'] * 100 if stats['total_pred'] > 0 else 0
        
        print(f"  {result['algo_id']:<8} {result['algo_name']:<25} {stats['total_gt']:<10} "
              f"{stats['total_pred']:<10} {stats['total_tp']:<10} "
              f"{recall:>6.2f}%     {precision:>6.2f}%")
    
    print(f"  {'-'*95}")
    overall_recall = total_tp_all / total_gt_all * 100 if total_gt_all > 0 else 0
    overall_precision = total_tp_all / total_pred_all * 100 if total_pred_all > 0 else 0
    print(f"  {'总计':<8} {'':<25} {total_gt_all:<10} {total_pred_all:<10} {total_tp_all:<10} "
          f"{overall_recall:>6.2f}%     {overall_precision:>6.2f}%")
    
    print("\n" + "=" * 80)
    print("  分析完成！")
    print("=" * 80)
    print("\n指标说明:")
    print("  - 召回率(Recall) = 正确预测数 / 真实标签数 (检测出了多少真实目标)")
    print("  - 精确率(Precision) = 正确预测数 / 预测总数 (预测的有多少是对的)")
    print()


if __name__ == '__main__':
    main()
