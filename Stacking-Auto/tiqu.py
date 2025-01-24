import pandas as pd
import numpy as np

def extract_enhancer_region_dynamic(probabilities, fragment_size=200, overlap_size=50, min_region_length=3):
    """
    动态确定所有增强子区域，并合并局部的低概率区域。
    
    参数：
    probabilities (list): 每个片段的分类概率值
    fragment_size (int): 每个片段的大小，默认200个碱基对
    overlap_size (int): 每个片段之间的重叠大小，默认50个碱基对
    min_region_length (int): 最小的连续片段数量，避免过小的增强子区域，默认3
    
    返回：
    list: 所有增强子区域的起始和结束索引
    """
    # 计算动态阈值
    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    threshold = mean_prob - 0.3 * std_prob  # 动态阈值
    print(f"动态阈值: {threshold}")

    # 寻找概率较高的区域
    positive_fragments = []
    for idx, prob in enumerate(probabilities):
        if prob > threshold:
            positive_fragments.append(idx)
    
    if not positive_fragments:
        return []  # 没有找到符合条件的片段
    
    # 合并连续的片段并剔除过小的区域
    enhancer_regions = []
    start = positive_fragments[0]
    end = start
    
    for i in range(1, len(positive_fragments)):
        if positive_fragments[i] == positive_fragments[i - 1] + 1:
            end = positive_fragments[i]
        else:
            if (end - start + 1) >= min_region_length:
                enhancer_regions.append((start, end))
            start = positive_fragments[i]
            end = start
    
    if (end - start + 1) >= min_region_length:
        enhancer_regions.append((start, end))  # 添加最后一段
    
    return enhancer_regions

def merge_regions(enhancer_regions, fragment_size, overlap_size):
    """
    合并部分重叠或相邻的增强子区域。
    
    参数：
    enhancer_regions (list): 增强子区域列表，每个区域为 (start_idx, end_idx)
    fragment_size (int): 每个片段的大小
    overlap_size (int): 每个片段之间的重叠大小
    
    返回：
    list: 合并后的增强子区域列表
    """
    if not enhancer_regions:
        return []

    # 按起始索引排序
    enhancer_regions.sort(key=lambda x: x[0])
    merged_regions = []
    current_start, current_end = enhancer_regions[0]

    for start, end in enhancer_regions[1:]:
        # 检查是否有重叠或相邻
        current_actual_end = fragment_size + current_end * overlap_size
        next_actual_start = start * overlap_size
        if next_actual_start <= current_actual_end:  # 有重叠或相邻
            current_end = max(current_end, end)
        else:
            merged_regions.append((current_start, current_end))
            current_start, current_end = start, end

    # 添加最后一个区域
    merged_regions.append((current_start, current_end))
    return merged_regions

def process_samples(
    probabilities, 
    sample_size=77, 
    fragment_size=200, 
    overlap_size=50, 
    min_region_length=3, 
    verbose=True
):
    """
    处理分类概率值，按每77个值为一个样本分开计算增强子区域，并输出每个样本概率最高的三个区域。

    参数：
    probabilities (list): 所有片段的分类概率值
    sample_size (int): 每个样本包含的概率值数量，默认77
    fragment_size (int): 每个片段的大小，默认200个碱基对
    overlap_size (int): 每个片段之间的重叠大小，默认50个碱基对
    min_region_length (int): 最小的连续片段数量，默认3
    verbose (bool): 是否输出详细日志信息，默认True

    返回：
    list: 每个样本的概率最高的三个增强子区域信息
    """
    results = []
    num_samples = len(probabilities) // sample_size

    for i in range(num_samples):
        # 获取当前样本的概率值
        sample_probabilities = probabilities[i * sample_size: (i + 1) * sample_size]
        
        # 提取增强子区域
        enhancer_regions = extract_enhancer_region_dynamic(
            sample_probabilities, 
            fragment_size=fragment_size, 
            overlap_size=overlap_size, 
            min_region_length=min_region_length
        )
        
        # 合并相邻或部分重叠的区域
        merged_regions = merge_regions(enhancer_regions, fragment_size, overlap_size)
        
        # 根据区域内的最大概率对区域排序，取概率最高的三个区域
        sorted_regions = sorted(
            merged_regions, 
            key=lambda region: max(sample_probabilities[region[0]:region[1] + 1]), 
            reverse=True
        )
        top_regions = sorted_regions[:3]
        
        # 保存每个样本的增强子区域信息
        for region in top_regions:
            start_index, end_index = region
            actual_start = start_index * overlap_size
            actual_end = fragment_size + end_index * overlap_size
            
            results.append({
                'sample_index': i,
                'start': actual_start,
                'end': actual_end,
                'start_idx': start_index,
                'end_idx': end_index
            })
    
    if verbose:
        if results:
            print("所有增强子区域：")
            for region in results:
                print(f"Sample {region['sample_index']}:")
                print(f"  Enhancer Region (Position): {region['start']} to {region['end']}")
                print(f"  Enhancer Region (Index): {region['start_idx']} to {region['end_idx']}")
                print("---")
        else:
            print("未找到增强子区域。")
    
    return results

# 读取CSV文件
csv_file = "predictions_with_proba.csv"  # 替换为实际的文件路径
data = pd.read_csv(csv_file)

# 提取概率列，假设概率列名为 '1'
probabilities = data['1'].values

# 处理每77个分类概率值作为一个样本，输出每个样本概率最高的三个增强子区域
enhancer_regions = process_samples(probabilities, sample_size=77, fragment_size=200, overlap_size=50)
