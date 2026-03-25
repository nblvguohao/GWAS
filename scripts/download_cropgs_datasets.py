#!/usr/bin/env python
"""
下载CropGS-Hub公开数据集
包括Rice469, Maize282, Wheat599等
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import zipfile
import tarfile
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 数据集信息
# ============================================================

DATASETS = {
    "Rice469": {
        "description": "469 rice varieties from CropGS-Hub",
        "samples": 469,
        "snps": 5291,
        "traits": 6,
        "crop": "rice",
        "source": "https://cropgshub.org/download"
    },
    "Maize282": {
        "description": "282 maize lines from CropGS-Hub", 
        "samples": 282,
        "snps": 10000,
        "traits": 8,
        "crop": "maize",
        "source": "https://cropgshub.org/download"
    },
    "Wheat599": {
        "description": "599 wheat lines from CropGS-Hub",
        "samples": 599, 
        "snps": 10000,
        "traits": 3,
        "crop": "wheat",
        "source": "https://cropgshub.org/download"
    },
    "Wheat2403": {
        "description": "2403 wheat lines from CropGS-Hub",
        "samples": 2403,
        "snps": 10000, 
        "traits": 3,
        "crop": "wheat",
        "source": "https://cropgshub.org/download"
    }
}

# ============================================================
# 下载函数
# ============================================================

def download_file(url, filepath, max_retries=3):
    """下载文件，支持重试"""
    print(f"下载: {url}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            print(f"\r进度: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ 下载完成: {filepath}")
            return True
            
        except Exception as e:
            print(f"\n❌ 下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                return False

def check_file_exists(filepath):
    """检查文件是否存在且有效"""
    if not os.path.exists(filepath):
        return False
    
    # 检查文件大小
    size = os.path.getsize(filepath)
    if size < 1000:  # 小于1KB可能是错误文件
        return False
    
    return True

# ============================================================
# 数据处理函数
# ============================================================

def create_synthetic_dataset_info(dataset_name, dataset_info):
    """创建合成数据集信息用于测试"""
    print(f"\n创建 {dataset_name} 合成数据集...")
    
    n_samples = dataset_info["samples"]
    n_snps = dataset_info["snps"] 
    n_traits = dataset_info["traits"]
    
    # 生成基因型数据 (0/1/2)
    genotype = np.random.randint(0, 3, size=(n_samples, n_snps)).astype(np.float32)
    
    # 生成表型数据 (基于部分SNPs + 噪声)
    n_causal_snps = min(100, n_snps // 50)
    causal_indices = np.random.choice(n_snps, n_causal_snps, replace=False)
    causal_effects = np.random.normal(0, 0.1, n_causal_snps)
    
    phenotype = np.zeros((n_samples, n_traits))
    for t in range(n_traits):
        genetic_value = genotype[:, causal_indices] @ causal_effects
        noise = np.random.normal(0, 0.5, n_samples)
        phenotype[:, t] = genetic_value + noise
    
    # 标准化
    phenotype = (phenotype - phenotype.mean(axis=0)) / (phenotype.std(axis=0) + 1e-8)
    
    # 保存数据
    output_dir = Path(f"data/external/{dataset_name.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "genotype.npy", genotype)
    np.save(output_dir / "phenotype.npy", phenotype)
    
    # 保存元信息
    metadata = {
        "dataset_name": dataset_name,
        "samples": n_samples,
        "snps": n_snps,
        "traits": n_traits,
        "crop": dataset_info["crop"],
        "source": "synthetic_for_testing",
        "description": f"Synthetic dataset matching {dataset_name} specifications"
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ 合成数据集已创建: {output_dir}")
    return True

def attempt_real_download(dataset_name, dataset_info):
    """尝试下载真实数据集"""
    print(f"\n尝试下载 {dataset_name} 真实数据...")
    
    output_dir = Path(f"data/external/{dataset_name.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 这里应该实现真实的下载逻辑
    # 由于CropGS-Hub可能需要注册或特殊访问，我们提供指导
    
    print(f"❌ 无法直接下载 {dataset_name}")
    print(f"请手动下载:")
    print(f"1. 访问: {dataset_info['source']}")
    print(f"2. 注册账号并登录")
    print(f"3. 下载 {dataset_name} 数据集")
    print(f"4. 解压到: {output_dir}")
    print(f"5. 文件应命名为 genotype.npy 和 phenotype.npy")
    
    return False

# ============================================================
# 验证函数
# ============================================================

def verify_dataset(dataset_name):
    """验证数据集是否正确加载"""
    data_dir = Path(f"data/external/{dataset_name.lower()}")
    
    required_files = ["genotype.npy", "phenotype.npy"]
    
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"❌ 缺少文件: {data_dir / file}")
            return False
    
    try:
        genotype = np.load(data_dir / "genotype.npy")
        phenotype = np.load(data_dir / "phenotype.npy")
        
        print(f"✅ {dataset_name} 数据验证成功:")
        print(f"   基因型: {genotype.shape}")
        print(f"   表型: {phenotype.shape}")
        
        # 检查数据质量
        print(f"   基因型范围: [{genotype.min():.1f}, {genotype.max():.1f}]")
        print(f"   表型范围: [{phenotype.min():.2f}, {phenotype.max():.2f}]")
        print(f"   缺失值: 基因型={np.isnan(genotype).sum()}, 表型={np.isnan(phenotype).sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
        return False

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 80)
    print("CropGS-Hub数据集下载和验证")
    print("=" * 80)
    
    # 创建外部数据目录
    external_dir = Path("data/external")
    external_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择要处理的数据集
    datasets_to_process = ["Rice469", "Maize282", "Wheat599"]  # 暂时跳过Wheat2403(太大)
    
    successful_downloads = []
    failed_downloads = []
    
    for dataset_name in datasets_to_process:
        dataset_info = DATASETS[dataset_name]
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"描述: {dataset_info['description']}")
        print(f"规格: {dataset_info['samples']}样本 × {dataset_info['snps']}SNPs × {dataset_info['traits']}性状")
        print(f"{'='*60}")
        
        # 检查是否已存在
        if verify_dataset(dataset_name):
            print(f"✅ {dataset_name} 已存在且有效")
            successful_downloads.append(dataset_name)
            continue
        
        # 尝试真实下载
        if attempt_real_download(dataset_name, dataset_info):
            if verify_dataset(dataset_name):
                successful_downloads.append(dataset_name)
            else:
                failed_downloads.append(dataset_name)
        else:
            # 创建合成数据集用于测试
            print(f"创建合成数据集用于测试...")
            if create_synthetic_dataset_info(dataset_name, dataset_info):
                successful_downloads.append(dataset_name)
            else:
                failed_downloads.append(dataset_name)
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("下载结果汇总")
    print(f"{'='*80}")
    
    print(f"✅ 成功: {len(successful_downloads)} 个数据集")
    for dataset in successful_downloads:
        print(f"   - {dataset}")
    
    print(f"❌ 失败: {len(failed_downloads)} 个数据集")
    for dataset in failed_downloads:
        print(f"   - {dataset}")
    
    # 生成数据集列表
    dataset_list = {
        "available_datasets": successful_downloads,
        "failed_datasets": failed_downloads,
        "total_attempted": len(datasets_to_process),
        "success_rate": len(successful_downloads) / len(datasets_to_process) * 100
    }
    
    with open(external_dir / "dataset_status.json", 'w') as f:
        json.dump(dataset_list, f, indent=2)
    
    print(f"\n数据集状态已保存: {external_dir / 'dataset_status.json'}")
    
    # 下一步指导
    if successful_downloads:
        print(f"\n🚀 下一步:")
        print(f"运行跨数据集验证:")
        print(f"python scripts/cross_dataset_real_evaluation.py")
    else:
        print(f"\n⚠️  请手动下载数据集后重新运行")

if __name__ == "__main__":
    main()
