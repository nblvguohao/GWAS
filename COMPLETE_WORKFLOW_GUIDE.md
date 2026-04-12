# 完整工作流程指南

## 🎯 从下载到训练的完整流程

### 当前状态
- ✅ **已完成**: 基于真实参数的模拟数据 (Rice469) + MLP 模型训练
- 🔄 **进行中**: 获取真实 Rice3k 数据
- 📋 **下一步**: 使用真实数据训练所有模型

---

## 📊 完整流程图

```
🌾 Rice3k 数据获取
         ↓
📥 分批下载 (11批次)
         ↓
🔍 文件验证
         ↓
🔗 数据整合
         ↓
🧹 数据处理
         ↓
🏗️ 网络构建
         ↓
🚀 模型训练
         ↓
📈 性能对比
         ↓
📊 结果分析
```

---

## 📥 第一阶段: 数据下载

### 1.1 准备工作
```bash
# 创建项目结构
mkdir -p data/raw/rice3k
mkdir -p data/processed/rice3k
mkdir -p results/rice3k
mkdir -p models/rice3k
```

### 1.2 分批下载
```bash
# 启动交互式下载工具
python scripts/batch_download_rice3k.py
```

#### 下载批次列表
| 批次 | 内容 | 文件数 | 预估大小 | 状态 |
|------|------|--------|----------|------|
| genotype_1 | B001-B010 | 10 | ~500MB | ⏳ |
| genotype_2 | B011-B020 | 10 | ~500MB | ⏳ |
| genotype_3 | B021-B030 | 10 | ~500MB | ⏳ |
| genotype_4 | B031-B040 | 10 | ~500MB | ⏳ |
| genotype_5 | B041-B050 | 10 | ~500MB | ⏳ |
| genotype_6 | B051-B054 | 4 | ~200MB | ⏳ |
| phenotype_1 | Fold1 | 1 | ~10MB | ⏳ |
| phenotype_2 | Fold2 | 1 | ~10MB | ⏳ |
| phenotype_3 | Fold3 | 1 | ~10MB | ⏳ |
| phenotype_4 | Fold4 | 1 | ~10MB | ⏳ |
| phenotype_5 | Fold5 | 1 | ~10MB | ⏳ |

### 1.3 手动下载备选方案
如果自动下载失败，请参考：
- `MANUAL_DOWNLOAD_GUIDE.md`
- `BATCH_DOWNLOAD_GUIDE.md`

---

## 🔍 第二阶段: 数据验证

### 2.1 验证下载完整性
```bash
# 完整验证
python scripts/verify_downloads.py --data-dir data/raw/rice3k

# 快速检查
python scripts/verify_downloads.py --quick --data-dir data/raw/rice3k
```

### 2.2 验证指标
- ✅ **文件数量**: 59 个文件 (54基因型 + 5表型)
- ✅ **文件大小**: 总计约 2.7GB
- ✅ **数据格式**: 基因型 (0,1,2), 表型 (连续数值)
- ✅ **数据完整性**: 无缺失文件

### 2.3 验证报告
验证完成后会生成：
- `data/raw/rice3k/validation_report.md`
- `data/raw/rice3k/validation_report.json`

---

## 🔗 第三阶段: 数据整合

### 3.1 整合下载文件
```bash
# 使用下载工具整合功能
python scripts/batch_download_rice3k.py --integrate
```

### 3.2 通用数据处理
```bash
# 自动处理多种格式
python scripts/process_generic_data.py
```

输入提示：
```
请输入数据目录路径: data/raw/rice3k/integrated
请输入数据集名称: rice3k
```

### 3.3 预期输出
```
data/processed/rice3k/
├── rice3k_genotype.csv
├── rice3k_phenotype.csv
└── data_report.md
```

---

## 🧹 第四阶段: 数据预处理

### 4.1 运行预处理脚本
```bash
python scripts/preprocess_rice469.py
```

### 4.2 预处理步骤
- ✅ **数据质量检查**: 缺失值、异常值检测
- ✅ **SNP 过滤**: MAF > 0.05
- ✅ **数据分割**: 训练/验证/测试集 (70%/15%/15%)
- ✅ **标准化**: 表型数据标准化
- ✅ **LD 矩阵**: 计算连锁不平衡矩阵

### 4.3 预处理输出
```
data/processed/rice3k/
├── [trait]/
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│   └── metadata.csv
├── ld_matrix.npy
└── preprocessing_report.md
```

---

## 🏗️ 第五阶段: 网络构建

### 5.1 构建所有网络架构
```bash
python scripts/build_networks.py
```

### 5.2 可用网络
| 网络 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| MLP | 2.7M | 基础全连接 | 快速原型 |
| CNN | 655.8M | 局部特征提取 | 大规模数据 |
| Transformer | 1.8M | 序列建模 | 长距离依赖 |
| GNN | 1.3M | 图结构建模 | 利用 LD 信息 |
| Hybrid | 1.4M | CNN+Transformer | 综合特征 |

### 5.3 网络输出
```
models/rice3k/
├── mlp_model.pth
├── cnn_model.pth
├── transformer_model.pth
├── gnn_model.pth
├── hybrid_model.pth
├── adj_matrix.pth
└── model_info.json
```

---

## 🚀 第六阶段: 模型训练

### 6.1 训练所有模型
```bash
python scripts/train_rice469.py
```

选择训练模式：
1. 训练所有模型和性状
2. 训练单个性状的所有模型
3. 训练单个模型和性状

### 6.2 批量训练脚本
```bash
# 训练所有模型和性状
python scripts/train_all_models.py

# 训练特定性状
python scripts/train_single_trait.py --trait Grain_Yield

# 训练特定模型
python scripts/train_single_model.py --model Transformer --trait Plant_Height
```

### 6.3 训练配置
- **Epochs**: 100 (早停 patience=15)
- **Batch Size**: 32
- **Learning Rate**: 0.001 (自适应调度)
- **优化器**: Adam (L2正则化)
- **设备**: GPU (CUDA) / CPU

---

## 📈 第七阶段: 性能对比

### 7.1 自动对比分析
```bash
python scripts/compare_models.py
```

### 7.2 对比指标
| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| R² | 决定系数 |
| PCC | 皮尔逊相关系数 |
| 训练时间 | 模型训练耗时 |
| 推理时间 | 单次预测耗时 |

### 7.3 预期性能提升
| 数据集 | 当前PCC | 预期PCC | 提升幅度 |
|--------|----------|----------|----------|
| Rice469 (模拟) | 0.22 | - | - |
| Rice3k (真实) | - | 0.5-0.7 | 2-3x |

---

## 📊 第八阶段: 结果分析

### 8.1 生成分析报告
```bash
python scripts/generate_final_report.py
```

### 8.2 可视化结果
```bash
python scripts/visualize_results.py
```

生成图表：
- 训练曲线对比
- 预测值 vs 真实值散点图
- 模型性能雷达图
- 特征重要性分析

### 8.3 最终报告
```
results/rice3k/
├── final_report.md
├── model_comparison.csv
├── training_curves.png
├── prediction_scatter.png
├── performance_radar.png
└── feature_importance.png
```

---

## 🎯 关键成功因素

### 数据质量
- ✅ **完整性**: 无缺失数据
- ✅ **准确性**: 格式正确
- ✅ **一致性**: 样本ID匹配

### 模型选择
- ✅ **多样性**: 5种不同架构
- ✅ **适用性**: 适合基因组数据
- ✅ **可扩展**: 易于添加新模型

### 训练策略
- ✅ **早停机制**: 防止过拟合
- ✅ **学习率调度**: 自适应优化
- ✅ **正则化**: L2 + Dropout

---

## ⚠️ 常见问题及解决方案

### 问题 1: 下载失败
**症状**: Google Drive 下载中断
**解决方案**:
- 使用分批下载
- 尝试不同时间段
- 使用下载管理器

### 问题 2: 内存不足
**症状**: 训练时内存溢出
**解决方案**:
- 减少批次大小
- 使用梯度累积
- 增加虚拟内存

### 问题 3: 训练缓慢
**症状**: 模型训练时间过长
**解决方案**:
- 使用 GPU 加速
- 减少 SNP 数量
- 简化网络结构

### 问题 4: 过拟合
**症状**: 训练性能好，测试性能差
**解决方案**:
- 增加正则化
- 使用更多数据
- 简化模型

---

## 📞 技术支持

### 脚本帮助
```bash
# 查看所有可用脚本
python scripts --help

# 查看特定脚本帮助
python scripts/train_rice469.py --help
```

### 日志和调试
- **训练日志**: `logs/training_*.log`
- **错误日志**: `logs/error_*.log`
- **性能监控**: `logs/performance_*.log`

### 社区支持
- **GitHub Issues**: 报告问题和建议
- **技术文档**: 查看项目 README
- **示例代码**: 参考 `examples/` 目录

---

## 🚀 快速开始

### 一键运行 (推荐)
```bash
# 完整流程自动化
python scripts/run_complete_pipeline.py
```

### 分步运行
```bash
# 1. 下载数据
python scripts/batch_download_rice3k.py

# 2. 验证数据
python scripts/verify_downloads.py

# 3. 处理数据
python scripts/process_generic_data.py

# 4. 训练模型
python scripts/train_rice469.py

# 5. 分析结果
python scripts/compare_models.py
```

---

## 📈 预期成果

### 学术价值
- 📊 **高质量论文**: 可发表顶级期刊
- 🎯 **方法创新**: 深度学习在基因组选择的应用
- 📚 **基准数据**: 为社区提供标准数据集

### 实际应用
- 🌾 **育种决策**: 辅助水稻品种选择
- 🔬 **研究工具**: 基因组功能研究
- 💡 **产业应用**: 精准农业解决方案

---

**最后更新**: 2026年3月25日  
**版本**: v1.0.0  
**状态**: 准备开始 Rice3k 数据下载
