# PlantHGNN 实验执行指南

**目标**: 为论文修改提供真实实验数据支持  
**预计总时间**: 2-3天 (如果并行运行)

---

## 一、快速开始 (5分钟)

### 1.1 测试脚本是否可用

```bash
# 1. 测试统计检验脚本 (使用已有数据，无需训练)
python scripts/compute_statistical_tests.py \
    --input results/gstp007/final_5fold3seed_results.json \
    --trait Grain_Length

# 期望输出: 统计检验表格，包含t-test和Wilcoxon结果
```

### 1.2 运行单个消融实验 (测试)

```bash
# 运行PPI-only消融 (约20-30分钟)
python scripts/run_ablation_study.py \
    --config ppi_only \
    --trait Grain_Length \
    --seeds 42 \
    --n_folds 2
```

---

## 二、完整实验流程

### 阶段1: 统计检验 (10分钟)

使用已有5-fold CV数据计算统计显著性。

```bash
# 计算所有性状的统计检验
for trait in Grain_Length Grain_Width Grain_Weight Plant_Height; do
    python scripts/compute_statistical_tests.py \
        --input results/gstp007/final_5fold3seed_results.json \
        --trait $trait \
        --output results/gstp007/stat_tests_${trait}.json
done
```

**输出**: `results/gstp007/stat_tests_*.json`  
**用途**: 填充论文Table 2

---

### 阶段2: 消融实验 (12-15小时)

这是最重要的补充实验，需要真实运行。

```bash
# 方法1: 逐个运行 (推荐，便于监控)

# PPI-only (~2小时)
python scripts/run_ablation_study.py \
    --config ppi_only \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5

# KEGG-only (~2小时)
python scripts/run_ablation_study.py \
    --config kegg_only \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5

# Average Fusion (~2小时)
python scripts/run_ablation_study.py \
    --config avg_fusion \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5

# Concatenation Fusion (~2小时)
python scripts/run_ablation_study.py \
    --config concat_fusion \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5

# No Transformer (~2小时)
python scripts/run_ablation_study.py \
    --config no_transformer \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5

# No GCN (~1小时)
python scripts/run_ablation_study.py \
    --config no_gcn \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5
```

或者 **一键运行所有** (需要整天运行):

```bash
# 方法2: 全部运行 (~14小时)
python scripts/run_ablation_study.py \
    --config all \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5 \
    --output results/gstp007/ablation_all_Grain_Length.json
```

**输出**: `results/gstp007/ablation/ablation_*.json`  
**用途**: 填充论文Table 3

---

### 阶段3: RF/XGBoost基线 (6-8小时)

添加传统机器学习基线对比。

```bash
# Random Forest (~3-4小时)
python scripts/run_rf_xgboost_baseline.py \
    --method rf \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5

# XGBoost (~3-4小时)
python scripts/run_rf_xgboost_baseline.py \
    --method xgboost \
    --trait Grain_Length \
    --seeds 42 123 456 \
    --n_folds 5
```

**注意**: 需要安装xgboost (`pip install xgboost`)  
**输出**: `results/gstp007/baseline_rf_xgboost_Grain_Length.json`  
**用途**: 扩展论文基线对比表

---

### 阶段4: 小模型对比 (可选, 6小时)

回应过拟合质疑，对比不同模型规模。

需要修改脚本以支持d_model参数，或使用现有脚本的不同配置。

---

## 三、结果整合

### 3.1 生成论文表格

运行完所有实验后，使用以下脚本整合结果：

```bash
# 生成Table 2 (统计检验)
python scripts/generate_table2_statistical_tests.py \
    --input results/gstp007/stat_tests_*.json \
    --output paper_latex/main/tables/table_statistical_tests.tex

# 生成Table 3 (消融实验)
python scripts/generate_table3_ablation.py \
    --input results/gstp007/ablation/ablation_all_Grain_Length.json \
    --output paper_latex/main/tables/table_ablation.tex
```

### 3.2 验证结果完整性

```bash
# 检查所有必要的实验结果是否存在
python scripts/verify_experiments.py \
    --trait Grain_Length \
    --check-ablation \
    --check-baselines \
    --check-stat-tests
```

---

## 四、时间规划建议

| 阶段 | 实验内容 | 时间 | 并行度 |
|------|----------|------|--------|
| 1 | 统计检验 | 10分钟 | 无需GPU |
| 2a | PPI-only消融 | 2小时 | 单GPU |
| 2b | KEGG-only消融 | 2小时 | 单GPU |
| 2c | Average Fusion | 2小时 | 单GPU |
| 2d | No Transformer | 2小时 | 单GPU |
| 2e | No GCN | 1小时 | 单GPU |
| 3a | Random Forest | 4小时 | CPU并行 |
| 3b | XGBoost | 4小时 | CPU并行 |

**并行策略**:
- 消融实验串行运行 (GPU限制)
- RF和XGBoost可以与其他实验并行 (CPU)
- 统计检验随时可运行

---

## 五、常见问题

### Q1: 实验运行时间过长？

**解决方案**:
- 减少seed数量: `--seeds 42` (单seed)
- 减少fold数量: `--n_folds 2` (快速测试)
- 使用CPU版本 (消融实验脚本支持)

### Q2: GPU内存不足？

**解决方案**:
- 使用CPU运行: 修改脚本中的`DEVICE = torch.device('cpu')`
- 减小batch size (需要在脚本中修改)

### Q3: 如何验证实验正确性？

**检查点**:
- PPI-only的结果应该接近MultiView (0.876左右)
- KEGG-only应该比PPI-only差 (0.85-0.86)
- No GCN应该最差 (0.85-0.86)

### Q4: 如果消融结果与预期不符？

**可能原因**:
- 实现bug
- 超参数不适合该配置
- 训练不充分

**解决方案**:
- 检查脚本输出日志
- 增加epochs数量
- 调整学习率

---

## 六、实验完成检查清单

- [ ] 统计检验脚本成功运行
- [ ] PPI-only消融完成
- [ ] KEGG-only消融完成
- [ ] Average Fusion消融完成
- [ ] Concatenation Fusion消融完成 (可选)
- [ ] No Transformer消融完成
- [ ] No GCN消融完成
- [ ] Random Forest基线完成
- [ ] XGBoost基线完成
- [ ] 所有结果文件已生成
- [ ] 论文Table 2已更新
- [ ] 论文Table 3已更新

---

## 七、联系与支持

如有问题:
1. 检查脚本日志输出
2. 查看`EXPERIMENTAL_PLAN.md`中的详细说明
3. 验证数据文件路径正确

---

**开始实验**: `python scripts/run_ablation_study.py --config ppi_only --trait Grain_Length`
