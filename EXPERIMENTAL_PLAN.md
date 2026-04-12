# PlantHGNN 实验补做计划

**日期**: 2026年4月8日  
**目标**: 为论文修改提供真实的实验数据支持

---

## 一、已有真实数据 ✅

### 1.1 主实验 (5-fold CV × 3 seeds = 15 runs)

| 性状 | MultiView_PPI_GO | NetGP | DNNGP | GBLUP |
|------|------------------|-------|-------|-------|
| Grain_Length | **0.8802±0.0148** | 0.8760±0.0162 | 0.8751±0.0155 | 0.8572±0.0158 |
| Grain_Width | 0.7892±0.0196 | **0.7940±0.0157** | 0.7868±0.0186 | 0.7859±0.0203 |
| Grain_Weight | 0.8320±0.0223 | **0.8396±0.0187** | 0.8310±0.0202 | 0.8082±0.0229 |
| Plant_Height | 0.7896±0.0202 | **0.7963±0.0198** | 0.7916±0.0256 | 0.7661±0.0248 |

**文件**: `results/gstp007/final_5fold3seed_results.json`

---

## 二、需要补做的实验 🔄

### 2.1 消融实验 (P0 - 必须)

为论文Table 3提供真实数据，需要运行：

| 配置 | 说明 | 优先级 | 预计时间 |
|------|------|--------|----------|
| **PPI-only** | 单PPI网络，无注意力融合 | P0 | ~2h |
| **KEGG-only** | 单KEGG网络，无注意力融合 | P0 | ~2h |
| **GO-only** | 单GO网络，无注意力融合 | P1 | ~2h |
| **Average Fusion** | PPI+GO，平均权重融合 | P0 | ~2h |
| **Concatenation Fusion** | PPI+GO，拼接后投影 | P1 | ~2h |
| **No Transformer** | 仅GCN编码+回归头 | P0 | ~1.5h |
| **No GCN (MLP)** | 纯MLP基线 | P0 | ~1h |

**总预计时间**: ~12-14小时 ( Grain_Length单性状)

**输出**: 每个配置需要5-fold CV × 3 seeds = 15 runs

### 2.2 小模型对比实验 (P1 - 强烈建议)

评审关注参数/样本比过高(~4,900:1)，需要对比：

| d_model | 参数量估计 | 预期PCC | 运行时间 |
|---------|-----------|---------|----------|
| 32 | ~100K (比例~80:1) | ~0.865 | ~1h |
| 64 | ~400K (比例~315:1) | ~0.872 | ~1h |
| 128 | ~1.6M (比例~1,260:1) | ~0.877 | ~1.5h |
| 256 | ~6.2M (当前, ~4,900:1) | 0.8802 | ~2h |

**总预计时间**: ~6小时 (单性状，5-fold CV)

**目的**: 证明模型复杂度与性能权衡，回应过拟合担忧

### 2.3 统计检验计算 (P0 - 必须)

基于现有5-fold CV数据计算：

- [x] 配对t-test (已有，可以复算)
- [ ] Wilcoxon signed-rank test (需计算)
- [ ] 95% 置信区间 (需计算)
- [ ] Bonferroni校正 (需计算)
- [ ] Cohen's d效应量 (可选)

**实现**: Python脚本，无需GPU

### 2.4 贝叶斯优化对比 (P2 - 建议)

用Optuna实现贝叶斯优化作为HPO对比：

```python
import optuna

def objective(trial):
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    # ... train and return PCC

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**预计时间**: ~20-30小时 (50 trials)

**目的**: 对比手动网格 vs 贝叶斯优化，回应HPO方法质疑

---

## 三、实验脚本需求

### 3.1 需要创建的新脚本

| 脚本名 | 功能 | 优先级 |
|--------|------|--------|
| `run_ablation_study.py` | 运行所有消融配置 | P0 |
| `run_model_scaling.py` | 对比d_model=32/64/128/256 | P1 |
| `compute_statistical_tests.py` | 计算所有统计检验 | P0 |
| `run_optuna_hpo.py` | 贝叶斯优化HPO | P2 |

### 3.2 需要修改的现有脚本

- [ ] `ablation_diverse_views.py` - 完善以支持所有消融配置
- [ ] `benchmark_5fold_cv.py` - 确保支持小模型配置

---

## 四、实验执行计划

### 阶段1: 紧急 (1-2天)
**目标**: 完成P0实验，更新论文Table 2-3

1. **Day 1**: 
   - [ ] 运行PPI-only消融 (5-fold CV × 3 seeds)
   - [ ] 运行KEGG-only消融
   - [ ] 运行Average Fusion消融
   - [ ] 运行No Transformer消融
   
2. **Day 2**:
   - [ ] 运行No GCN (MLP)消融
   - [ ] 计算所有统计检验 (t-test, Wilcoxon, CI)
   - [ ] 更新论文Table 2和Table 3

### 阶段2: 重要 (3-5天)
**目标**: 回应过拟合质疑，提供模型缩放证据

3. **Day 3-4**:
   - [ ] 运行d_model=32对比实验
   - [ ] 运行d_model=64对比实验
   - [ ] 运行d_model=128对比实验
   
4. **Day 5**:
   - [ ] 分析模型缩放结果
   - [ ] 更新Discussion和Limitations部分

### 阶段3: 加分项 (1-2周)
**目标**: 展示现代HPO方法

5. **Week 2**:
   - [ ] 实现Optuna贝叶斯优化
   - [ ] 运行50 trials搜索
   - [ ] 对比手动网格 vs 贝叶斯优化结果

---

## 五、论文数据更新检查清单

### Table 2: 统计显著性检验
- [ ] 使用真实5-fold CV数据计算
- [ ] 确认MultiView vs NetGP的p-value
- [ ] 添加95% CI

### Table 3: 消融实验
- [ ] PPI-only: 真实运行值 (预估~0.876)
- [ ] KEGG-only: 真实运行值 (预估~0.856)
- [ ] Average Fusion: 真实运行值 (预估~0.877)
- [ ] No Transformer: 真实运行值 (预估~0.869)
- [ ] No GCN: 真实运行值 (预估~0.862)

### Figure更新
- [ ] 如果有显著差异，更新网络贡献图

---

## 六、风险评估

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| KEGG-only表现过差 | 评审质疑网络质量 | 诚实报告，讨论KEGG密度问题 |
| PPI-only接近MultiView | 注意力融合价值不明显 | 强调稳定性和跨性状一致性 |
| 小模型(d=32)性能接近 | 大模型过拟合证据 | 这正是我们想展示的权衡 |
| 实验时间过长 | 延误修改 | 优先P0，并行运行 |

---

## 七、执行建议

### 立即可执行 (今天)
```bash
# 1. 启动PPI-only消融实验
python scripts/run_ablation_study.py --config ppi_only --trait Grain_Length

# 2. 同时启动统计检验计算
python scripts/compute_statistical_tests.py --input results/gstp007/final_5fold3seed_results.json
```

### 需要开发 (明天)
- 完善 `run_ablation_study.py` 脚本
- 实现所有消融配置的支持

### 需要较长时间 (本周)
- 小模型对比实验
- 贝叶斯优化实验 (可选)

---

**结论**: 最少需要运行**5个消融配置 × 15 runs = 75个实验**，预计耗时12-15小时。建议优先完成P0实验，确保论文数据真实可靠。
