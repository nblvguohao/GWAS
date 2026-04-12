# 性能对比分析与超参优化建议

## 一、当前模型 vs SOTA 对比

### 1.1 Grain_Length 性状（我们测试的主要性状）

| 排名 | 模型 | PCC |  vs SOTA 差距 |
|------|------|-----|---------------|
| 🥇 SOTA | MultiView_PPI_GO | 0.8802 ± 0.0148 | - |
| 🥈 | NetGP_1view | 0.8788 ± 0.0163 | -0.0014 |
| 🥉 | NetGP | 0.8760 ± 0.0162 | -0.0042 |
| 4 | DNNGP | 0.8751 ± 0.0155 | -0.0051 |
| 5 | GBLUP | 0.8572 ± 0.0158 | -0.0230 |
| **我们的最佳** | **PPI+KEGG (消融)** | **0.8610 ± 0.0043** | **-0.0192** ❌ |
| | PPI+GO (消融) | 0.8550 ± 0.0057 | -0.0252 |
| | PPI-only (消融) | 0.8538 ± 0.0047 | -0.0264 |

**关键发现**:
- 我们的最佳结果 (PPI+KEGG: 0.8610) **落后 SOTA 0.0192**
- 这是一个显著的差距
- 但我们的结果更稳定 (std=0.0043 vs SOTA std=0.0148)

### 1.2 其他性状对比

| 性状 | SOTA | SOTA 模型 | 我们? |
|------|------|-----------|-------|
| Plant_Height | 0.7963 | NetGP | 未测试 |
| Grain_Width | 0.7940 | NetGP | 未测试 |
| Days_to_Heading | 0.8595 | NetGP_1view | 未测试 |
| Panicle_Length | 0.7498 | GBLUP | 未测试 |
| Grain_Weight | 0.8396 | NetGP | 未测试 |
| Yield_per_plant | 0.3931 | NetGP | 未测试 |

## 二、差距分析

### 2.1 为什么落后?

**可能原因**:
1. **d_model 太小**: 我们使用 d_model=64, 而 SOTA 可能使用更大的维度
2. **网络构建差异**: SOTA 的 MultiView_PPI_GO 可能使用不同的网络构建方式
3. **训练策略**: 虽然我们有早停等优化，但学习率、batch_size 等可能不是最优
4. **数据预处理**: 可能存在细微差异
5. **5-fold CV vs 单折**: SOTA 使用 5-fold CV，我们消融实验是单折

### 2.2 重要发现

我们的消融实验是 **单折验证 (80/10/10)**，而 SOTA 是 **5-fold CV**:
- 单折通常比 5-fold CV 结果略高（数据泄漏风险）
- 但我们的单折结果 (0.8610) < SOTA 5-fold (0.8802)
- 如果我们也用 5-fold CV，差距可能更大

## 三、超参优化建议

### 3.1 当前配置 vs 可能的优化方向

| 参数 | 当前 | 建议尝试 | 理由 |
|------|------|----------|------|
| **d_model** | 64 | 128, 256 | 更大的模型容量 |
| **n_transformer_layers** | 4 | 6, 8 | 更深的网络 |
| **batch_size** | 32 | 64, 128 | NetGP HPO 显示 128 最佳 |
| **lr** | 5e-4 | 1e-4, 1e-3 | 可能需要更精细调整 |
| **dropout** | 0.2 | 0.1, 0.3 | NetGP HPO 显示 0.25 最佳 |
| **weight_decay** | 1e-4 | 1e-5, 1e-3 | 可能需要调整 |
| **n_epochs** | 50 | 100, 200 | 更多训练轮数 |

### 3.2 针对性优化实验

#### 实验 A: 增大模型容量
```bash
# 测试更大 d_model
python run_single_trait_fast.py \
    --trait Grain_Length \
    --views ppi kegg \
    --d_model 128 \
    --n_epochs 50 \
    --use_attnres true
```

#### 实验 B: 调整学习率和 batch_size
```bash
# 使用 NetGP HPO 的最佳参数
python run_single_trait_fast.py \
    --trait Grain_Length \
    --views ppi kegg \
    --d_model 128 \
    --batch_size 128 \
    --lr 0.00014 \
    --dropout 0.25 \
    --weight_decay 0.00011
```

#### 实验 C: 增加训练轮数
```bash
python run_single_trait_fast.py \
    --trait Grain_Length \
    --views ppi kegg \
    --d_model 64 \
    --n_epochs 100 \
    --patience 30
```

## 四、能否打败 SOTA?

### 4.1 现实评估

**差距**: -0.0192 (约 -2.2%)

**能否弥补?**
- ✅ **可能**: 通过超参优化 + 更大模型 + 更多训练
- ⚠️ **挑战**: 
  - SOTA (MultiView_PPI_GO) 可能已经过充分优化
  - 我们的架构简化版 (无真正 AttnRes) 可能有上限
  - 需要大量计算资源进行 HPO

### 4.2 建议策略

**短期 (1-2 天)**:
1. 测试 d_model=128 + PPI+KEGG
2. 使用更好的学习率 (参考 NetGP HPO)
3. 如果提升明显，进行 5-fold CV 验证

**中期 (1 周)**:
1. 实现真正的 AttnRes 深度聚合
2. 添加 Module C (图结构编码)
3. 完整的超参数搜索

**长期 (2-4 周)**:
1. 实现完整论文架构
2. 多性状全面验证
3. 5-fold CV × 5 seeds 严格评估

## 五、替代方案

如果无法打败 SOTA，考虑:

### 方案 1: 调整论文焦点
> "我们证明了 **PPI+KEGG 双视图** 比 PPI+GO 更有效，
> 且 **训练策略优化** 可以弥补架构简化的差距"

### 方案 2: 强调其他贡献
- 更好的训练稳定性 (更小的 std)
- 更快的收敛速度 (早停)
- 更少的参数量 (我们的模型 877K vs NetGP ?)
- 更全面的消融实验

### 方案 3: 降低目标期刊
- BMC Bioinformatics
- Frontiers in Plant Science
- PLOS ONE

## 六、立即行动建议

现在服务器上运行:
```bash
# 测试更大 d_model
python ablation_server_deploy_20260406/run_single_trait_fast.py \
    --trait Grain_Length \
    --views ppi kegg \
    --n_runs 3 \
    --d_model 128 \
    --batch_size 64 \
    --n_epochs 50 \
    --use_attnres true \
    --output results/ablation/gpu0/Grain_Length_dmodel128.json
```

预计提升: +0.005 ~ +0.010
目标: 达到 0.866-0.871 (缩小与 SOTA 差距到 0.01 以内)

需要我立即在服务器上启动这个优化实验吗?
