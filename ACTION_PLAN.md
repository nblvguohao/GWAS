# PlantHGNN 完整行动方案

## 当前状态 (2026-04-09)

### ✅ 已完成
- [x] d_model=256消融实验运行中 (PPI+KEGG Run 4/5)
- [x] RF/XGBoost基线运行中 (RF完成, XGB进行中)
- [x] 全性状5-Fold CV排队中 (GPU 0)
- [x] 论文表格模板创建
- [x] 术语修正 (Heterogeneous→Multi-View)

### 🔄 进行中
- d_model=256消融: PPI+KEGG Run 4/5, PPI+GO+KEGG待开始
- XGBoost基线: 预计今晚完成
- 全性状实验: 等待GPU 0空闲

---

## Phase 1: 实验完成监控 (今晚-明早)

### 1.1 自动监控任务
```bash
# Windows
.\monitor_experiments.bat

# 或详细检查
bash check_server_progress.sh
```

### 1.2 实验完成检查点
- [ ] d256消融完成 (预计今晚8-10点)
- [ ] XGBoost完成 (预计今晚7-8点)
- [ ] 生成JSON结果文件

### 1.3 数据同步
```bash
bash sync_results_from_server.sh
```

**产出文件**:
- `results/gstp007/ablation_d256_gpu1.json`
- `results/gstp007/rf_xgboost_baseline.json`

---

## Phase 2: 论文表格更新 (明早)

### 2.1 自动更新脚本
```bash
python scripts/update_paper_with_results.py
```

### 2.2 手动验证表格
| 表格 | 文件 | 检查项 |
|------|------|--------|
| Table 3 消融 | `table_ablation_study_v2.tex` | 5个配置完整, 无TODO |
| Table 4 基线 | `table_baseline_comparison.tex` | 7个方法完整, 含RF/XGB |
| Table 5 统计 | `table_statistical_tests.tex` | p值正确标注 |

### 2.3 LaTeX编译测试
```bash
cd paper_latex/main
pdflatex main.tex
```

---

## Phase 3: 全性状实验 (明天)

### 3.1 触发条件
GPU 0当前任务完成且空闲

### 3.2 执行命令
```bash
# 服务器端自动启动 (已配置)
bash run_all_traits_5fold_gpu0.sh
```

### 3.3 实验范围
- 6个性状: Grain_Length, Grain_Width, Grain_Weight, Panicle_Length, Plant_Height, Yield_per_plant
- 5-fold CV × 3 seeds = 15 runs per trait
- 预计时间: 12-18小时

### 3.4 产出
- `all_traits_5fold_d256.json`
- 更新Figure 4多性状热图

---

## Phase 4: 可视化与统计 (后天)

### 4.1 生成图表
```bash
python scripts/generate_paper_figures.py
```

**生成图表**:
- Figure 1: 带误差棒的HPO对比图
- Figure 2: 训练曲线
- Figure 3: 消融研究对比
- Figure 4: 多性状热图 (更新)
- Figure 5: 残差分析
- Figure 6: 注意力权重可视化

### 4.2 统计检验更新
```bash
python ablation_server_deploy_20260406/compute_statistical_tests.py
```

---

## Phase 5: 最终整合与检查 (后天)

### 5.1 论文检查清单
```markdown
- [ ] 所有表格无TODO标记
- [ ] 所有图表清晰可见
- [ ] 统计检验p值正确
- [ ] 参考文献格式统一
- [ ] 无编译错误
- [ ] PDF页数符合要求
```

### 5.2 对照评审报告确认
| 评审问题 | 修改状态 | 验证方式 |
|----------|----------|----------|
| 术语修正 | ✅ | 全文搜索"Heterogeneous" |
| 消融实验 | 🔄 | Table 3完整数据 |
| 统计检验 | 🔄 | Table 5更新 |
| RF/XGBoost基线 | 🔄 | Table 4包含 |
| 全性状验证 | ⏳ | Figure 4更新 |
| 参考文献 | ✅ | 25篇已扩展 |

### 5.3 最终提交准备
- [ ] Cover Letter撰写
- [ ] Response to Reviewers准备
- [ ] 作者信息确认
- [ ] 利益冲突声明
- [ ] 数据可用性声明

---

## 执行时间表

| 日期 | 时间段 | 任务 | 负责 |
|------|--------|------|------|
| 4/9 | 今晚 | 监控实验, 等待完成 | 自动 |
| 4/10 | 上午 | 同步结果, 更新表格 | Claude |
| 4/10 | 下午 | 启动全性状实验 | 自动 |
| 4/11 | 全天 | 等待全性状完成 | 自动 |
| 4/12 | 上午 | 同步结果, 生成图表 | Claude |
| 4/12 | 下午 | 最终整合, 生成PDF | Claude |
| 4/13 | - | 作者审阅, 投稿准备 | 用户 |

---

## 紧急联系方案

如果实验失败或结果异常:
1. 检查日志: `tail -100 results/gstp007/*.log`
2. 重启实验: 修改脚本后重新提交
3. 备用方案: 使用已有d_model=64结果补充说明

---

## 当前可执行操作

现在可以执行:
1. ✅ 监控实验进度
2. ✅ 准备论文其他部分 (Introduction, Discussion)
3. ✅ 撰写Cover Letter草稿
4. ⏳ 等待实验完成后: 同步结果并更新表格

是否开始执行监控和准备工作?
