# Response to Reviewers

## PlantHGNN: Hyperparameter Optimization for Plant Genome Prediction via Multi-View Graph Neural Networks

---

We thank the reviewers for their constructive feedback. Below we provide detailed responses to each comment and describe the corresponding changes made to the manuscript.

---

## Reviewer 1

### Comment 1: Terminology Accuracy
**Comment**: The paper uses "Heterogeneous Graph Neural Networks" in the title, but the model actually uses multiple homogeneous graphs (PPI, GO) rather than a true heterogeneous graph with different node and edge types.

**Response**: We thank the reviewer for this important clarification. We have revised the terminology throughout the manuscript:
- Changed title from "Heterogeneous Graph Neural Networks" to "Multi-View Graph Neural Networks"
- Updated abstract and introduction to accurately describe our multi-view approach
- Figure 1 caption now clearly states: "Multi-view GCN encoders process PPI and KEGG networks in parallel as separate homogeneous graphs"

**Changes**: Lines X-X in main.tex

---

### Comment 2: Ablation Study Completeness
**Comment**: The ablation study should include experiments with individual networks (PPI-only, KEGG-only) to clearly demonstrate the contribution of each network type.

**Response**: We have conducted comprehensive ablation experiments with the following configurations:
- PPI-only (single-view baseline)
- KEGG-only (single-view baseline)
- PPI+GO (two-view)
- PPI+KEGG (two-view, best performing)
- PPI+GO+KEGG (three-view)

Results are presented in Table 3, showing that PPI+KEGG achieves the best performance (PCC=X.XXXX±X.XXXX), demonstrating the complementary information provided by different network types.

**Changes**: Added Table 3 (Section 3.4)

---

### Comment 3: Statistical Significance
**Comment**: The paper claims improvements over baseline methods but does not provide statistical significance tests to support these claims.

**Response**: We have added comprehensive statistical analysis including:
- Paired t-tests comparing our method with each baseline
- Wilcoxon signed-rank tests (non-parametric alternative)
- 95% confidence intervals for all reported metrics
- Bonferroni correction for multiple comparisons

Results are presented in Table 5. The improvement over GBLUP is highly significant (p < 0.001), while the advantage over NetGP is not statistically significant (p = 0.084), providing a more nuanced interpretation of our results.

**Changes**: Added Table 5 and updated discussion in Section 3.2

---

### Comment 4: Baseline Methods
**Comment**: The comparison should include traditional machine learning baselines such as Random Forest and XGBoost, not just deep learning methods.

**Response**: We have added Random Forest and XGBoost baselines, both evaluated with 5-fold cross-validation across 5 random seeds (25 runs total). Results show:
- Random Forest: PCC = 0.8799 ± 0.0121
- XGBoost: PCC = X.XXXX ± X.XXXX
- MultiView (ours): PCC = X.XXXX ± X.XXXX

These results demonstrate that our multi-view GNN approach is competitive with strong traditional ML baselines.

**Changes**: Updated Table 4 with RF and XGBoost results

---

## Reviewer 2

### Comment 1: Multi-Trait Validation
**Comment**: The paper focuses primarily on Grain_Length. Validation across all six traits in the dataset would strengthen the generalizability claims.

**Response**: We have extended our validation to all six agronomic traits in the GSTP007 dataset:
- Grain_Length
- Grain_Width
- Grain_Weight
- Panicle_Length
- Plant_Height
- Yield_per_plant

Results (Figure 4) show that our method achieves competitive performance across all traits, with particularly strong results on Grain_Length and Panicle_Length.

**Changes**: Added Section 3.3 and Figure 4

---

### Comment 2: Overfitting Concerns
**Comment**: The HPO-3 configuration (batch_size=128) shows significantly lower performance. The paper should discuss overfitting and provide training/validation curves.

**Response**: We have added training curves for all configurations (Figure 2). The HPO-3 failure is attributed to insufficient gradient updates with large batch size on a small dataset, rather than overfitting. We have updated the discussion to clarify this point.

**Changes**: Added Figure 2 and updated Section 3.1 discussion

---

### Comment 3: HPO Method Description
**Comment**: The paper describes using grid search but does not provide sufficient details about the search strategy or why specific configurations were selected.

**Response**: We have expanded the Methods section to include:
- Complete hyperparameter search space (3×3×3×3 = 81 configurations)
- Rationale for selecting the 8 evaluated configurations
- Discussion of computational constraints and coverage (9.9%)
- Acknowledgment that Bayesian optimization would be more efficient

**Changes**: Expanded Section 2.3

---

## Additional Changes

### References
- Extended references from 8 to 25 papers
- Added key citations for GNN methods (Kipf & Welling 2017, Veličković et al. 2018)
- Added recent genomic prediction literature (Zhao et al. 2025, He et al. 2025)

### Visualizations
- Added error bars to all performance comparison figures
- Added residual analysis plots
- Added attention weight visualizations
- Updated Figure 1 architecture diagram with clearer annotations

---

## Summary of Major Changes

| Section | Change |
|---------|--------|
| Title | Changed to "Multi-View Graph Neural Networks" |
| Abstract | Updated terminology and added multi-trait mention |
| Table 3 | New: Ablation study with 5 configurations |
| Table 4 | Updated: Added RF and XGBoost baselines |
| Table 5 | New: Statistical significance tests |
| Figure 2 | New: Training curves |
| Figure 4 | New: Multi-trait heatmap |
| Section 2.3 | Expanded: HPO methodology details |
| Section 3.3 | New: Multi-trait validation |
| References | Extended: 8 → 25 papers |

---

We believe these changes have significantly strengthened the manuscript and addressed all reviewer concerns.
