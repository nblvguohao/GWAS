# PlantMEGNN 论文审稿报告

## Part 1: 审稿意见 (The Review Report)

### Summary

PlantMEGNN presents a multi-view graph neural network framework that integrates genomic data with three biological networks (PPI, GO, KEGG) for rice genomic prediction, achieving modest but consistent improvements over GBLUP (6.2%) and NetGP (3.8%) across three independent populations. The key contribution is the view-differentiated redesign that addresses attention mechanism uniformity by expanding gene coverage from 831 to 1,712 (GO) and 4,053 (KEGG) genes, achieving 13,000× attention variance increase.

### Strengths

**1. Biological Interpretability Through Attention Mechanism**

The paper successfully demonstrates that learned attention weights reveal biologically meaningful patterns—KEGG pathways dominate grain morphology traits while PPI networks are preferred for complex yield traits. This interpretability bridges statistical prediction and biological mechanism, offering practical value to breeders for candidate gene prioritization. The view-differentiated redesign (831→1,712→4,053 genes) addresses a fundamental limitation of uniform network coverage and provides quantitative evidence (13,000× variance increase) that attention mechanisms can learn meaningful view selection when given diverse information sources.

**2. Rigorous Cross-Population Validation**

The evaluation spans three genetically diverse populations (1,495 + 705 + 378 accessions), demonstrating consistent performance gains (+5.5% to +7.8% over GBLUP). This establishes generalizability across indica, japonica, and admixed germplasm, addressing a critical practical concern for breeding applications. The multi-environment analysis on GSTP008 (4 locations) adds further robustness, though with mixed results.

**3. Comprehensive Ablation Study Design**

The ablation study goes beyond standard architectural ablations to isolate the contribution of graph structure versus gene-level feature aggregation (Gene-MLP baseline). The finding that graph structure provides modest gains (+0.006 PCC average) but strong trait-dependent effects (+0.021-0.027 for grain morphology) is nuanced and scientifically valuable. The inclusion of 8 baseline methods (GBLUP, Ridge, DNNGP, Transformer, NetGP, GeneSeqGNN, LightGBM, Gene-MLP) provides fair comparison coverage.

### Weaknesses (Critical)

**1. Marginal Performance Gains Over Strong Baselines**

The primary concern is the modest improvement margin over well-tuned traditional methods. PlantMEGNN achieves 0.786 average PCC on GSTP007, compared to 0.768 for LightGBM (+2.3%) and 0.757 for NetGP (+3.8%). While statistically significant, the practical significance is debatable—LightGBM, a non-deep-learning method requiring no GPU and minimal tuning, achieves 96% of PlantMEGNN's performance. The Gene-MLP ablation (0.753) further suggests that much of the "deep learning advantage" comes from nonlinear modeling of gene-level features rather than the multi-view graph architecture that constitutes the paper's claimed contribution.

**2. Cross-Population Transfer Claims Are Overstated**

Section 4.3 claims "practical implications" for cross-population transfer, but Table 4 shows highly inconsistent results: Grain Length transfers well (PCC=0.325-0.864), but Plant Height and Days to Heading show negative transfer (-0.643 to -0.042). The claim that "models may transfer across populations for traits with major-effect QTLs" is technically accurate but reduces the contribution to "it works when it works." The small sample of transfer directions (only 3 source-target pairs) limits generalizability claims.

**3. Attention Mechanism Predictive Value Remains Limited**

Despite the view-differentiated redesign achieving substantial attention weight differentiation (13,000× variance increase), the predictive advantage of learnable versus uniform attention remains minimal (Table 3: 0.755 vs 0.755 average PCC). The paper acknowledges this indirectly ("learned attention weights provided minimal predictive advantage") but buries this critical limitation in the Interpretability subsection rather than confronting it directly in the Ablation Study. The attention mechanism succeeds at providing interpretability but fails at improving prediction—a distinction that should be front-and-center.

**4. Incomplete Ablation Study for View-Differentiated Design**

The paper introduces the view-differentiated redesign as a major methodological improvement but lacks complete ablation results. Section 3.2 mentions "PPI+GO, PPI+KEGG, and All views configurations are being evaluated" but only reports Grain Length results. Table 3 compares learnable vs uniform attention for the original 831-gene design, not the new differentiated design. Without showing that PPI-only (diverse) < PPI+GO (diverse) < All views (diverse), the claim that "view differentiation enables effective multi-view learning" remains incompletely validated.

### Rating

**Score: 6/10 (Borderline Accept with Major Revisions)**

The paper presents a well-executed study with rigorous cross-population validation and valuable interpretability insights. However, the marginal performance gains over strong baselines (especially LightGBM), the incomplete validation of the view-differentiated design's contribution to prediction accuracy, and the overstated cross-population transfer claims limit the contribution. The attention mechanism's effectiveness for prediction (not just interpretability) needs more compelling evidence.

---

## Part 2: 战略建议 (Strategic Advice)

### 问题根源分析

**Weakness 1: Marginal Gains Over LightGBM**

*Root Cause:* This is not a design flaw but a reality of genomic prediction—many traits have primarily additive genetic architectures where sophisticated deep learning offers limited advantage over well-tuned gradient boosting. The paper does not adequately contextualize this finding within the broader GP literature.

**Weakness 2: Overstated Transfer Claims**

*Root Cause:* The writing in Section 4.3 frames mixed results as "practical implications" when they are better characterized as "exploratory findings." The claim structure oversells limited empirical support.

**Weakness 3: Attention Predictive Value**

*Root Cause:* Methodological. The attention mechanism is lightweight (895 parameters, 0.1% of model) and applied post-GCN, where view-specific features are already compressed to d_model=128. The architecture may not provide sufficient capacity for attention to substantially affect predictions.

**Weakness 4: Incomplete Ablation**

*Root Cause:* Experimental oversight. The view-differentiated ablation (PPI-only, PPI+GO, PPI+KEGG, All views) is mentioned but not fully reported, likely due to the computational cost of re-running all configurations.

### 可救性判断

| Issue | Salvageable? | Required Action |
|-------|--------------|-----------------|
| Marginal gains vs LightGBM | Partially | Add discussion contextualizing this finding; consider it a feature not bug (efficiency vs accuracy tradeoff) |
| Overstated transfer claims | Yes | Downgrade claims to "exploratory"; add caveats about population-specific training |
| Attention predictive value | Partially | Be explicit about interpretability vs prediction distinction; don't oversell |
| Incomplete ablation | Yes | Complete PPI+GO+KEGG ablation and update Table 3 (critical) |

### 行动指南

**Priority 1: Complete View-Differentiated Ablation (Critical)**

Complete the ablation study for the view-differentiated design:
- PPI-only (diverse): 831 genes
- PPI+GO (diverse): 831 + 881 exclusive GO genes
- PPI+KEGG (diverse): 831 + 3,222 exclusive KEGG genes
- All views (diverse): Combined

Update Table 3 to show the progression. If PPI-only < PPI+GO < All views (even modestly), this validates the redesign. If not, reconsider claims about multi-view effectiveness.

**Priority 2: Recontextualize Transfer Results**

In Section 4.3:
- Change "practical implications" to "exploratory analysis"
- Add explicit caveat: "Population-specific training is recommended for production deployment"
- Emphasize that the stronger contribution is within-population accuracy (Table 2), not transfer

**Priority 3: Clarify Attention Mechanism Value Proposition**

In Abstract and Conclusion:
- Current: "learned attention weights reveal trait-specific network preferences" (true but incomplete)
- Revised: "learned attention weights provide biological interpretability, revealing trait-specific network preferences, though their contribution to prediction accuracy is modest compared to uniform weighting"

In Section 3.2:
- Move the "learnable vs uniform performance" comparison to the beginning of the ablation subsection
- Explicitly state: "The attention mechanism's primary value lies in interpretability rather than predictive gain"

**Priority 4: Address LightGBM Comparison**

Add to Discussion:
"The strong performance of LightGBM (0.768 vs PlantMEGNN 0.786) highlights that gradient boosting remains highly competitive for genomic prediction. PlantMEGNN's advantage lies not in dramatic accuracy improvements but in (1) interpretability through attention weights, (2) consistent gains across diverse populations, and (3) biological insights through network integration. For breeding programs prioritizing accuracy alone, LightGBM offers an efficient alternative; for those seeking mechanistic insights, PlantMEGNN provides unique value."

**Priority 5: Clarify Network Coverage Limitation**

In Discussion 4.2:
- Explicitly state the limitation: "Expanding beyond the 831-gene PPI network to genome-wide coverage would capture more genetic variation"
- Add analysis: Compare performance of genes inside vs outside the network (Supplementary)

---

## 具体修改清单

### 必须修改 (Before Submission)

1. **Complete Table 3** with view-differentiated ablation results
2. **Downgrade Section 4.3** from "practical implications" to "exploratory analysis"
3. **Add LightGBM context paragraph** to Discussion
4. **Clarify attention value proposition** in Abstract, Introduction, and Conclusion

### 建议修改 (Strongly Recommended)

5. Add network coverage limitation discussion
6. Add Supplementary analysis comparing in-network vs out-of-network genes
7. Re-run statistical significance tests (Wilcoxon) for all comparisons
8. Add confidence intervals to all PCC reports (not just std)

### 可选增强 (If Time Permits)

9. Test additional traits beyond Grain Length for view-differentiated ablation
10. Generate attention weight visualization figures
11. Add runtime/efficiency comparison with all baselines

---

## 预期影响

Implementing these changes would:
- Address the "incomplete ablation" criticism (Priority 1)
- Reduce overclaiming attack surface (Priority 2, 3, 4)
- Demonstrate intellectual honesty that strengthens rather than weakens the paper
- Raise rating from 6/10 to 7-8/10 (Accept)

The core contribution—view-differentiated multi-view learning with biological interpretability—remains valid and valuable. The revisions ensure the claims match the evidence.
