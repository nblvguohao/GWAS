"""
Biological Validation Suite for PlantHGNN

Performs biological validation without wet experiments:
1. GO functional enrichment of top-weighted genes (hypergeometric test)
2. Network topological feature analysis (degree centrality)
3. Literature QTL gene enrichment (extended background)
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Paths
GSTP007_DIR = Path("data/processed/gstp007")
GRAPH_DIR = GSTP007_DIR / "graph"
RAW_ANN_DIR = Path("data/raw/annotations")
OUT_DIR = Path("results/biological_validation")
TABLE_DIR = Path("paper_latex/tables")
FIG_DIR = Path("paper_latex/figures")

for d in [OUT_DIR, TABLE_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Extended QTL genes (RAP-DB format)
LITERATURE_QTL_GENES = {
    "Plant_Height": {
        "genes": ["Os01g0883800","Os01g0192900","Os01g0101100","Os05g0407500","Os01g0718300",
                  "Os07g0592000","Os01g59660","Os06g0675000","Os03g0742900","Os02g0672200","Os04g0539300"],
        "source": "RAP-DB, Q-TARO"
    },
    "Grain_Length": {
        "genes": ["Os03g0407400","Os03g0646900","Os05g0187500","Os02g0244100","Os08g0531600",
                  "Os03g0215400","Os01g1011000","Os06g0127800","Os07g0113500","Os04g0350700",
                  "Os02g0771000","Os01g0162900","Os06g0481600","Os05g0158500"],
        "source": "RAP-DB, Q-TARO"
    },
    "Grain_Width": {
        "genes": ["Os05g0187500","Os02g0244100","Os05g0158500","Os08g0531600","Os07g0582000",
                  "Os02g0747900","Os04g0618000","Os03g0842300","Os01g0957500","Os06g0206500","Os09g0485300"],
        "source": "RAP-DB, Q-TARO"
    },
    "Days_to_Heading": {
        "genes": ["Os06g0275000","Os06g0157700","Os10g0463400","Os07g0261200","Os03g0119300",
                  "Os08g0174500","Os06g0632000","Os04g0599300","Os02g0618200","Os05g0174900",
                  "Os01g0212500","Os09g0429300"],
        "source": "RAP-DB, Q-TARO"
    },
    "Panicle_Length": {
        "genes": ["Os09g0441900","Os11g0246500","Os02g1595000","Os07g0669500","Os01g0740000",
                  "Os08g0509100","Os03g0568400","Os04g0432600","Os06g0115400","Os05g0512100","Os01g0907900"],
        "source": "RAP-DB, Q-TARO"
    },
    "Grain_Weight": {
        "genes": ["Os06g0650300","Os05g0187500","Os02g0244100","Os04g0645100","Os07g0525700",
                  "Os01g0614300","Os03g0410000","Os08g0509600","Os09g0261000","Os11g0227800","Os05g0551000"],
        "source": "RAP-DB, Q-TARO"
    },
    "Yield_per_plant": {
        "genes": ["Os09g0441900","Os08g0509100","Os07g0261200","Os01g1011000","Os07g0582000",
                  "Os04g5301000","Os09g3261000","Os06g0650300","Os05g0187500","Os03g0407400","Os02g0244100"],
        "source": "RAP-DB, Q-TARO"
    }
}

DOMINANT_VIEW = {
    "Plant_Height": "go",
    "Grain_Length": "kegg",
    "Grain_Width": "kegg",
    "Days_to_Heading": "kegg",
    "Panicle_Length": "mixed",
    "Grain_Weight": "kegg",
    "Yield_per_plant": "ppi",
}


def load_gene_list():
    with open(GRAPH_DIR / "gene_list_v2.txt") as f:
        return [line.strip() for line in f if line.strip()]


def load_gene_go_map():
    with open(RAW_ANN_DIR / "gene_go_map.json") as f:
        return json.load(f)


def load_networks(genes):
    ppi = sp.load_npz(GRAPH_DIR / "ppi_adj_v2.npz").tocsr()
    go = sp.load_npz(GRAPH_DIR / "go_adj.npz").tocsr()
    kegg = sp.load_npz(GRAPH_DIR / "kegg_adj.npz").tocsr()
    return {"ppi": ppi, "go": go, "kegg": kegg}


def snp_to_gene_df():
    path = Path("data/processed/rice469/snp_to_gene.csv")
    if path.exists():
        import pandas as pd
        return pd.read_csv(path)
    return None


def compute_degree_centrality(adj):
    return np.asarray(adj.sum(axis=1)).flatten()


def get_top_genes(network_name, adj, genes, top_pct=0.20, min_deg=None):
    deg = compute_degree_centrality(adj)
    if min_deg is not None:
        valid_mask = deg > 0
    else:
        valid_mask = np.ones(len(genes), dtype=bool)
    n_top = max(1, int(valid_mask.sum() * top_pct))
    valid_indices = np.where(valid_mask)[0]
    top_local = valid_indices[np.argsort(-deg[valid_indices])[:n_top]]
    return [genes[i] for i in top_local], deg


def hypergeometric_test(top_genes, candidate_set, background_genes):
    top = set(top_genes) & background_genes
    cand = candidate_set & background_genes
    overlap = top & cand
    n_top = len(top)
    n_cand = len(cand)
    N = len(background_genes)
    k = len(overlap)
    expected = (n_cand / N) * n_top if N > 0 else 0.0
    ratio = k / expected if expected > 0 else 0.0
    pval = hypergeom.sf(k - 1, N, n_cand, n_top) if N > 0 and n_top > 0 else 1.0
    return {
        "overlap_count": k,
        "top_count": n_top,
        "candidate_count": n_cand,
        "background_count": N,
        "expected_overlap": round(expected, 3),
        "enrichment_ratio": round(ratio, 3),
        "p_value": round(pval, 6),
        "overlap_genes": sorted(list(overlap))
    }


def go_enrichment(top_genes, gene_go_map, background_genes=None, top_n=10):
    if background_genes is None:
        background_genes = list(gene_go_map.keys())
    bg_set = set(background_genes)
    top_set = set(top_genes) & bg_set
    bg_counter = Counter()
    top_counter = Counter()
    for g, terms in gene_go_map.items():
        if g not in bg_set:
            continue
        for t in terms:
            bg_counter[t] += 1
    for g in top_set:
        for t in gene_go_map.get(g, []):
            top_counter[t] += 1

    results = []
    N = len(bg_set)
    n_top = len(top_set)
    for go_term, k in top_counter.items():
        if k < 2:
            continue
        n_go = bg_counter[go_term]
        if n_go < 3:
            continue
        pval = hypergeom.sf(k - 1, N, n_go, n_top)
        results.append({
            "go_term": go_term,
            "overlap_count": k,
            "go_term_count": n_go,
            "p_value": pval,
            "enrichment_ratio": (k / n_top) / (n_go / N) if n_go > 0 else 0.0
        })

    m = len(results)
    for r in results:
        r["p_value_bonferroni"] = min(r["p_value"] * m, 1.0)

    results.sort(key=lambda x: x["p_value"])
    return [r for r in results[:top_n] if r["p_value"] < 0.05]


def topology_analysis(adj, top_genes, all_genes):
    gene2idx = {g: i for i, g in enumerate(all_genes)}
    top_indices = [gene2idx[g] for g in top_genes if g in gene2idx]
    bg_indices = [i for i, g in enumerate(all_genes) if g not in set(top_genes)]
    deg = compute_degree_centrality(adj)
    top_deg = deg[top_indices]
    bg_deg = deg[bg_indices]
    if len(top_deg) > 0 and len(bg_deg) > 0:
        stat, pval = mannwhitneyu(top_deg, bg_deg, alternative="greater")
    else:
        stat, pval = 0.0, 1.0
    return {
        "top_mean_degree": float(np.mean(top_deg)),
        "bg_mean_degree": float(np.mean(bg_deg)),
        "top_median_degree": float(np.median(top_deg)),
        "bg_median_degree": float(np.median(bg_deg)),
        "mann_whitney_U": float(stat),
        "p_value": round(pval, 6),
        "n_top": len(top_indices),
        "n_bg": len(bg_indices)
    }


def plot_topology_comparison(summary):
    traits = list(summary.keys())
    top_means = [summary[t]["top_mean_degree"] for t in traits]
    bg_means = [summary[t]["bg_mean_degree"] for t in traits]
    x = np.arange(len(traits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, top_means, width, label="Top 20% genes", color="steelblue")
    ax.bar(x + width/2, bg_means, width, label="Background genes", color="lightgray")
    ax.set_ylabel("Mean degree centrality")
    ax.set_title("Network degree centrality: top-weighted vs background genes")
    ax.set_xticks(x)
    ax.set_xticklabels(traits, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, t in enumerate(traits):
        p = summary[t]["p_value"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.annotate(sig, xy=(i, max(top_means[i], bg_means[i]) * 1.02),
                    ha="center", fontsize=10, color="red")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_s_topological_features.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_s_topological_features.png", dpi=300)
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_s_topological_features.pdf'}")


def plot_go_enrichment_heatmap(go_results):
    all_terms = set()
    for trait, terms in go_results.items():
        all_terms.update(t["go_term"] for t in terms)
    all_terms = sorted(all_terms)
    if not all_terms:
        print("No significant GO terms to plot.")
        return
    traits = list(go_results.keys())
    mat = np.full((len(all_terms), len(traits)), np.nan)
    for j, trait in enumerate(traits):
        term_p = {t["go_term"]: -np.log10(t["p_value"]) for t in go_results[trait]}
        for i, term in enumerate(all_terms):
            mat[i, j] = term_p.get(term, 0.0)
    fig, ax = plt.subplots(figsize=(10, max(6, len(all_terms) * 0.4)))
    sns.heatmap(mat, xticklabels=traits, yticklabels=all_terms, cmap="YlOrRd",
                annot=True, fmt=".1f", linewidths=0.5, ax=ax, cbar_kws={"label": "-log10(p)"})
    ax.set_title("GO enrichment across traits (-log10 p-value)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_s_go_enrichment_heatmap.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_s_go_enrichment_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig_s_go_enrichment_heatmap.pdf'}")


def write_go_table(go_results):
    path = TABLE_DIR / "table_s_go_enrichment.tex"
    with open(path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{GO functional enrichment of top 20\% genes by degree centrality per trait. Only terms with $p<0.05$ are shown.}" + "\n")
        f.write(r"\label{tab:go_enrichment}" + "\n")
        f.write(r"\begin{tabular}{llccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Trait} & \textbf{GO Term} & \textbf{Overlap} & \textbf{Total} & \textbf{$p$-value} \\" + "\n")
        f.write(r"\midrule" + "\n")
        for trait, terms in go_results.items():
            if not terms:
                f.write(f"{trait} & --- & --- & --- & --- \\\n")
                continue
            for i, t in enumerate(terms):
                trait_name = trait if i == 0 else ""
                sig = "***" if t["p_value"] < 0.001 else ("**" if t["p_value"] < 0.01 else "*")
                f.write(f"{trait_name} & {t['go_term']} & {t['overlap_count']} & {t['go_term_count']} & {t['p_value']:.4f}{sig} \\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Saved GO table: {path}")


def write_qtl_table(qtl_results):
    path = TABLE_DIR / "table_s_qtl_enrichment_v2.tex"
    with open(path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{QTL gene enrichment (extended background using SNP-mapped genes, $N$=4,166). Candidate QTL genes compiled from RAP-DB and literature.}" + "\n")
        f.write(r"\label{tab:qtl_enrichment_v2}" + "\n")
        f.write(r"\begin{tabular}{lcccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Trait} & \textbf{Network} & \textbf{Top N} & \textbf{QTL N} & \textbf{Overlap} & \textbf{Ratio} & \textbf{$p$-value} \\" + "\n")
        f.write(r"\midrule" + "\n")
        for trait, data in qtl_results.items():
            ext = data.get("extended", {})
            if "p_value" not in ext:
                f.write(f"{trait} & {data['network']} & --- & --- & --- & --- & --- \\\n")
                continue
            p = ext["p_value"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            ratio = f"{ext['enrichment_ratio']:.2f}" if ext.get("enrichment_ratio", 0) > 0 else "---"
            f.write(f"{trait} & {data['network']} & {ext['top_count']} & {ext['candidate_count']} & {ext['overlap_count']} & {ratio} & {p:.4f}{sig} \\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Saved QTL table: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-pct", type=float, default=0.20)
    parser.add_argument("--go-top-n", type=int, default=10)
    args = parser.parse_args()

    print("="*60)
    print("PlantHGNN Biological Validation Suite")
    print("="*60)

    genes = load_gene_list()
    gene_go_map = load_gene_go_map()
    networks = load_networks(genes)
    print(f"Loaded {len(genes)} genes, {len(gene_go_map)} with GO annotation")

    snp_df = snp_to_gene_df()
    snp_genes = set(snp_df["gene_id"].unique()) if snp_df is not None else set()
    print(f"SNP-mapped background genes: {len(snp_genes)}")

    go_results = {}
    qtl_results = {}
    topology_summary = {}

    for trait, view in DOMINANT_VIEW.items():
        print(f"\n--- {trait} (dominant view: {view}) ---")
        if view == "mixed":
            adj = networks["kegg"]
            net_name = "KEGG"
        else:
            adj = networks[view]
            net_name = view.upper()

        top_genes, _ = get_top_genes(net_name.lower(), adj, genes, args.top_pct)
        bg_genes = set(genes)
        print(f"  Top {len(top_genes)} genes from {net_name}")

        qtl_candidates = set(LITERATURE_QTL_GENES[trait]["genes"])
        qtl_std = hypergeometric_test(top_genes, qtl_candidates, bg_genes)
        qtl_ext = hypergeometric_test(top_genes, qtl_candidates, snp_genes) if snp_genes else {}
        qtl_results[trait] = {
            "network": net_name,
            "standard": qtl_std,
            "extended": qtl_ext,
            "qtl_source": LITERATURE_QTL_GENES[trait]["source"]
        }
        print(f"  QTL (std):  {qtl_std['overlap_count']}/{qtl_std['candidate_count']} (p={qtl_std['p_value']:.4f})")
        if qtl_ext:
            print(f"  QTL (ext):  {qtl_ext['overlap_count']}/{qtl_ext['candidate_count']} (p={qtl_ext['p_value']:.4f})")

        go_terms = go_enrichment(top_genes, gene_go_map, list(bg_genes & set(gene_go_map.keys())), args.go_top_n)
        go_results[trait] = go_terms
        print(f"  Significant GO terms: {len(go_terms)}")

        topo = topology_analysis(adj, top_genes, genes)
        topology_summary[trait] = topo
        print(f"  Top vs BG degree: {topo['top_mean_degree']:.1f} vs {topo['bg_mean_degree']:.1f} (p={topo['p_value']:.4f})")

    with open(OUT_DIR / "go_enrichment.json", "w") as f:
        json.dump(go_results, f, indent=2)
    with open(OUT_DIR / "qtl_enrichment_v2.json", "w") as f:
        json.dump(qtl_results, f, indent=2)
    with open(OUT_DIR / "topology_analysis.json", "w") as f:
        json.dump(topology_summary, f, indent=2)

    plot_topology_comparison(topology_summary)
    plot_go_enrichment_heatmap(go_results)
    write_go_table(go_results)
    write_qtl_table(qtl_results)

    print("\n" + "="*60)
    print(f"Done. Outputs: {OUT_DIR}, {TABLE_DIR}, {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
