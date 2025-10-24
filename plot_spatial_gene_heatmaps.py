"""为预设的 Visium 切片生成空间基因表达热图。

脚本会加载多个包含空间转录组数据的 H5AD 文件，并针对可配置的基因列表
绘制表达热图。H5AD 文件路径写死在脚本中，但可以直接在脚本内修改基因列表
以便快速迭代。

使用方式：
    python plot_spatial_gene_heatmaps.py

生成的图片将保存至 ``outputs/spatial_gene_heatmaps`` 目录下，文件层级结构为
``<sample_id>/<gene>.png``。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


# 输入 Visium H5AD 文件的绝对路径。
H5AD_PATHS = {
    "151673": "/home/yangqx/YYY/151673_RGB.h5ad",
    "151674": "/home/yangqx/YYY/151674_RGB.h5ad",
    "151675": "/home/yangqx/YYY/151675_RGB.h5ad",
    "151676": "/home/yangqx/YYY/151676_RGB.h5ad",
}

# 将你想要查看的基因名称填入该列表。
GENES_OF_INTEREST = [
    "IGKC",
    "IGHA1",
    "MBP",
    "GFAP",
    "IGLC2",
    "GFAP",
    "SNAP25",
    "CHN1",
    "S100B",
    "RTN1",
    "IGHG3",
    "IGHG4",
]

# 所有输出图片会保存到该目录。
OUTPUT_DIR = Path("outputs/spatial_gene_heatmaps")


def ensure_genes_exist(adata: "sc.AnnData", genes: Iterable[str]) -> list[str]:
    """过滤并返回在 AnnData 对象中实际存在的基因。"""
    available = set(adata.var_names)
    found = []
    for gene in genes:
        if gene in available:
            found.append(gene)
        else:
            print(f"[警告] 数据中未找到基因 '{gene}'，已跳过。")
    return found


def extract_expression(adata: "sc.AnnData", gene: str) -> np.ndarray:
    """以 NumPy 数组返回指定基因的表达向量。"""
    expr = adata[:, gene].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr).ravel()
    return expr


def plot_gene_heatmap(sample_id: str, adata: "sc.AnnData", gene: str) -> None:
    """绘制并保存指定基因的空间热图。"""
    if "spatial" not in adata.obsm:
        raise KeyError("AnnData 对象缺少 'spatial' 空间坐标。")

    coords = adata.obsm["spatial"]
    if coords.shape[1] < 2:
        raise ValueError("空间坐标至少需要两个维度。")

    expression = extract_expression(adata, gene)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=expression,
        cmap="viridis",
        s=20,
        edgecolors="none",
    )
    ax.set_title(f"{sample_id} - {gene}")
    ax.set_xlabel("空间 X")
    ax.set_ylabel("空间 Y")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="表达量")
    ax.invert_yaxis()  # 与常见的 Visium 坐标方向保持一致。

    sample_dir = OUTPUT_DIR / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f"{gene}.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"已保存热图：{output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_id, path in H5AD_PATHS.items():
        if not os.path.exists(path):
            print(f"[警告] 找不到文件 '{path}'，已跳过样本 {sample_id}。")
            continue

        print(f"正在处理样本 {sample_id}，数据来源：{path} ...")
        adata = sc.read_h5ad(path)
        genes = ensure_genes_exist(adata, GENES_OF_INTEREST)

        if not genes:
            print(f"样本 {sample_id} 未找到可用的基因，跳过绘图。")
            continue

        for gene in genes:
            plot_gene_heatmap(sample_id, adata, gene)


if __name__ == "__main__":
    main()
