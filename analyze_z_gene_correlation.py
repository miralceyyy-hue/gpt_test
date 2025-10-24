"""用于分析 RGB 潜在表示与基因表达之间关系的实用脚本。

该脚本会读取 ``test_new_gat_cross_attn_res_RGB.py`` 训练过程中保存的潜在 RGB
表示（``z``），并计算每个 RGB 通道与所有基因表达的相关性。流程会重用训练脚
本中的 ``load_and_prepare`` 函数，以确保 AnnData 对象与训练时保持完全一致
（细胞顺序、基因过滤、归一化方式等不会发生变化）。

使用示例
--------

.. code-block:: bash

    python analyze_z_gene_correlation.py \
        --z-path /home/yangqx/YYY/new_gat_cross_attn_res_slice73-76/epoch_223/z_epoch223.npy \
        --adata-paths /home/yangqx/YYY/151673_RGB.h5ad /home/yangqx/YYY/151674_RGB.h5ad \
        --adata-paths /home/yangqx/YYY/151675_RGB.h5ad /home/yangqx/YYY/151676_RGB.h5ad \
        --output-csv epoch223_rgb_gene_correlation.csv

生成的 CSV 会列出每个基因与各个 RGB 通道之间的皮尔逊相关系数，同时会在终端
打印每个通道相关性绝对值最大的若干基因，便于快速查看主要关联。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from test_new_gat_cross_attn_res_RGB import load_and_prepare


def _asarray(x) -> np.ndarray:
    """将（可能为稀疏的）矩阵转换为 ``float64`` 类型的致密数组。"""

    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _pearson_corr(z: np.ndarray, gene_matrix: np.ndarray) -> np.ndarray:
    """计算 ``z`` 与每一个基因表达列之间的皮尔逊相关系数。"""

    if z.ndim != 1:
        raise ValueError("Expected z to be one-dimensional for a single channel.")

    z_centered = z - z.mean()
    z_denom = np.sqrt(np.sum(z_centered ** 2))
    if z_denom == 0.0:
        return np.zeros(gene_matrix.shape[1], dtype=np.float64)

    gene_centered = gene_matrix - gene_matrix.mean(axis=0, keepdims=True)
    gene_denom = np.sqrt(np.sum(gene_centered ** 2, axis=0))

    # 避免对方差为零的基因进行除法，防止出现除零错误。
    valid = gene_denom > 0
    corr = np.zeros(gene_matrix.shape[1], dtype=np.float64)
    if np.any(valid):
        numerator = z_centered @ gene_centered[:, valid]
        corr_valid = numerator / (z_denom * gene_denom[valid])
        corr[valid] = corr_valid
    return corr


def compute_correlations(
    z_path: Path,
    adata_paths: Iterable[Path],
    output_csv: Path,
    top_k: int,
) -> None:
    """加载数据、计算各通道相关性并写入磁盘。"""

    z = np.load(z_path)
    if z.ndim != 2 or z.shape[1] != 3:
        raise ValueError(
            f"Expected z to have shape (n_cells, 3), got {z.shape}."
        )

    print(f"已从 {z_path} 读取形状为 {z.shape} 的 z 嵌入。")

    out_dict = load_and_prepare(list(map(str, adata_paths)))
    adata = out_dict["adata_all"]
    gene_matrix = _asarray(adata.X)
    if gene_matrix.shape[0] != z.shape[0]:
        raise ValueError(
            "Number of cells in AnnData does not match z embedding: "
            f"{gene_matrix.shape[0]} vs {z.shape[0]}."
        )

    gene_names = np.asarray(adata.var_names)

    correlations = {}
    channel_labels = ["R", "G", "B"]
    for idx, label in enumerate(channel_labels):
        correlations[label] = _pearson_corr(z[:, idx].astype(np.float64), gene_matrix)

    df = pd.DataFrame({"gene": gene_names})
    for label in channel_labels:
        df[f"corr_{label}"] = correlations[label]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"已将完整的相关性表格保存至 {output_csv}。")

    if top_k > 0:
        for label in channel_labels:
            corr_values = correlations[label]
            order = np.argsort(np.abs(corr_values))[::-1][:top_k]
            print("\n通道", label, "的 Top 基因：")
            for rank, gene_idx in enumerate(order, start=1):
                gene = gene_names[gene_idx]
                value = corr_values[gene_idx]
                print(f"  {rank:2d}. {gene}: 相关系数 = {value:.4f}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析学习得到的 RGB 嵌入与基因表达之间的相关性。"
    )
    parser.add_argument(
        "--z-path",
        type=Path,
        required=True,
        help="保存的 z 数组路径（形状需为 [n_cells, 3]）。",
    )
    parser.add_argument(
        "--adata-paths",
        type=Path,
        nargs="+",
        required=True,
        help="训练时使用的一个或多个 .h5ad 文件（顺序必须与训练一致）。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("z_gene_correlations.csv"),
        help="相关性表格输出的 CSV 文件路径。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="终端打印的每个通道 Top 基因数量。",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    compute_correlations(args.z_path, args.adata_paths, args.output_csv, args.top_k)


if __name__ == "__main__":
    main()
