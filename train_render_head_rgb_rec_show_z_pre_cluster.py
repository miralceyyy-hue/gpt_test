import argparse
import csv
import os
import random
from typing import List, Optional

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

from test_new_gat_cross_attn_res_RGB import (
    SEED,
    _ridge_fit,
    coupling_dir_loss,
    coupling_ratio_loss,
    kl_rgb_loss,
    load_and_prepare,
    save_epoch_z_and_rgb_global_to_img_range,
    compute_rgb_range_from_patches,
)


DEFAULT_ADATA_PATHS = [
    "/home/yangqx/YYY/151673_RGB.h5ad",
    "/home/yangqx/YYY/151674_RGB.h5ad",
    "/home/yangqx/YYY/151675_RGB.h5ad",
    "/home/yangqx/YYY/151676_RGB.h5ad",
]

DEFAULT_H_FUSE_PATH = (
    "/home/yangqx/YYY/new_gat_cross_attn_res_slice73-76_loss_cpl/epoch_165/h_epoch165.npy"
)

DEFAULT_CLUSTER_LABELS_PATH = (
    "/home/yangqx/YYY/new_gat_cross_attn_res_slice73-76_loss_cpl/epoch_165/labels_k7_epoch165.npy"
)

DEFAULT_OUTPUT_DIR = f"render_head_rgb_output_rec_gene_show_z_pre_cluster_loss{3}"

def _set_global_seed(seed: int) -> None:
    """为 Python、NumPy 和 PyTorch 设置随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _prepare_rgb_range(paths: List[str]):
    """利用提供的 h5ad 文件路径计算全局 RGB 范围。"""

    adatas = [sc.read(p) for p in paths]
    rgb_min, rgb_max = compute_rgb_range_from_patches(
        adatas, patch_size=15, p_lo=1.0, p_hi=99.0
    )
    return rgb_min, rgb_max


class GeneReconstructor(nn.Module):
    """简单的解码器，用于从 z 重构基因表达。"""

    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(z_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, output_dim),
            nn.Linear(z_dim, output_dim),

        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class RenderHeadR(nn.Module):
    def __init__(self, h_dim: int = 32, z_dim: int = 3, hidden: int = 64, gaussian: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, 10),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 替换 ReLU 为 LeakyReLU,
            nn.Linear(10, z_dim)
        )

    def forward(self, h: torch.Tensor):
        z = self.net(h)  # (N, out_dim)
        z = torch.sigmoid(z)  # 将 z 控制到 0-1 之间
        return {"z": z}

class ZClassifier(nn.Module):
    """简单的线性分类头，用于对 z 预测聚类标签。"""

    def __init__(self, z_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(z_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

# def orthogonality_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """计算 z 通道之间的正交约束损失。"""
#
#     # 计算两个维度的点积，检查它们是否正交
#     dot_12 = torch.sum(z[:, 0] * z[:, 1])  # 计算第1列和第2列的点积
#     dot_13 = torch.sum(z[:, 0] * z[:, 2])  # 计算第1列和第3列的点积
#     dot_23 = torch.sum(z[:, 1] * z[:, 2])  # 计算第2列和第3列的点积
#
#     print(f"Dot product between feature 1 and feature 2: {dot_12.item()}")
#     print(f"Dot product between feature 1 and feature 3: {dot_13.item()}")
#     print(f"Dot product between feature 2 and feature 3: {dot_23.item()}")
#
#     # 判断是否正交
#     if torch.allclose(dot_12, torch.tensor(0.0)) and torch.allclose(dot_13, torch.tensor(0.0)) and torch.allclose(
#             dot_23, torch.tensor(0.0)):
#         print("The features are orthogonal.")
#     else:
#         print("The features are not orthogonal.")
#
#
#     if z.ndim != 2:
#         raise ValueError("正交损失仅支持二维张量：形状应为 (N, C)")
#
#     z_centered = z - z.mean(dim=0, keepdim=True)
#     cov = torch.matmul(z_centered.T, z_centered) / (z.size(0) + eps)
#     off_diag = cov - torch.diag(torch.diag(cov))
#
#     print("Before centering:", z.mean(dim=0))
#     print("After centering:", z_centered.mean(dim=0))
#
#     return off_diag.pow(2).sum()
def orthogonality_loss(z: torch.Tensor) -> torch.Tensor:
    """计算 z 通道之间的正交约束损失。"""
    # 计算Gram矩阵
    gram_matrix = torch.matmul(z, z.T)

    # 获取非对角线的元素
    eye = torch.eye(gram_matrix.size(0), device=z.device)
    non_diag_mask = 1 - eye  # 生成非对角线的掩码

    # 计算非对角线元素的损失
    non_diag_loss = (gram_matrix * non_diag_mask) ** 2

    non_diag_loss = torch.mean(non_diag_loss)  # 对所有非对角线元素求平均

    return non_diag_loss


def _plot_clusters_points(
    coords: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    palette: np.ndarray,
    point_size: float = 6.0,
    invert_y: bool = False,
    title: Optional[str] = None,
) -> None:
    """将聚类结果绘制为散点图并保存，主要用于空间/UMAP 可视化。"""

    # 为避免绘制时的覆盖伪影，使用和坐标相关的排序方式
    order = np.argsort(coords[:, 0] + coords[:, 1])
    coords_sorted = coords[order]
    labels_sorted = labels[order]
    colors_sorted = palette[labels_sorted]

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(
        coords_sorted[:, 0],
        coords_sorted[:, 1],
        c=colors_sorted,
        s=point_size,
        marker="s",
        linewidth=0,
    )
    plt.gca().set_aspect("equal", adjustable="box")
    if invert_y:
        plt.gca().invert_yaxis()
    if title:
        plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def _build_palette(num_colors: int) -> np.ndarray:
    """根据聚类簇数构造调色板，保证颜色数量充足。"""

    cmap = plt.get_cmap("tab10", num_colors)
    colors = cmap(np.arange(num_colors))[:, :3]
    return colors.astype(np.float32)


def _analyze_z_space(
    z_tensor: torch.Tensor,
    coords_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    out_dir: str,
    epoch: int,
    cluster_k: int,
    random_state: int,
    region_tensor: Optional[torch.Tensor] = None,
) -> None:
    """对当前轮次的 z 空间执行聚类、UMAP，并保存多种可视化结果。"""

    os.makedirs(out_dir, exist_ok=True)
    epoch_dir = os.path.join(out_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # ====== 数据准备：将张量移动至 CPU 并转为 numpy ======
    z_np = z_tensor.detach().cpu().numpy()
    coords_np = coords_tensor.detach().cpu().numpy()
    slice_np = slice_tensor.detach().cpu().numpy()
    region_np = (
        region_tensor.detach().cpu().numpy() if region_tensor is not None else None
    )

    # ====== UMAP 降维（保持随机性可复现） ======
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    umap_coords = reducer.fit_transform(z_np)

    # ====== 使用 R 语言的 Mclust 模型进行聚类（与 h_fuse 分析保持一致） ======

    robjects.r.library("mclust")

    robjects.r["set.seed"](random_state)
    rmclust = robjects.r["Mclust"]

    r_data = numpy2ri.py2rpy(z_np)

    res = rmclust(r_data, cluster_k, "EEE")
    labels = np.array(res[-2]).astype(np.int64) - 1  # R 返回 1 开始的标签，这里转为 0 开始

    palette = _build_palette(int(labels.max() + 1))

    # ====== 保存数组与聚类标签 ======
    np.save(os.path.join(epoch_dir, "z.npy"), z_np)
    np.save(os.path.join(epoch_dir, f"labels_k{cluster_k}.npy"), labels)
    np.save(os.path.join(epoch_dir, "umap.npy"), umap_coords)

    # ====== 绘制 UMAP 聚类结果 ======
    umap_fig_path = os.path.join(epoch_dir, f"umap_k{cluster_k}.png")
    _plot_clusters_points(
        umap_coords,
        labels,
        umap_fig_path,
        palette=palette,
        point_size=10,
        invert_y=False,
        title=f"UMAP | k={cluster_k} | epoch {epoch}",
    )

    # ====== 按切片分别绘制空间聚类结果，避免多个切片叠加 ======
    slice_ids = slice_np.astype(np.int64)
    unique_slices = np.unique(slice_ids)
    for slice_id in unique_slices:
        slice_mask = slice_ids == slice_id
        coords_slice = coords_np[slice_mask]
        labels_slice = labels[slice_mask]
        spatial_fig_path = os.path.join(
            epoch_dir, f"spatial_slice{int(slice_id)}_k{cluster_k}.png"
        )
        _plot_clusters_points(
            coords_slice,
            labels_slice,
            spatial_fig_path,
            palette=palette,
            point_size=15,
            invert_y=True,
            title=f"Spatial | Slice {int(slice_id)} | k={cluster_k} | epoch {epoch}",
        )

    # ====== 如有真实标签，则额外计算 ARI 指标并绘图 ======
    if region_np is not None:
        ari = adjusted_rand_score(region_np, labels)
        print(
            f"[z 聚类] epoch {epoch} | k={cluster_k} | 与真实 Region 的 ARI={ari:.4f}"
        )
        region_palette = _build_palette(int(region_np.max() + 1))
        true_fig_path = os.path.join(epoch_dir, "spatial_region.png")
        _plot_clusters_points(
            coords_np,
            region_np.astype(np.int64),
            true_fig_path,
            palette=region_palette,
            point_size=8,
            invert_y=True,
            title=f"Spatial Region | epoch {epoch}",
        )

    # ====== 按切片绘制，便于观察跨切片表现 ======
    slice_palette = _build_palette(int(slice_np.max() + 1))
    slice_fig_path = os.path.join(epoch_dir, "spatial_slice.png")
    _plot_clusters_points(
        coords_np,
        slice_np.astype(np.int64),
        slice_fig_path,
        palette=slice_palette,
        point_size=8,
        invert_y=True,
        title=f"Spatial Slice | epoch {epoch}",
    )

    print(f"[z 聚类] 已保存第 {epoch} 轮的 UMAP 与空间聚类结果至：{epoch_dir}")


def train_render_head(
    adata_paths: List[str],
    h_fuse_path: str,
    output_dir: str,
    cluster_labels_path: str = DEFAULT_CLUSTER_LABELS_PATH,
    epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_zkl: float = 0.01,
    lambda_cpl_dir: float = 2.0,
    lambda_cpl_ratio: float = 0.5,
    ridge_l2: float = 1e-3,
    hidden_dim: int = 32,
    recon_hidden: int = 128,
    lambda_recon: float = 1.0,
    lambda_ortho: float = 0.1,
    lambda_cls: float = 1.0,
    cluster_k: int = 7,
    enable_z_analysis: bool = True,
):
    """使用预先计算好的 h_fuse 表征单独训练 RGB 渲染头。"""

    if len(adata_paths) < 2:
        raise ValueError("至少需要提供两个切片的路径进行渲染训练。")

    for path in adata_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到切片数据文件：{path}")

    if not os.path.exists(h_fuse_path):
        raise FileNotFoundError(f"找不到 h_fuse 表征文件：{h_fuse_path}")

    if not os.path.exists(cluster_labels_path):
        raise FileNotFoundError(f"找不到聚类标签文件：{cluster_labels_path}")

    _set_global_seed(SEED)

    print("[渲染头] 使用的切片文件：")
    for idx, path in enumerate(adata_paths, start=1):
        print(f"  切片 {idx}: {path}")

    print("[渲染头] 正在加载数据集信息……")
    out = load_and_prepare(adata_paths)
    device = out["X_t"].device

    print(f"[渲染头] 从 {h_fuse_path} 读取预先计算的 h_fuse")
    h_fuse_np = np.load(h_fuse_path)
    h_fuse = torch.tensor(h_fuse_np, device=device, dtype=torch.float32)

    if h_fuse.shape[0] != out["X_t"].shape[0]:
        raise ValueError(
            "h_fuse 的样本数量与当前数据集不一致："
            f"{h_fuse.shape[0]} != {out['X_t'].shape[0]}"
        )

    print(f"[渲染头] 从 {cluster_labels_path} 读取预聚类标签")
    cluster_labels_np = np.load(cluster_labels_path)
    cluster_labels = torch.tensor(
        cluster_labels_np, device=device, dtype=torch.long
    )

    if cluster_labels.shape[0] != out["X_t"].shape[0]:
        raise ValueError(
            "聚类标签的样本数量与当前数据集不一致："
            f"{cluster_labels.shape[0]} != {out['X_t'].shape[0]}"
        )

    num_unique_labels = int(cluster_labels.max().item() + 1)
    if num_unique_labels > cluster_k:
        raise ValueError(
            "聚类标签的类别数超过当前设置的 cluster_k："
            f"{num_unique_labels} > {cluster_k}"
        )
    if num_unique_labels < cluster_k:
        print(
            f"[渲染头] 聚类标签包含 {num_unique_labels} 类，"
            f"将使用 cluster_k={cluster_k} 训练分类头"
        )

    print("[渲染头] 正在准备 RGB 范围统计……")
    rgb_min, rgb_max = _prepare_rgb_range(adata_paths)
    print("  R:", rgb_min[0], "~", rgb_max[0])
    print("  G:", rgb_min[1], "~", rgb_max[1])
    print("  B:", rgb_min[2], "~", rgb_max[2])

    os.makedirs(output_dir, exist_ok=True)
    print(f"[渲染头] 输出目录：{os.path.abspath(output_dir)}")

    R = RenderHeadR(
        h_dim=h_fuse.shape[1], z_dim=3, hidden=hidden_dim, gaussian=True
    ).to(device)

    # ====== 新增：重构模块与损失系数 ======
    reconstructor = GeneReconstructor(
        z_dim=3, hidden_dim=recon_hidden, output_dim=out["X_t"].shape[1]
    ).to(device)

    classifier = ZClassifier(z_dim=3, num_classes=cluster_k).to(device)

    optimizer = torch.optim.Adam(
        list(R.parameters())
        + list(reconstructor.parameters())
        + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    mu_rgb = out["mu_rgb_t"]
    var_rgb = out["var_rgb_t"]
    X_target = out["X_t"].detach()

    loss_history = []

    for epoch in range(1, epochs + 1):
        R.train()
        reconstructor.train()
        classifier.train()

        r_out = R(h_fuse)
        # loss_zkl = kl_rgb_loss(r_out["mu_z"], r_out["logvar_z"], mu_rgb, var_rgb)

        dz = r_out["z"]
        dh = h_fuse.detach()
        # W_star = _ridge_fit(dh, dz, l2=ridge_l2)
        # loss_cpl_dir = coupling_dir_loss(dz, dh, W_star)
        # loss_cpl_ratio = coupling_ratio_loss(dz, dh, W_star)

        # ====== 新增：基因表达重构损失 ======
        recon_pred = reconstructor(dz)
        loss_recon = F.mse_loss(recon_pred, X_target)

        # ====== 新增：z 分类损失 ======
        logits = classifier(dz)
        loss_cls = F.cross_entropy(logits, cluster_labels)

        # ====== 新增：z 通道正交约束 ======
        loss_ortho = orthogonality_loss(dz)

        loss = (
            # lambda_zkl * loss_zkl
            # + lambda_cpl_dir * loss_cpl_dir
            # + lambda_cpl_ratio * loss_cpl_ratio
            + lambda_recon * loss_recon
            + lambda_cls * loss_cls
            + lambda_ortho * loss_ortho
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"[第 {epoch} 轮] "
            # f"zKL={loss_zkl.item():.4f} "
            # f"| cpl_dir={loss_cpl_dir.item():.4f} "
            # f"| cpl_ratio={loss_cpl_ratio.item():.4f} "
            f"| 重构={loss_recon.item():.4f} "
            f"| 分类={loss_cls.item():.4f} "
            f"| 正交={loss_ortho.item():.4f} "
            f"| 总损失={loss.item():.4f}"
        )

        loss_history.append(
            [
                epoch,
                loss.item(),
                # loss_zkl.item(),
                # loss_cpl_dir.item(),
                # loss_cpl_ratio.item(),
                loss_recon.item(),
                loss_cls.item(),
                loss_ortho.item(),
            ]
        )

        with torch.no_grad():
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            save_epoch_z_and_rgb_global_to_img_range(
                res_z_tensor=r_out["z"],
                out_dict=out,
                epoch=epoch,
                out_dir=epoch_dir,
                rgb_min=rgb_min,
                rgb_max=rgb_max,
                point_size=25,
            )

            if enable_z_analysis:
                _analyze_z_space(
                    z_tensor=r_out["z"],
                    coords_tensor=out["coords_t"],
                    slice_tensor=out["slice_id_t"],
                    region_tensor=out.get("Region_t"),
                    out_dir=os.path.join(output_dir, "z_space_analysis"),
                    epoch=epoch,
                    cluster_k=cluster_k,
                    random_state=SEED,
                )

    loss_log_path = os.path.join(output_dir, "loss_history.csv")
    with open(loss_log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "轮次",
                "总损失",
                # "zKL",
                # "方向耦合损失",
                # "比例耦合损失",
                "重构损失",
                "分类损失",
                "正交损失",
            ]
        )
        for row in loss_history:
            writer.writerow(row)

    torch.save(
        {
            "render_head": R.state_dict(),
            "reconstructor": reconstructor.state_dict(),
            "classifier": classifier.state_dict(),
        },
        os.path.join(output_dir, "render_head.pt"),
    )
    print("训练完成，结果已保存至：", os.path.abspath(output_dir))
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用已缓存的 h_fuse 表征训练 RGB 渲染头。"
    )
    parser.add_argument(
        "--adata-path",
        action="append",
        dest="adata_paths",
        default=None,
        help=(
            "路径到包含 RGB 统计的 .h5ad 文件，可多次提供；"
            "若未指定，则默认使用 4 个切片的固定路径。"
        ),
    )
    parser.add_argument(
        "--h-fuse",
        dest="h_fuse_path",
        default=DEFAULT_H_FUSE_PATH,
        help=(
            "预先训练得到的 h_fuse 表征 (npy)；"
            "若未指定，则默认读取第 165 轮的缓存文件。"
        ),
    )
    parser.add_argument(
        "--cluster-labels",
        dest="cluster_labels_path",
        default=DEFAULT_CLUSTER_LABELS_PATH,
        help=(
            "预聚类得到的标签 (npy)；"
            "若未指定，则默认读取第 165 轮的 k=7 标签。"
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "输出目录，用于保存渲染结果与日志；"
            "默认写入 ./render_head_rgb_output。"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-zkl", type=float, default=0.01)
    parser.add_argument("--lambda-cpl-dir", type=float, default=2.0)
    parser.add_argument("--lambda-cpl-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-ortho", type=float, default=0.1)
    parser.add_argument("--lambda-cls", type=float, default=3.0)
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--recon-hidden", type=int, default=128)
    parser.add_argument("--cluster-k", type=int, default=7, help="聚类的类别数量")
    parser.add_argument(
        "--skip-z-analysis",
        action="store_true",
        help="若指定，则跳过 z 空间的聚类与 UMAP 可视化",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    adata_paths = args.adata_paths or list(DEFAULT_ADATA_PATHS)

    train_render_head(
        adata_paths=adata_paths,
        h_fuse_path=args.h_fuse_path,
        cluster_labels_path=args.cluster_labels_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_zkl=args.lambda_zkl,
        lambda_cpl_dir=args.lambda_cpl_dir,
        lambda_cpl_ratio=args.lambda_cpl_ratio,
        lambda_recon=args.lambda_recon,
        lambda_ortho=args.lambda_ortho,
        lambda_cls=args.lambda_cls,
        ridge_l2=args.ridge_l2,
        hidden_dim=args.hidden_dim,
        recon_hidden=args.recon_hidden,
        cluster_k=args.cluster_k,
        enable_z_analysis=not args.skip_z_analysis,
    )


if __name__ == "__main__":
    main()
