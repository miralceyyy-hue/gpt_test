import argparse
import csv
import os
import random
from typing import List

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F

from test_new_gat_cross_attn_res_RGB import (
    SEED,
    RenderHeadR,
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

DEFAULT_OUTPUT_DIR = "render_head_rgb_output_no_rec_no_ouhe"


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


def orthogonality_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算 z 通道之间的正交约束损失。"""

    if z.ndim != 2:
        raise ValueError("正交损失仅支持二维张量：形状应为 (N, C)")

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = torch.matmul(z_centered.T, z_centered) / (z.size(0) + eps)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum()


def train_render_head(
    adata_paths: List[str],
    h_fuse_path: str,
    output_dir: str,
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
):
    """使用预先计算好的 h_fuse 表征单独训练 RGB 渲染头。"""

    if len(adata_paths) < 2:
        raise ValueError("至少需要提供两个切片的路径进行渲染训练。")

    for path in adata_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到切片数据文件：{path}")

    if not os.path.exists(h_fuse_path):
        raise FileNotFoundError(f"找不到 h_fuse 表征文件：{h_fuse_path}")

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

    optimizer = torch.optim.Adam(
        list(R.parameters()) + list(reconstructor.parameters()),
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

        r_out = R(h_fuse)
        loss_zkl = kl_rgb_loss(r_out["mu_z"], r_out["logvar_z"], mu_rgb, var_rgb)

        dz = r_out["z"]
        dh = h_fuse.detach()
        # W_star = _ridge_fit(dh, dz, l2=ridge_l2)
        # loss_cpl_dir = coupling_dir_loss(dz, dh, W_star)
        # loss_cpl_ratio = coupling_ratio_loss(dz, dh, W_star)

        # ====== 新增：基因表达重构损失 ======
        recon_pred = reconstructor(dz)
        loss_recon = F.mse_loss(recon_pred, X_target)

        # ====== 新增：z 通道正交约束 ======
        # loss_ortho = orthogonality_loss(dz)

        loss = (
            lambda_zkl * loss_zkl
            # + lambda_cpl_dir * loss_cpl_dir
            # + lambda_cpl_ratio * loss_cpl_ratio
            + lambda_recon * loss_recon
            # + lambda_ortho * loss_ortho
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"[第 {epoch} 轮] "
            f"zKL={loss_zkl.item():.4f} "
            # f"| cpl_dir={loss_cpl_dir.item():.4f} "
            # f"| cpl_ratio={loss_cpl_ratio.item():.4f} "
            f"| 重构={loss_recon.item():.4f} "
            # f"| 正交={loss_ortho.item():.4f} "
            f"| 总损失={loss.item():.4f}"
        )

        loss_history.append(
            [
                epoch,
                loss.item(),
                loss_zkl.item(),
                # loss_cpl_dir.item(),
                # loss_cpl_ratio.item(),
                loss_recon.item(),
                # loss_ortho.item(),
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

    loss_log_path = os.path.join(output_dir, "loss_history.csv")
    with open(loss_log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "轮次",
                "总损失",
                "zKL",
                # "方向耦合损失",
                # "比例耦合损失",
                "重构损失",
                # "正交损失",
            ]
        )
        for row in loss_history:
            writer.writerow(row)

    torch.save(
        {
            "render_head": R.state_dict(),
            "reconstructor": reconstructor.state_dict(),
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
        "--output",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "输出目录，用于保存渲染结果与日志；"
            "默认写入 ./render_head_rgb_output。"
        ),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-zkl", type=float, default=0.01)
    parser.add_argument("--lambda-cpl-dir", type=float, default=2.0)
    parser.add_argument("--lambda-cpl-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-ortho", type=float, default=0.1)
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--recon-hidden", type=int, default=128)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    adata_paths = args.adata_paths or list(DEFAULT_ADATA_PATHS)

    train_render_head(
        adata_paths=adata_paths,
        h_fuse_path=args.h_fuse_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_zkl=args.lambda_zkl,
        lambda_cpl_dir=args.lambda_cpl_dir,
        lambda_cpl_ratio=args.lambda_cpl_ratio,
        lambda_recon=args.lambda_recon,
        lambda_ortho=args.lambda_ortho,
        ridge_l2=args.ridge_l2,
        hidden_dim=args.hidden_dim,
        recon_hidden=args.recon_hidden,
    )


if __name__ == "__main__":
    main()
