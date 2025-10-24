import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import os
os.environ['R_HOME'] = '/home/yangqx/.conda/envs/staligner/lib/R'
os.environ['R_USER'] = '/home/yangqx/.conda/envs/staligner/lib/python3.8/site-packages/rpy2'

import random  # 导入random模块
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import anndata as ad
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn.init as init

# ========= 配置（你可按需改动） =========
COORD_X_KEY = "coor_x"
COORD_Y_KEY = "coor_y"
K_INTRA = 6               # 同切片邻居数（任务2）
K_CROSS = 50              # 跨切片MNN初始K（会做互为最近筛选）
max_k = 6
PCA_NCOMPS = 50           # 输入特征用的PCA维度
HVG_PER_SLICE = 5000      # 每个切片内先选HVG数
SEED = 3407

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # 如果使用多GPU



# ========= 小工具 =========
def _ensure_coords(adata: ad.AnnData) -> np.ndarray:
    if (COORD_X_KEY in adata.obs.columns) and (COORD_Y_KEY in adata.obs.columns):
        coords = adata.obs[[COORD_X_KEY, COORD_Y_KEY]].to_numpy().astype(np.float32)
    elif 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'][:, :2], dtype=np.float32)
    else:
        raise KeyError(f"未找到坐标：请在 obs 中提供 '{COORD_X_KEY}','{COORD_Y_KEY}' 或 obsm['spatial']")
    return coords

def _check_rgb_obs(adata: ad.AnnData):
    keys = ['rgb_mean_R','rgb_mean_G','rgb_mean_B','rgb_var_R','rgb_var_G','rgb_var_B']
    for k in keys:
        if k not in adata.obs.columns:
            raise KeyError(f"obs 缺少列: {k}")

def _prep_single_slice(adata: ad.AnnData, n_hvg: int = HVG_PER_SLICE, n_comps: int = PCA_NCOMPS) -> ad.AnnData:
    adata = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', subset=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, max_value=10)
    # sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')
    # adata.obsm['feat'] = np.asarray(adata.obsm['X_pca'], dtype=np.float32)
    adata = adata[:, adata.var['highly_variable']]
    return adata



def _nearest_neighbors(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    d, idx = nbrs.kneighbors(X)
    return d, idx


def _to_uint8_rgb_with_target_range(z_slice_01: np.ndarray,
                                    rgb_min: np.ndarray,
                                    rgb_max: np.ndarray) -> np.ndarray:
    """
    已经把 z 映射到 0~1 的数组 (Ns,3)，再线性映射到给定的 RGB 目标范围。
    """
    # 线性拉伸到目标范围
    mapped = z_slice_01 * (rgb_max[None, :] - rgb_min[None, :]) + rgb_min[None, :]
    rgb255 = np.clip(np.round(mapped), 0, 255).astype(np.uint8)
    return rgb255

def _plot_rgb_points(coords: np.ndarray, rgb255: np.ndarray, save_path: str,
                     point_size: float = 6.0, invert_y: bool = True, title: str = None):
    """
    用散点图按坐标上色绘制；绘制顺序可控：默认按从左下到右上的顺序。
    """
    # ---- 新增：排序下标 ----
    # 方式1: 按 x 再按 y 排序
    # order = np.lexsort((coords[:, 1], coords[:, 0]))
    # 方式2: 按 x+y 总和排序（更快，但效果类似）
    order = np.argsort(coords[:,0] + coords[:,1])

    coords_sorted = coords[order]
    colors_sorted = (rgb255 / 255.0)[order]

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                c=colors_sorted, s=point_size, marker='s', linewidth=0)
    plt.gca().set_aspect('equal', adjustable='box')
    if invert_y:
        plt.gca().invert_yaxis()
    if title:
        plt.title(title)
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def _plot_grayscale_points(coords: np.ndarray, gray_image: np.ndarray, save_path: str,
                           point_size: float = 6.0, invert_y: bool = True, title: str = None):
    """
    用散点图按坐标绘制灰度图，单通道灰度图显示。
    """
    # ---- 新增：排序下标 ----
    order = np.argsort(coords[:, 0] + coords[:, 1])

    coords_sorted = coords[order]
    gray_sorted = gray_image[order]

    # 标准化灰度图像值到0~1范围
    gray_sorted_normalized = (gray_sorted - gray_sorted.min()) / (gray_sorted.max() - gray_sorted.min())

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                c=gray_sorted_normalized, s=point_size, marker='s', linewidth=0, cmap='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    if invert_y:
        plt.gca().invert_yaxis()
    if title:
        plt.title(title)
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
def _get_spatial_keys(adata, image_key='hires', sample_key=None):
    """
    返回 (img_uint8, coords_scaled)，自动从 adata.uns['spatial'] 找 sample id。
    """
    # 取 sample id（Visium 常见只有一个）
    if sample_key is None:
        sample_key = list(adata.uns['spatial'].keys())[0]
    img = adata.uns['spatial'][sample_key]['images'][image_key]        # float 0~1 或 0~255
    if img.dtype != np.uint8:
        img_uint8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    else:
        img_uint8 = img
    scale = adata.uns['spatial'][sample_key]['scalefactors'][f'tissue_{image_key}_scalef']
    coords = adata.obsm['spatial'] * scale
    return img_uint8, coords

def compute_rgb_range_from_patches(adatas, patch_size=15, p_lo=1.0, p_hi=99.0, sample_keys=None):
    """
    统计所有切片的“spot覆盖区域内像素”的RGB范围（按百分位），返回:
      rgb_min (3,), rgb_max (3,)
    - adatas: List[AnnData]
    - p_lo/p_hi: 百分位 (0~100)，例如(1,99) 抗极端
    """
    half = patch_size // 2
    hist = np.zeros((3, 256), dtype=np.int64)  # R/G/B 三个通道直方图

    for i, adata in enumerate(adatas):
        sample_key = None if sample_keys is None else sample_keys[i]
        img_uint8, coords = _get_spatial_keys(adata, image_key='hires', sample_key=sample_key)
        h, w, _ = img_uint8.shape

        coords_int = coords.astype(int)
        for x, y in coords_int:
            x0, x1 = max(0, x - half), min(w, x + half)
            y0, y1 = max(0, y - half), min(h, y + half)
            patch = img_uint8[y0:y1, x0:x1]  # (ph,pw,3)
            # 三通道分别累加直方图
            for c in range(3):
                counts = np.bincount(patch[..., c].ravel(), minlength=256)
                hist[c] += counts

    # 由直方图计算百分位
    rgb_min = np.zeros(3, dtype=np.float32)
    rgb_max = np.zeros(3, dtype=np.float32)
    for c in range(3):
        cdf = np.cumsum(hist[c])
        total = cdf[-1]
        def pct_to_val(p):
            if total == 0:
                return 0.0
            thresh = p / 100.0 * total
            return float(np.searchsorted(cdf, thresh, side='left'))
        rgb_min[c] = pct_to_val(p_lo)
        rgb_max[c] = pct_to_val(p_hi)

    return rgb_min, rgb_max






def save_epoch_z_and_rgb(res_z_tensor, out_dict, epoch: int, path: str,
                         norm_mode: str = "per_slice"):
    """
    保存当轮 z，并按切片绘制 RGB 图。
    - res_z_tensor: (N,3) torch.Tensor
    - out_dict: 你的 load_and_prepare 返回的 dict（需用到 coords_t, slice_id_t）
    - norm_mode: "per_slice"（默认）或 "global"（整批一次拉伸）
    """
    z = res_z_tensor.detach().cpu().numpy()          # (N,3)
    coords = out_dict["coords_t"].detach().cpu().numpy()   # (N,2)
    slice_id = out_dict["slice_id_t"].detach().cpu().numpy()  # (N,)

    # 1) 原始 z 数组存盘（便于后续分析/复现）
    out_dir = os.path.join(path, f'epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"z_epoch{epoch:03d}.npy"), z)

    # 2) 可视化：按切片分别绘制
    if norm_mode == "global":
        # 全局一次 min-max（让不同切片同一色标，易于比较）
        z_rgb255_global = _to_uint8_rgb(z)
    for sid in np.unique(slice_id):
        idx = (slice_id == sid)
        coords_s = coords[idx]
        z_s = z[idx]

        if norm_mode == "per_slice":
            rgb255 = _to_uint8_rgb(z_s)
        else:
            rgb255 = z_rgb255_global[idx]

        save_path = os.path.join(out_dir, f"slice{int(sid)}_epoch{epoch:03d}.png")
        _plot_rgb_points(coords_s, rgb255, save_path,
                         point_size=6.0, invert_y=True,
                         title=f"slice {int(sid)} | epoch {epoch}")
def _z_to_uint8_by_percentile(z_all: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    # z_all: (N,3) in float
    lo = np.percentile(z_all, p_lo, axis=0, keepdims=True)
    hi = np.percentile(z_all, p_hi, axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-8)
    z01 = np.clip((z_all - lo) / rng, 0.0, 1.0)
    return (np.round(z01 * 255.0)).astype(np.uint8)


def save_epoch_z_and_rgb_global_to_img_range(res_z_tensor, out_dict, epoch: int, out_dir: str,
                                                rgb_min: np.ndarray, rgb_max: np.ndarray,
                                                point_size: float = 6.0):
    """
    每个切片的RGB空间上分别绘制4张图：R, G, B 和 RGB 合成图。
    """
    z = res_z_tensor.detach().cpu().numpy()  # (N, 3)
    coords = out_dict["coords_t"].detach().cpu().numpy()
    slice_id = out_dict["slice_id_t"].detach().cpu().numpy()

    # 替换原来的 min-max + _to_uint8_rgb_with_target_range 两步
    rgb_all = _z_to_uint8_by_percentile(z, p_lo=1.0, p_hi=99.0)

    # 保存z值的原始数据
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"z_epoch{epoch:03d}.npy"), z)

    for sid in np.unique(slice_id):
        idx = (slice_id == sid)
        coords_s = coords[idx]
        rgb_s = rgb_all[idx]

        # 单独提取 RGB 通道
        r_channel = rgb_s[:, 0]  # R通道
        g_channel = rgb_s[:, 1]  # G通道
        b_channel = rgb_s[:, 2]  # B通道

        # 绘制每个切片的四张图（R, G, B, RGB合成图）
        save_path_r = os.path.join(out_dir, f"slice{int(sid)}_epoch{epoch:03d}_R.png")
        _r = np.zeros_like(rgb_s)
        _r[:, 0] = r_channel  # 仅填充R通道
        _r[:, 1] = 0  # G通道设置为0
        _r[:, 2] = 0  # B通道设置为0
        _plot_rgb_points(coords_s, _r, save_path_r,
                         point_size=point_size, invert_y=True,
                         title=f"slice {int(sid)} | epoch {epoch} | R channel")

        save_path_g = os.path.join(out_dir, f"slice{int(sid)}_epoch{epoch:03d}_G.png")
        _g = np.zeros_like(rgb_s)
        _g[:, 0] = 0  # R通道设置为0
        _g[:, 1] = g_channel  # 仅填充G通道
        _g[:, 2] = 0  # B通道设置为0
        _plot_rgb_points(coords_s, _g, save_path_g,
                         point_size=point_size, invert_y=True,
                         title=f"slice {int(sid)} | epoch {epoch} | G channel")

        save_path_b = os.path.join(out_dir, f"slice{int(sid)}_epoch{epoch:03d}_B.png")
        _b = np.zeros_like(rgb_s)
        _b[:, 0] = 0  # R通道设置为0
        _b[:, 1] = 0  # G通道设置为0
        _b[:, 2] = b_channel  # 仅填充B通道
        _plot_rgb_points(coords_s, _b, save_path_b,
                         point_size=point_size, invert_y=True,
                         title=f"slice {int(sid)} | epoch {epoch} | B channel")

        # RGB合成图
        save_path_rgb = os.path.join(out_dir, f"slice{int(sid)}_epoch{epoch:03d}_RGB.png")
        _plot_rgb_points(coords_s, rgb_s, save_path_rgb,
                         point_size=point_size, invert_y=True,
                         title=f"slice {int(sid)} | epoch {epoch} | RGB")

        # 计算灰度图 (RGB -> 灰度)
        gray_image = 0.2989 * rgb_s[:, 0] + 0.5870 * rgb_s[:, 1] + 0.1140 * rgb_s[:, 2]

        # 绘制灰度图，使用新的绘图函数
        save_path_gray = os.path.join(out_dir, f"slice{int(sid)}_epoch{epoch:03d}_Gray.png")
        _plot_grayscale_points(coords_s, gray_image, save_path_gray,
                               point_size=point_size, invert_y=True,
                               title=f"slice {int(sid)} | epoch {epoch} | Grayscale")


# ========= 主函数：读多切片 + 统一预处理 =========
def load_and_prepare(paths):
    """
    读取多个 .h5ad（至少2个），做：
      1) 各自标准化+log+HVG
      2) HVG交集后按基因交集拼接
      3) 合并后 scale + PCA → obsm['feat'] 作为输入特征 X
      4) 生成同切片 6-NN 的索引 nbr_idx_t 与几何 d_intra
      5) 计算跨切片 MNN（变长→pad 到 K_CROSS，并提供 mask）

    返回字典：含张量 X_t, coords_t, mu_rgb_t, var_rgb_t, slice_id_t, nbr_idx_t, d_intra_t,
             cross_idx_t, cross_mask_t，以及辅助统计（非张量）。
    """

    assert len(paths) >= 2, "至少提供两个切片路径"
    adatas = []
    for i, p in enumerate(paths):
        ad_i = sc.read(p)
        _check_rgb_obs(ad_i)
        ad_i = _prep_single_slice(ad_i, HVG_PER_SLICE)
        ad_i.obs['slice_id'] = i  # 标注切片编号
        adatas.append(ad_i)

    ad_all = ad.concat(adatas, label='slice_id', keys=None, index_unique=None)


    # ---- 抽出需要的观测量 ----
    coords = _ensure_coords(ad_all)                                 # (N,2)

    # 初始化 LabelEncoder
    label_encoder = LabelEncoder()
    Region = label_encoder.fit_transform(ad_all.obs['Region'])

    mu_rgb = np.stack([ad_all.obs['rgb_mean_R'].to_numpy(),
                       ad_all.obs['rgb_mean_G'].to_numpy(),
                       ad_all.obs['rgb_mean_B'].to_numpy()], axis=1).astype(np.float32)  # (N,3)
    var_rgb = np.stack([ad_all.obs['rgb_var_R'].to_numpy(),
                        ad_all.obs['rgb_var_G'].to_numpy(),
                        ad_all.obs['rgb_var_B'].to_numpy()], axis=1).astype(np.float32)  # (N,3)
    var_rgb = np.clip(var_rgb, 1e-6, None)

    slice_id = ad_all.obs['slice_id'].to_numpy().astype(np.int64)   # (N,)

    N = ad_all.n_obs
    nbr_idx = np.full((N, K_INTRA), -1, dtype=np.int64)



    for sid in np.unique(slice_id):
        idx_slice = np.where(slice_id == sid)[0]
        coords_s = coords[idx_slice]
        d_s, idx_s = _nearest_neighbors(coords_s, k=K_INTRA+1)  # 含自身
        nbr_idx_s = idx_slice[idx_s[:, 1:K_INTRA+1]]            # 去自身，映射回全局索引
        nbr_idx[idx_slice] = nbr_idx_s

    from sklearn.neighbors import NearestNeighbors

    # ---- 跨切片 MNN（任务3，基因表达值计算余弦相似度）----
    unique_slices = np.unique(slice_id)
    assert len(unique_slices) >= 2, "需要至少两个切片做MNN"

    X_raw = np.asarray(ad_all.X.todense(), dtype=np.float32)

    idx_by_slice: Dict[int, np.ndarray] = {
        sid: np.where(slice_id == sid)[0] for sid in unique_slices
    }

    cross_lists: List[set] = [set() for _ in range(N)]
    mnn_pairs: List[Tuple[int, int]] = []

    print('计算MNN配对（支持多切片）')
    for i, sid_a in enumerate(unique_slices):
        idx_a = idx_by_slice[sid_a]
        if idx_a.size == 0:
            continue
        X_a = X_raw[idx_a]
        for sid_b in unique_slices[i + 1:]:
            idx_b = idx_by_slice[sid_b]
            if idx_b.size == 0:
                continue

            X_b = X_raw[idx_b]

            n_ab = min(K_CROSS, idx_b.size)
            n_ba = min(K_CROSS, idx_a.size)
            if n_ab == 0 or n_ba == 0:
                continue

            knn_ab = NearestNeighbors(n_neighbors=n_ab, metric='cosine').fit(X_b)
            _, j_ab = knn_ab.kneighbors(X_a)

            knn_ba = NearestNeighbors(n_neighbors=n_ba, metric='cosine').fit(X_a)
            _, j_ba = knn_ba.kneighbors(X_b)

            j_ba_sets = [set(neigh.tolist()) for neigh in j_ba]

            for a_local, neighs_in_b in enumerate(j_ab):
                a_global = idx_a[a_local]
                for b_local in neighs_in_b:
                    if a_local in j_ba_sets[b_local]:
                        b_global = idx_b[b_local]
                        cross_lists[a_global].add(b_global)
                        cross_lists[b_global].add(a_global)
                        mnn_pairs.append((a_global, b_global))








    cross_idx = np.full((N, max_k), -1, dtype=np.int64)
    cross_mask = np.zeros((N, max_k), dtype=bool)
    for i, neighbors in enumerate(cross_lists):
        if not neighbors:
            continue
        neigh_arr = np.array(sorted(neighbors), dtype=np.int64)
        neigh_arr = neigh_arr[:max_k]
        cross_idx[i, :len(neigh_arr)] = neigh_arr
        cross_mask[i, :len(neigh_arr)] = True

    if mnn_pairs:
        unique_cells_by_slice: Dict[int, set] = {sid: set() for sid in unique_slices}
        for a_global, b_global in mnn_pairs:
            unique_cells_by_slice[int(slice_id[a_global])].add(int(a_global))
            unique_cells_by_slice[int(slice_id[b_global])].add(int(b_global))

        num_pairs = len(mnn_pairs)
        print(f"[MNN统计] 总配对数: {num_pairs}")
        for sid in unique_slices:
            covered = len(unique_cells_by_slice[sid])
            print(f"[MNN统计] 切片 {sid} 覆盖细胞数: {covered}")

        if 'Region' in ad_all.obs.columns:
            from collections import Counter

            region = ad_all.obs['Region'].to_numpy()
            pair_bucket = Counter()
            num_match = 0
            num_mismatch = 0
            for a_global, b_global in mnn_pairs:
                ra = region[a_global]
                rb = region[b_global]
                pair_bucket[(ra, rb)] += 1
                if ra == rb:
                    num_match += 1
                else:
                    num_mismatch += 1

            print("[MNN统计] Region一致:", num_match, "| 不一致:", num_mismatch)
            print("[MNN统计] Region组合:")
            for (ra, rb), cnt in pair_bucket.items():
                print(f"  - {ra} vs {rb}: {cnt}")
        else:
            print('[MNN统计] 未找到任何互为最近邻配对')




    print('开始合并')
    # 合并邻居时的处理
    final_nbrs = []
    valid_mask = []

    for i in range(N):
        # 获取该点的同切片邻居
        intra_neighbors = nbr_idx[i]
        # 获取该点的跨切片邻居
        cross_neighbors = cross_idx[i]

        # 如果跨切片邻居是无效的（例如全是-1），则不使用这些邻居
        valid_cross_neighbors = cross_neighbors[cross_neighbors != -1]

        # 合并邻居，去重
        combined_neighbors = np.unique(np.concatenate([intra_neighbors, valid_cross_neighbors]))
        max_n = max_k + K_INTRA
        # 创建有效值mask，标记哪些邻居是有效的
        mask = np.zeros(max_n, dtype=bool)
        mask[:len(combined_neighbors)] = True  # 有效的邻居标记为 True
        # print(len(combined_neighbors))
        final_nbrs.append(combined_neighbors)
        valid_mask.append(mask)
    # exit()
    # 将结果存入最终的邻居数组
    final_nbrs_array = np.array([np.pad(neigh, (0, max_n - len(neigh)), constant_values=-1) for neigh in final_nbrs])

    # 将 mask 数组转换为 numpy 数组
    valid_mask_array = np.array([np.pad(mask, (0, max_n - len(mask)), constant_values=False) for mask in valid_mask])

    degrees = np.sum(valid_mask, axis=1)  # 计算每个节点的有效邻居数量


    # ========= 打包为张量（变量名与后续模型对齐）=========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_t         = torch.tensor(ad_all.X.todense(),    device=device, dtype=torch.float32)  # (N,PCA_NCOMPS)

    # X_t         = torch.tensor(X,        device=device, dtype=torch.float32)   # (N,PCA_NCOMPS)
    coords_t    = torch.tensor(coords,   device=device, dtype=torch.float32)   # (N,2)
    Region_t    = torch.tensor(Region,   device=device, dtype=torch.long)
    mu_rgb_t    = torch.tensor(mu_rgb,   device=device, dtype=torch.float32)   # (N,3)
    var_rgb_t   = torch.tensor(var_rgb,  device=device, dtype=torch.float32)   # (N,3)
    slice_id_t  = torch.tensor(slice_id, device=device, dtype=torch.long)      # (N,)

    # ---- 节点度数 ----
    degree_t = torch.tensor(degrees, device=device, dtype=torch.long)  # (N,)

    nbr_idx_t   = torch.tensor(final_nbrs_array,  device=device, dtype=torch.long)      # (N,6)

    nbr_mask_t= torch.tensor(valid_mask_array,device=device, dtype=torch.bool)     # (N,max_n)



    return dict(
        # ========= 原始与对齐数据 =========
        adata_all=ad_all,  # AnnData，合并后的总对象（包含所有切片，基因已对齐）。
        # 方便后续如果要往 obs/obsm 回写结果（例如保存 z 或 loss）直接用。

        # ========= 模型输入特征 =========
        X_t=X_t,  # torch.Tensor (N, PCA_NCOMPS)，输入 MLP 的特征（合并切片后 PCA 表征）。
        # 每一行对应一个细胞。

        coords_t=coords_t,  # torch.Tensor (N,2)，空间坐标 (x,y)。只在单切片内有意义，跨切片不比较。
        # 用于任务2 的邻域与 d_intra 计算。

        mu_rgb_t=mu_rgb_t,  # torch.Tensor (N,3)，每个点在 RGB 三通道的均值 (来自 obs['rgb_mean_*'])。
        var_rgb_t=var_rgb_t,  # torch.Tensor (N,3)，每个点在 RGB 三通道的方差 (来自 obs['rgb_var_*'])。
        # 任务1 中作为“目标高斯分布参数”，和模型输出的 (mu, logvar) 做 KL 散度。

        slice_id_t=slice_id_t,  # torch.LongTensor (N,)，每个点属于哪个切片（0,1,...）。
        # 用于区分 intra-slice vs cross-slice 邻居。

        # ========= 节点度数 =========
        degree_t=degree_t,  # torch.LongTensor (N,)，每个点的度数。

        # ========= 任务2（同切片邻域） =========
        nbr_idx_t=nbr_idx_t,  # torch.LongTensor (N,6)，每个点在**同一切片**内的 6 个最近邻的全局索引。
        # 用它从 z 中索引邻居 → (N,6,3)。

        nbr_mask_t=nbr_mask_t,  # torch.BoolTensor (N,K_max)，对应 cross_idx_t 的有效掩码。
        # True 表示该位置有真实邻居，False 表示 pad 出来的空位。
        Region_t = Region_t
    )

# Step 4: 定义训练过程（最小共享嵌入 + 邻域索引展开）
class RGBModel(nn.Module):
    """
    共享编码器：x -> (mu, logvar, z)，其中 z ~ N(mu, diag(exp(logvar))).
    仅在 forward 接收到 nbr_idx_t 时，返回 z_neighbors = z[nbr_idx_t] (N,6,3)。
    先不做注意力聚合/损失，后续逐步补。
    """
    def __init__(self, input_dim, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 替换 ReLU 为 LeakyReLU,
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 替换 ReLU 为 LeakyReLU,
            nn.Linear(16, 8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 替换 ReLU 为 LeakyReLU,
        )
        self.fc_mu     = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

    def forward(self, x, nbr_idx_t=None):
        """
        x          : (N, input_dim)  —— 例如 out["X_t"]
        nbr_idx_t  : (N, 6) LongTensor，可选。若提供，则返回 z_neighbors = z[nbr_idx_t].
        return     : dict{ 'mu','logvar','z', ['z_neighbors'] }
        """
        h = self.encoder(x)                 # (N, 8)
        mu = self.fc_mu(h)                  # (N, 3)
        logvar = self.fc_logvar(h)          # (N, 3)

        # reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std                  # (N, 3)

        out = {"mu": mu, "logvar": logvar, "z": z}

        if nbr_idx_t is not None:
            # 确保索引与 z 在同一设备
            nbr_idx_t = nbr_idx_t.to(z.device)
            z_neighbors = z[nbr_idx_t]      # (N, 6, 3)
            out["z_neighbors"] = z_neighbors

        return out

# 新增：高维表示编码器（替代/重命名原 RGBModel）
class HEncoder(nn.Module):
    """
    x -> h；可选返回 h_neighbors = h[nbr_idx_t]
    """
    def __init__(self, input_dim: int):
        super().__init__()

        self.dropout = nn.Dropout(0.2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        )

        init.xavier_normal_(self.encoder[0].weight, gain=1.414)  # encoder[0] 是 nn.Linear(emb_dim, emb_dim)
        init.zeros_(self.encoder[0].bias)
    def forward(self, x: torch.Tensor, nbr_idx_t: torch.Tensor = None, nbr_mask_t: torch.Tensor = None, degree_t: torch.Tensor = None):


        h = self.dropout(self.encoder(x))  # (N, input_dim)

        out = {"h": h}  # 初始输出，包含每个点的特征表示

        if nbr_idx_t is not None and nbr_mask_t is not None:
            # 确保邻居索引和掩码被转换到与输入特征相同的设备
            nbr_idx_t = nbr_idx_t.to(h.device)
            nbr_mask_t = nbr_mask_t.to(h.device)

            # 创建一个布尔掩码，表示有效邻居（即不是 -1）
            valid_mask = nbr_mask_t  # (N, max_k), 标记哪些邻居有效

            # 通过 valid_mask 来过滤掉无效的邻居
            # 这里只取有效的邻居索引，并用它们来获取特征
            h_neighbors = torch.zeros_like(h).unsqueeze(1).repeat(1, nbr_idx_t.shape[1], 1)  # (N, max_k, input_dim)
            # 对于每个有效的邻居，将其对应位置的特征提取出来
            for i in range(nbr_idx_t.shape[0]):  # 遍历每个点
                valid_indices = nbr_idx_t[i, valid_mask[i]]  # 只取有效的邻居索引
                h_neighbors[i, :len(valid_indices), :] = h[valid_indices]  # 填充有效邻居特征

            # 存储有效邻居特征到输出字典中
            out["h_neighbors"] = h_neighbors  # (N, max_k, input_dim)

        return out



# ====== 放到你的训练脚本里（模型定义之后）======
def kl_rgb_loss(mu_pred: torch.Tensor,
                logvar_pred: torch.Tensor,
                mu_rgb_t: torch.Tensor,
                var_rgb_t: torch.Tensor,
                eps: float = 1e-8,
                reduce: str = "mean") -> torch.Tensor:
    """
    mu_pred     : (N,3)  模型输出均值
    logvar_pred : (N,3)  模型输出 log 方差
    mu_rgb_t    : (N,3)  观测RGB均值（来自obs）
    var_rgb_t   : (N,3)  观测RGB方差（来自obs）
    """
    var_pred = torch.exp(logvar_pred)                   # (N,3)
    var_obs  = torch.clamp(var_rgb_t, min=eps)         # (N,3)
    log_ratio = torch.log(var_obs + eps) - torch.log(var_pred + eps)
    frac = (var_pred + (mu_pred - mu_rgb_t)**2) / (var_obs + eps)
    kl = 0.5 * (log_ratio + frac - 1.0).sum(dim=-1)    # (N,)
    if reduce == "mean":
        return kl.mean()
    elif reduce == "sum":
        return kl.sum()
    return kl

class IntraEnvAggregator(nn.Module):
    """
    输入/输出都在 h 空间；返回 h_env (N, h_dim)
    """
    def __init__(self, emb_dim: int = 32, out_dim: int = 10, num_heads: int = 2,dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = emb_dim
        self.head_dim = emb_dim // num_heads  # 每个头的维度

        assert self.head_dim * num_heads == emb_dim, "d_model must be divisible by num_heads"

        # 第一层的权重矩阵
        self.Wq1 = nn.Linear(emb_dim, emb_dim, bias=True)
        self.Wk1 = nn.Linear(emb_dim, emb_dim, bias=True)
        self.Wv1 = nn.Linear(emb_dim, emb_dim, bias=True)
        self.Wo1 = nn.Linear(emb_dim, out_dim, bias=True)



        self.scale = self.head_dim ** 0.5
        self.drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, nbr_mask: torch.Tensor):


        # 创建一个布尔掩码，表示有效邻居（即不是 -1）
        valid_mask = nbr_mask  # (N, max_k), 标记哪些邻居有效

        # 通过 valid_mask 来过滤掉无效的邻居
        # 这里只取有效的邻居索引，并用它们来获取特征
        h_neighbors = torch.zeros_like(h).unsqueeze(1).repeat(1, nbr_idx.shape[1], 1)  # (N, max_k, h_dim)
        # 对于每个有效的邻居，将其对应位置的特征提取出来
        for i in range(nbr_idx.shape[0]):  # 遍历每个点
            valid_indices = nbr_idx[i, valid_mask[i]]  # 只取有效的邻居索引
            h_neighbors[i, :len(valid_indices), :] = h[valid_indices]  # 填充有效邻居特征


        N = h.size(0)
        K = h_neighbors.size(1)

        # 第一层聚合
        # 计算每个头的 q, k, v
        q1 = self.Wq1(h).view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, 1, head_dim)
        k1 = self.Wk1(h_neighbors).view(N, K, self.num_heads, self.head_dim).transpose(1,
                                                                                     2)  # (N, num_heads, K, head_dim)
        v1 = self.Wk1(h_neighbors).view(N, K, self.num_heads, self.head_dim).transpose(1,
                                                                                     2)  # (N, num_heads, K, head_dim)

        # 计算注意力得分
        scores1 = (q1 * k1).sum(dim=-1) / self.scale  # (N, num_heads, K)
        # scores1 = (q1 - k1).sum(dim=-1)  # (N, num_heads, K)

        scores1 = scores1.masked_fill(nbr_mask.unsqueeze(1) == 0, float('-inf'))  # (N, num_heads, K)

        # 计算注意力
        attn1 = torch.sigmoid(scores1)

        attn1 = torch.softmax(attn1, dim=-1)  # (N, num_heads, K)
        attn1 = self.drop(attn1)
        context1 = (attn1.unsqueeze(-1) * v1).sum(dim=-2)  # (N, num_heads, head_dim)
        # 将多个头拼接起来
        context1 = context1.transpose(1, 2).contiguous().view(N, self.d_model)  # (N, d_model)

        context1 = context1 + h #残差

        # 通过线性变换获得输出
        delta1 = self.Wo1(context1)  # (N, emb_dim)
        h_env1 = delta1

        h_env1 = self.norm1(h_env1)

        return h_env1
# class IntraEnvAggregator(nn.Module):
#     """
#     输入/输出都在 h 空间；返回 h_env (N, h_dim)
#     """
#     def __init__(self, emb_dim: int = 32, out_dim: int = 10, num_heads: int = 3,dropout: float = 0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.d_model = out_dim * num_heads
#         self.head_dim = out_dim  # 每个头的维度
#
#         # 第一层的权重矩阵
#         self.Wq1 = nn.Linear(emb_dim, out_dim * num_heads, bias=True)
#         self.Wk1 = nn.Linear(emb_dim, out_dim * num_heads, bias=True)
#         self.Wv1 = nn.Linear(emb_dim, out_dim * num_heads, bias=True)
#         self.Wo1 = nn.Linear(out_dim * num_heads, out_dim, bias=True)
#
#         init.xavier_normal_(self.Wq1.weight, gain=1.414)  # (emb_dim, out_dim * num_heads)
#         init.xavier_normal_(self.Wk1.weight, gain=1.414)  # (emb_dim, out_dim * num_heads)
#         init.xavier_normal_(self.Wo1.weight, gain=1.414)
#
#         init.zeros_(self.Wq1.bias)
#         init.zeros_(self.Wk1.bias)
#         init.zeros_(self.Wo1.bias)
#
#         self.scale = self.head_dim ** 0.5
#         self.drop = nn.Dropout(dropout)
#
#         self.norm1 = nn.LayerNorm(out_dim)
#
#     def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, nbr_mask: torch.Tensor):
#
#         # 获取样本数量 N 和最大邻居数 max_k
#         N, max_k = nbr_idx.shape
#
#         # 创建一个布尔掩码，表示有效邻居（即不是 -1）
#         valid_mask = nbr_mask  # (N, max_k)，标记哪些邻居有效
#
#         # 在 nbr_idx 中将样本自身的索引加入，每个节点的邻居索引中加入其自身（即自身索引）
#         self_idx = torch.arange(N, device=h.device)  # (N,) 创建一个包含节点本身索引的张量
#         nbr_idx_with_self = torch.cat([self_idx.unsqueeze(1), nbr_idx], dim=-1)  # 在每行的开头加入样本本身的索引
#         valid_mask_with_self = torch.cat([torch.ones(N, 1, device=h.device), valid_mask], dim=-1)  # 在每行的开头标记样本本身为有效邻居
#
#         # 创建 h_neighbors 张量，用来存储每个节点的邻居特征
#         h_neighbors = torch.zeros_like(h).unsqueeze(1).repeat(1, nbr_idx_with_self.shape[1], 1)  # (N, max_k+1, h_dim)
#
#         # 对于每个有效的邻居，将其对应位置的特征提取出来
#         for i in range(N):  # 遍历每个点
#             valid_indices = nbr_idx_with_self[i, valid_mask_with_self[i].bool()]  # 取出有效邻居索引，包含自身
#             h_neighbors[i, :len(valid_indices), :] = h[valid_indices]  # 填充有效邻居特征
#
#
#         N = h.size(0)
#         K = h_neighbors.size(1)
#
#         # 第一层聚合
#         # 计算每个头的 q, k, v
#         q1 = self.Wq1(h).view(N, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, 1, head_dim)
#         k1 = self.Wk1(h_neighbors).view(N, K, self.num_heads, self.head_dim).transpose(1,
#                                                                                      2)  # (N, num_heads, K, head_dim)
#         v1 = self.Wk1(h_neighbors).view(N, K, self.num_heads, self.head_dim).transpose(1,
#                                                                                      2)  # (N, num_heads, K, head_dim)
#
#         # 计算注意力得分
#         scores1 = (q1 * k1).sum(dim=-1) / self.scale  # (N, num_heads, K)
#         # scores1 = (q1 + k1).sum(dim=-1)  # (N, num_heads, K)
#
#         scores1 = scores1.masked_fill(valid_mask_with_self.unsqueeze(1) == 0, float('-inf'))  # (N, num_heads, K)
#
#         # 计算注意力
#         attn1 = torch.sigmoid(scores1)
#         attn1 = torch.softmax(scores1, dim=-1)  # (N, num_heads, K)
#         attn1 = self.drop(attn1)
#         context1 = (attn1.unsqueeze(-1) * v1).sum(dim=-2)  # (N, num_heads, head_dim)
#         # 将多个头拼接起来
#         context1 = context1.transpose(1, 2).contiguous().view(N, self.d_model)  # (N, d_model)
#         # 通过线性变换获得输出
#         delta1 = self.Wo1(context1)  # (N, emb_dim)
#         h_env1 = delta1
#
#         h_env1 = self.norm1(h_env1)
#
#         return h_env1


# ====== 任务3：多正样本 InfoNCE（跨切片 MNN）======

# class DecoderMLP(nn.Module):
#     def __init__(self, input_dim, output_dim, num_slices=2):
#         super(DecoderMLP, self).__init__()
#         self.num_slices = num_slices  # 处理切片的数量
#         self.fc1 = nn.ModuleList([nn.Linear(input_dim, 50) for _ in range(num_slices)])  # 为每个切片分配一个线性层
#         self.fc2 = nn.ModuleList([nn.Linear(50, 50) for _ in range(num_slices)])
#         self.fc3 = nn.ModuleList([nn.Linear(50, output_dim) for _ in range(num_slices)])
#
#         # 使用 LeakyReLU 激活函数
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
#
#     def forward(self, x, slice_ids):
#         """
#         批量处理输入并根据切片ID选择相应的解码器通道
#         :param x: 输入数据（大小为 [batch_size, input_dim]）
#         :param slice_ids: 切片ID（大小为 [batch_size]）
#         :return: 输出结果（大小为 [batch_size, output_dim]）
#         """
#         outputs = []
#         for i in range(x.size(0)):  # 遍历每个样本
#             slice_id = slice_ids[i].item()  # 获取当前样本的切片ID
#             xi = x[i:i + 1]  # 获取当前样本的数据
#             # print(f"Processing sample {i}, slice_id = {slice_id}")
#
#             # 使用对应的通道进行前向传播
#             xi = self.leaky_relu(self.fc1[slice_id](xi))
#             xi = self.leaky_relu(self.fc2[slice_id](xi))
#             xi = self.fc3[slice_id](xi)
#             outputs.append(xi)
#
#         return torch.cat(outputs, dim=0)  # 合并所有输出

class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization for each slice
    """

    def __init__(self, num_features, n_domains, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domains = n_domains
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for _ in range(n_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)
        for i in range(self.n_domains):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]

        return out

class DecoderMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_slices=2):
        super(DecoderMLP, self).__init__()
        self.num_slices = num_slices  # 处理切片的数量

        # 共享的线性层
        self.fc1 = nn.Linear(input_dim, output_dim)  # 第一层：从输入到输出维度
        self.fc2 = nn.Linear(output_dim, output_dim)  # 第二层：输出维度到输出维度

        # 领域特定的 BatchNorm 层
        self.dsbnorm = DSBatchNorm(output_dim, num_slices)  # 使用 DSBatchNorm 对每个切片进行归一化

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, slice_ids):
        """
        批量处理输入并根据切片ID选择相应的解码器通道
        :param x: 输入数据（大小为 [batch_size, input_dim]）
        :param slice_ids: 切片ID（大小为 [batch_size]）
        :return: 输出结果（大小为 [batch_size, output_dim]）
        """

        x = self.dropout(F.elu(self.dsbnorm(self.fc1(x), slice_ids)))

        # 第二层线性变换
        x = self.fc2(x)

        return x

# class DecoderMLP(nn.Module):
#     def __init__(self, input_dim, output_dim, num_slices=2):
#         super(DecoderMLP, self).__init__()
#         self.num_slices = num_slices  # 处理切片的数量
#         self.fc1 = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_slices)])  # 为每个切片分配一个线性层
#         self.fc2 = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_slices)])
#         self.fc3 = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_slices)])
#
#         # 使用 LeakyReLU 激活函数
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
#
#     def forward(self, x, slice_ids):
#         """
#         批量处理输入并根据切片ID选择相应的解码器通道
#         :param x: 输入数据（大小为 [batch_size, input_dim]）
#         :param slice_ids: 切片ID（大小为 [batch_size]）
#         :return: 输出结果（大小为 [batch_size, output_dim]）
#         """
#         outputs = []
#         for i in range(x.size(0)):  # 遍历每个样本
#             slice_id = slice_ids[i].item()  # 获取当前样本的切片ID
#             xi = x[i:i + 1]  # 获取当前样本的数据
#             # print(f"Processing sample {i}, slice_id = {slice_id}")
#
#             # 使用对应的通道进行前向传播
#             xi = self.leaky_relu(self.fc1[slice_id](xi))
#             xi = self.leaky_relu(self.fc2[slice_id](xi))
#             xi = self.fc3[slice_id](xi)
#             outputs.append(xi)
#
#         return torch.cat(outputs, dim=0)  # 合并所有输出


# 新增：渲染头 R（h -> (mu_z, logvar_z, z)）
class RenderHeadR(nn.Module):
    def __init__(self, h_dim: int = 32, z_dim: int = 3, hidden: int = 64, gaussian: bool = True):
        super().__init__()
        self.gaussian = gaussian
        out_dim = 2 * z_dim if gaussian else z_dim
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 替换 ReLU 为 LeakyReLU,
            nn.Linear(hidden, out_dim)
        )

    def forward(self, h: torch.Tensor):
        y = self.net(h)  # (N, out_dim)
        if self.gaussian:
            mu_z, logvar_z = y.chunk(2, dim=-1)     # (N,3), (N,3)

            logvar_z = torch.clamp(logvar_z, min=-3.0, max=3.0)  # 方差 e^[−3,3]，可视性更稳定

            return {"mu_z": mu_z, "logvar_z": logvar_z, "z": mu_z}
        else:
            z = y
            return {"mu_z": z, "logvar_z": torch.zeros_like(z), "z": z}

# ---- 标尺 W*：批内岭回归闭式解（不参与反传）----
def _ridge_fit(Hd: torch.Tensor, Zd: torch.Tensor, l2: float = 1e-3) -> torch.Tensor:
    """
    Hd: (B, Dh)   Δh 堆叠
    Zd: (B, 3)    对应的 Δz^(R) 堆叠
    returns W*: (Dh, 3)   最小二乘 + 岭正则的闭式解
    """
    Dh = Hd.shape[1]
    A = Hd.T @ Hd + l2 * torch.eye(Dh, device=Hd.device, dtype=Hd.dtype)  # (Dh, Dh)
    B = Hd.T @ Zd                                                         # (Dh, 3)
    W = torch.linalg.solve(A, B)                                          # (Dh, 3)
    return W

# ---- 耦合线损失：方向为主，幅度为辅（可选）----
def coupling_dir_loss(delta_z: torch.Tensor, delta_h: torch.Tensor, W_star: torch.Tensor, eps: float = 1e-8):
    """
    方向一致性：1 - cos(Δz^(R), W*Δh)
    只更新 R；W*、Δh stopgrad
    """
    pred = delta_h @ W_star            # (B,3)
    num = (delta_z * pred).sum(dim=-1)
    den = (delta_z.norm(dim=-1) + eps) * (pred.norm(dim=-1) + eps)
    cos = num / den
    return (1.0 - cos).mean()

def coupling_ratio_loss(delta_z: torch.Tensor, delta_h: torch.Tensor, W_star: torch.Tensor, eps: float = 1e-8):
    """
    幅度比的稳健约束：log( ||Δz|| / ||W*Δh|| )^2
    小权重使用，避免量纲过强约束
    """
    pred = delta_h @ W_star
    r = delta_z.norm(dim=-1) / (pred.norm(dim=-1) + eps)
    return (r.log() ** 2).mean()


def info_nce_loss_fuse_self(        # graph_recon_loss
        h_fuse: torch.Tensor,  # (N, h_dim)  # h_fuse 表示
        nbr_idx_t: torch.Tensor,  # (N, Kmax)  # 同切片邻居索引
        nbr_mask: torch.Tensor,  # (N, Kmax)  # 同切片邻居有效掩码
        temperature: float = 0.07,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    新的对比学习损失（`h_fuse` 和 `h_fuse`，邻居为正样本）
    - 正样本为每个样本的邻居
    - 负样本为其他样本
    """
    device = h_fuse.device
    N = h_fuse.shape[0]

    # 归一化 h_fuse
    h_fuse_norm = h_fuse / (h_fuse.norm(dim=-1, keepdim=True) + eps)

    # 相似度矩阵：计算所有点对之间的相似度
    sim_matrix = torch.mm(h_fuse_norm, h_fuse_norm.T) / temperature  # (N, N)

    # 使用 masked_fill 来替代原地操作，避免直接修改计算图中的张量
    sim_matrix = sim_matrix.masked_fill(torch.eye(N, device=device).bool(), -1e9)

    # 构造正样本掩码：同切片邻居 + 跨切片邻居
    pos_mask = torch.zeros((N, N), dtype=torch.bool, device=device)

    # 同切片邻居掩码更新：确保只包括有效的邻居
    valid_nbr_mask = nbr_mask.bool()  # (N, Kmax), 转为布尔型
    pos_mask[torch.arange(N).unsqueeze(1), nbr_idx_t] = valid_nbr_mask  # 用 valid_nbr_mask 来设置 pos_mask

    # 计算正样本的相似度
    pos_sim = sim_matrix * pos_mask  # 只保留正样本的相似度

    # 计算损失
    exp_sim = torch.exp(sim_matrix)  # 所有相似度的exp值
    # 检查 exp_sim 是否存在 NaN 或 Inf
    if torch.any(torch.isnan(exp_sim)) or torch.any(torch.isinf(exp_sim)):
        print("NaN or Inf detected in exp_sim")
    exp_pos_sim = exp_sim * pos_mask  # 正样本的exp相似度
    # 检查 exp_sim 是否存在 NaN 或 Inf
    if torch.any(torch.isnan(exp_pos_sim)) or torch.any(torch.isinf(exp_pos_sim)):
        print("NaN or Inf detected in exp_pos_sim")
    exp_all_sim = exp_sim.sum(dim=1, keepdim=True)  # 所有样本的exp相似度总和
    if torch.any(torch.isnan(exp_all_sim)) or torch.any(torch.isinf(exp_all_sim)):
        print("NaN or Inf detected in exp_all_sim")

    loss = -torch.log(exp_pos_sim.sum(dim=1) / (exp_all_sim + 1e-8)).mean()
    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
        print("NaN or Inf detected in loss")
    return loss

from sklearn.mixture import GaussianMixture

def update_neighbor_mask_with_clustering(h_fuse: torch.Tensor, nbr_idx: torch.Tensor, nbr_mask: torch.Tensor, epoch: int = 0,
                                         update_interval: int = 10, num_clusters: int = 7):
    """
    更新每个点的邻居 mask，删除不在同一簇中的邻居。
    使用高斯混合模型 (Gaussian Mixture Model, GMM) 对 h_fuse 进行聚类。
    """
    # # 只有在达到更新间隔时才更新邻居 mask
    # if epoch % update_interval != 0:
    #     return nbr_mask

    N, _ = h_fuse.shape  # N是节点数，H_DIM是每个节点的特征维度

    # Step 1: 高斯混合模型 (GMM) 聚类
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    h_fuse_np = h_fuse.cpu().detach().numpy()  # 将h_fuse转换为numpy数组供GMM使用
    gmm.fit(h_fuse_np)  # 使用GMM对 h_fuse 进行聚类

    # 获取每个点所属的簇标签
    cluster_labels = gmm.predict(h_fuse_np)  # (N,) 每个点对应的簇标签

    # Step 2: 打印更新前后的邻居统计信息
    # 更新前，统计每个点的邻居数量
    nbr_count_before = nbr_mask.sum(dim=1)  # 每个点的邻居数量
    avg_nbr_before = nbr_count_before.float().mean().item()  # 转换为 float 后计算均值
    print(
        f"Before updating nbr mask - Total neighbors: {nbr_count_before.sum().item()}, Average neighbors: {avg_nbr_before:.2f}")


    # Step 2: 更新邻居 mask
    updated_nbr_mask = nbr_mask.clone()  # 复制原始邻居 mask

    for i in range(N):
        # 获取当前点的邻居索引（仅获取有效邻居）
        valid_neighbors_idx = nbr_idx[i][updated_nbr_mask[i]]  # 筛选出当前有效的邻居
        valid_neighbors_embeddings = h_fuse[valid_neighbors_idx]  # 获取这些有效邻居的 h_fuse 嵌入

        # 获取当前点所属的簇
        current_point_cluster = cluster_labels[i]

        # Step 3: 更新每个邻居的 mask，保留同一簇中的邻居
        for j, neighbor_idx in enumerate(valid_neighbors_idx):
            # 获取邻居点的簇标签
            neighbor_cluster = cluster_labels[neighbor_idx]

            # 如果邻居不在同一个簇中，则删除该邻居
            if current_point_cluster != neighbor_cluster:
                updated_nbr_mask[i, nbr_idx[i] == neighbor_idx] = False  # 更新邻居的 mask，删除不在同一簇的邻居

    # 更新后统计邻居数量
    nbr_count_after = updated_nbr_mask.sum(dim=1)  # 更新后的邻居数量
    avg_nbr_after = nbr_count_after.float().mean().item()  # 转换为 float 后计算均值
    print(
        f"After updating nbr mask - Total neighbors: {nbr_count_after.sum().item()}, Average neighbors: {avg_nbr_after:.2f}")

    # 统计更新后的全0邻居数（即没有有效邻居的点）
    num_zero_neighbors = (nbr_count_after == 0).sum().item()
    print(f"Number of points with zero neighbors: {num_zero_neighbors}")
    # exit()
    return updated_nbr_mask


def normalize_coords_per_slice(coords: torch.Tensor, slice_ids: torch.Tensor) -> torch.Tensor:
    """
    按每个切片标准化坐标
    coords: 原始坐标，形状为 (N, 2)，torch.Tensor
    slice_ids: 每个点所属的切片 ID，形状为 (N,)，torch.Tensor
    """
    normalized_coords = []
    unique_slices = torch.unique(slice_ids)  # 获取所有独特的切片ID

    for slice_id in unique_slices:
        # 获取当前切片的坐标
        slice_coords = coords[slice_ids == slice_id]

        # 计算最小值和最大值
        min_vals = slice_coords.min(dim=0).values
        max_vals = slice_coords.max(dim=0).values

        # 进行标准化
        range_vals = max_vals - min_vals
        normalized_slice_coords = (slice_coords - min_vals) / range_vals  # 标准化

        normalized_coords.append(normalized_slice_coords)

    # 将所有切片的标准化坐标拼接在一起
    return torch.cat(normalized_coords, dim=0)


class UnifiedDecoder(nn.Module):
    def __init__(self, input_dim, coord_output_dim):
        super(UnifiedDecoder, self).__init__()

        # 共享部分：用于生成嵌入特征
        self.shared_fc1 = nn.Linear(input_dim, 16)
        self.shared_fc2 = nn.Linear(16, 8)
        # 空间坐标重构分支
        self.coord_fc = nn.Linear(8, coord_output_dim)  # 输出空间坐标（2D）

    def forward(self, x):
        # 共享部分
        h = torch.relu(self.shared_fc1(x))
        h = torch.relu(self.shared_fc2(h))
        # 空间坐标重构
        coord_output = self.coord_fc(h)  # (N, coord_output_dim)

        return coord_output


if __name__ == "__main__":
    # out = load_and_prepare('/home/lix/YYY/RGB/73_74_with_pca.h5ad')

    paths = [
        "/home/yangqx/YYY/151673_RGB.h5ad",
        "/home/yangqx/YYY/151674_RGB.h5ad",
        "/home/yangqx/YYY/151675_RGB.h5ad",
        "/home/yangqx/YYY/151676_RGB.h5ad",
    ]
    out = load_and_prepare(paths)



    device = out["X_t"].device

    # 初始化 nbr_mask 为全 1 的张量，表示所有邻居有效
    nbr_mask = out["nbr_mask_t"]           # (N, max_k)
    paths = [
        "/home/yangqx/YYY/151673_RGB.h5ad",
        "/home/yangqx/YYY/151674_RGB.h5ad",
        "/home/yangqx/YYY/151675_RGB.h5ad",
        "/home/yangqx/YYY/151676_RGB.h5ad",
    ]
    adatas_list = [sc.read(p) for p in paths]
    rgb_min, rgb_max = compute_rgb_range_from_patches(
        adatas_list, patch_size=15, p_lo=1, p_hi=99
    )
    print("Global RGB target range from tissue coverage (percentiles):")
    print("  R:", rgb_min[0], "~", rgb_max[0])
    print("  G:", rgb_min[1], "~", rgb_max[1])
    print("  B:", rgb_min[2], "~", rgb_max[2])

    print(out["X_t"].shape[1])


    # === 1) 模型组件 ===
    H_DIM = 512
    out_dim = 32
    enc = HEncoder(input_dim=out["X_t"].shape[1]).to(device)
    intra_agg1 = IntraEnvAggregator(emb_dim=out["X_t"].shape[1], out_dim=H_DIM, dropout=0.2).to(device)
    intra_agg2 = IntraEnvAggregator(emb_dim=H_DIM, out_dim=out_dim, dropout=0.2).to(device)
    # 目标是重构原始的 PCA 特征，假设原始输入特征维度为 `input_dim`
    decoder = DecoderMLP(out_dim, out["X_t"].shape[1]).to(device)
    # 定义解码器
    decoder_coord = UnifiedDecoder(out_dim, coord_output_dim=2).to(device)
    R = RenderHeadR(h_dim=out_dim, z_dim=3, hidden=32, gaussian=True).to(device)

    optimizer = torch.optim.Adam(
        list(enc.parameters()) +
        list(intra_agg1.parameters()) +
        list(intra_agg2.parameters()) +
        list(decoder_coord.parameters()) +
        list(R.parameters()),
        lr=1e-4, weight_decay=1e-5
    )

    normalized_coords = normalize_coords_per_slice(out["coords_t"], out["slice_id_t"])

    # === 2) 训练超参 ===
    EPOCHS = 500
    # 渲染/耦合（只更新 R）
    lambda_zkl   = 0.01         # z 端 KL（弱化 H&E 主导）
    lambda_cpl_d = 2.0          # 耦合线方向
    lambda_cpl_r = 0.5          # 耦合线幅度（辅）
    # Fusion（只更新 fusion_head）

    lambda_align = 0.3

    update_interval = 10        #更新频率
    max_up_epoch = 41           #最大更新epoch

    # update_interval = 20  # 更新频率
    # max_up_epoch = 81  # 最大更新epoch

    # update_interval = 30  # 更新频率
    # max_up_epoch = 121  # 最大更新epoch

    # 初始化字典保存每个切片的最佳ARI值和其对应的epoch
    best_ari_per_slice = {}  # key: slice_id, value: (best_ari, best_epoch)

    # --- 小工具：根据 d 或 u_feat 计算 m(d_hat)（与 band_loss_* 同口径） ---

    for epoch in range(1, EPOCHS + 1):


        enc.train(); intra_agg1.train(); intra_agg2.train(); R.train()

        # ---- 编码到 h，并展开同片邻居 ----
        res_h = enc(out["X_t"], nbr_idx_t=out["nbr_idx_t"], nbr_mask_t=nbr_mask, degree_t=out['degree_t'])   # {'h','h_neighbors'}
        h = res_h["h"]; h_neighbors = res_h["h_neighbors"]

        # ---- 任务2：同片一次性注意力 + 带状容忍（在 h 空间）----
        h_fuse_temp = intra_agg1(h, out["nbr_idx_t"], nbr_mask)
        h_fuse = intra_agg2(h_fuse_temp, out["nbr_idx_t"], nbr_mask)
        # h_fuse = intra_out["h_env"]

        # ---- 渲染 & z-KL（只训练 R；h 不回传）----
        with torch.no_grad():
            h_fuse_stopped = h_fuse.detach()
        r_base = R(h_fuse_stopped)  # {'mu_z', 'logvar_z', 'z'}
        # 计算 KL 损失
        loss_zkl = kl_rgb_loss(
            r_base["mu_z"], r_base["logvar_z"], out["mu_rgb_t"], out["var_rgb_t"]
        )

        # ---- 耦合线（只训练 R；W* 为批内岭回归标尺，stopgrad）----
        dz = r_base["z"]   # (N,3)
        dh = h_fuse_stopped      # (N,H_DIM)


        W_star = _ridge_fit(dh, dz, l2=1e-3)         # (H_DIM,3)

        loss_cpl_dir   = coupling_dir_loss(dz, dh, W_star)
        loss_cpl_ratio = coupling_ratio_loss(dz, dh, W_star)


        # === (G) 对比学习损失 ===
        contrastive_loss_fuse_self = info_nce_loss_fuse_self(
            h_fuse, out["nbr_idx_t"], nbr_mask
        )

        # reconstructed_coords = decoder_coord(h_fuse)
        # coord_reconstruction_loss = nn.MSELoss()(reconstructed_coords, normalized_coords)

        # 解码器输出
        reconstructed_input = decoder(h_fuse, out['slice_id_t'])
        # 计算重构损失
        loss_reconstruction = nn.MSELoss()(reconstructed_input, out["X_t"])

        # with torch.no_grad():
        #     h_fuse_stop = h_fuse.detach()
        #
        # contrastive_loss_h_fuse = info_nce_loss_h_fuse(
        #     h_fuse_stop, h
        # )

        # ---- 总损失 & 反传 ----
        loss = (
            lambda_zkl   * loss_zkl
          + contrastive_loss_fuse_self/10
          + loss_reconstruction
          + loss_cpl_dir
          + loss_cpl_ratio
          # + coord_reconstruction_loss
          # + contrastive_loss_h_fuse

        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 300:
            print(f"Updating neighbor mask at epoch {epoch}")
            updated_nbr_mask = update_neighbor_mask_with_clustering(h_fuse, out["nbr_idx_t"], nbr_mask,
                                                                    update_interval=update_interval,
                                                                    epoch=epoch)
            # 使用更新后的邻居 mask 进行后续计算
            nbr_mask = updated_nbr_mask
        print(f"[Epoch {epoch}] "
              f"zKL={loss_zkl.item():.4f} "
              # f"| contrastive={contrastive_loss_fuse_self.item():.4f} | contrastive={contrastive_loss_h_fuse.item():.4f}}}"
              f"| contrastive={contrastive_loss_fuse_self.item():.4f}"
              f"| reconstruction={loss_reconstruction.item():.4f}"
              f"| cpl_dir={loss_cpl_dir.item():.4f}"
              f"| cpl_ratio={loss_cpl_ratio.item():.4f}"
              # f"| coords_reconstruction={coord_reconstruction_loss.item():.4f}"
              f"| total={loss.item():.4f}")

        # 调用处（计算好 z 后）
        out_dir = f"./new_gat_cross_attn_res_slice73-76_loss_cpl/epoch_{epoch}"
        save_epoch_z_and_rgb_global_to_img_range(
            res_z_tensor=r_base["z"],
            out_dict=out,
            epoch=epoch,
            out_dir=out_dir,
            # rgb_min=np.array([0, 0, 0], dtype=np.float32),
            # rgb_max=np.array([255, 255, 255], dtype=np.float32),
            rgb_min=rgb_min,
            rgb_max=rgb_max,
            point_size=25,
        )

        # ====== 聚类可视化（每个 epoch 末尾）======
        from sklearn.cluster import KMeans
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        from sklearn.metrics import adjusted_rand_score

        def _plot_clusters_points(coords: np.ndarray,
                                  labels: np.ndarray,
                                  save_path: str,
                                  palette: np.ndarray,
                                  point_size: float = 6.0,
                                  invert_y: bool = True,
                                  title: str = None):
            """
            coords: (Ns,2), labels: (Ns,), palette: (K,3) in 0~1
            """
            # 固定绘制顺序，避免覆盖伪影
            order = np.argsort(coords[:, 0] + coords[:, 1])
            coords_sorted = coords[order]
            labels_sorted = labels[order]
            colors_sorted = palette[labels_sorted]

            plt.figure(figsize=(6, 6), dpi=200)
            plt.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                        c=colors_sorted, s=point_size, marker='s', linewidth=0)
            plt.gca().set_aspect('equal', adjustable='box')
            if invert_y:
                plt.gca().invert_yaxis()
            if title:
                plt.title(title)
            plt.axis('off')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()


        # ------- 在训练循环内，每个 epoch 结束时插入 -------
        with torch.no_grad():
            # 1) 取本轮 h、坐标、切片
            # h_np = h.detach().cpu().numpy()  # (N, H_DIM)
            h_np = h_fuse.detach().cpu().numpy()  # (N, H_DIM)
            coords = out["coords_t"].detach().cpu().numpy()  # (N, 2)
            slice_id = out["slice_id_t"].detach().cpu().numpy()  # (N,)

            # 1) 进行 UMAP 降维
            reducer = umap.UMAP(n_components=2, n_jobs=20)  # 设置为 4 线程
            umap_coords = reducer.fit_transform(h_np)

            # # 2) 标准化 h（聚类更稳）
            # h_mean = h_np.mean(axis=0, keepdims=True)
            # h_std = h_np.std(axis=0, keepdims=True) + 1e-8
            # h_z = (h_np - h_mean) / h_std
            h_z = h_np

            k = 7


            import rpy2.robjects as robjects

            robjects.r.library("mclust")

            import rpy2.robjects.numpy2ri as numpy2ri

            r_random_seed = robjects.r['set.seed']
            r_random_seed(SEED)
            rmclust = robjects.r['Mclust']

            # Convert numpy ndarray to R object using numpy2ri.py2rpy
            r_data = numpy2ri.py2rpy(h_z)

            # Perform clustering using Mclust in R
            res = rmclust(r_data, k, 'EEE')
            mclust_res = np.array(res[-2])  # Get the clustering results

            labels = mclust_res.astype(np.int64)-1

            # # 3) 全局 k-means 聚 7 类（保证两片的标签一致）
            # k = 7
            # km = KMeans(n_clusters=k, n_init=10, random_state=42)
            # labels = km.fit_predict(h_z).astype(np.int64)  # (N,)

            # 4) 固定调色板（同一簇在两片上颜色一致）
            # 使用 Matplotlib 'tab10' 的前 7 色
            TAB10 = np.array([
                [0.1216, 0.4667, 0.7059],
                [1.0000, 0.4980, 0.0549],
                [0.1725, 0.6275, 0.1725],
                [0.8392, 0.1529, 0.1569],
                [0.5804, 0.4039, 0.7412],
                [0.5490, 0.3373, 0.2941],
                [0.8902, 0.4667, 0.7608],
            ], dtype=np.float32)  # shape: (7, 3), 值域 0~1

            # 5) 保存标签与可视化
            os.makedirs(out_dir, exist_ok=True)

            np.save(os.path.join(out_dir, f"h_epoch{epoch:03d}.npy"), h_np)
            np.save(os.path.join(out_dir, f"labels_k{k}_epoch{epoch:03d}.npy"), labels)
            # 4) 绘制 UMAP 图并保存
            save_path = os.path.join(out_dir, f"umap_all_k{k}_epoch{epoch:03d}.png")
            _plot_clusters_points(
                umap_coords, labels, save_path,
                palette=TAB10, point_size=10, invert_y=True,
                title=f"UMAP | k={k} | epoch {epoch}"
            )

            true_labels = out["Region_t"].detach().cpu().numpy()
            save_path_true = os.path.join(out_dir, f"umap_true_k{k}_epoch{epoch:03d}.png")
            _plot_clusters_points(
                umap_coords, true_labels, save_path_true,
                palette=TAB10, point_size=10, invert_y=True,
                title=f"UMAP | k={k} | epoch {epoch}"
            )

            # 另外，如果你想按切片批次绘制，可以这样：
            save_path_batch = os.path.join(out_dir, f"umap_batch_epoch{epoch:03d}.png")
            _plot_clusters_points(
                umap_coords, slice_id, save_path_batch,
                palette=TAB10, point_size=10, invert_y=True,
                title=f"UMAP Batch | epoch {epoch}"
            )

            print(f"UMAP and clustering results saved for epoch {epoch}.")

            # 计算每个切片的ARI值，并与历史最佳ARI值对比
            for sid in np.unique(slice_id):
                idx = (slice_id == sid)
                coords_s = coords[idx]
                labels_s = labels[idx]

                # 假设 Region 是预处理过程中已经加入的真实标签
                true_labels_s = out["Region_t"][idx].detach().cpu().numpy()  # 真实标签，假设已经包含在out["Region"]中

                # 计算ARI
                ari = adjusted_rand_score(true_labels_s, labels_s)

                # 获取当前切片的最佳ARI值和出现的epoch
                if sid in best_ari_per_slice:
                    best_ari, best_epoch = best_ari_per_slice[sid]
                else:
                    best_ari, best_epoch = -1.0, -1  # 初始值，表示还没有计算过

                # 如果当前的ARI更好，更新最好的ARI和对应的epoch
                if ari > best_ari:
                    best_ari_per_slice[sid] = (ari, epoch)
                    best_ari, best_epoch = ari, epoch

                # 打印每个切片的ARI和当前最好的ARI及对应的epoch
                print(f"Epoch {epoch}: Slice {int(sid)} ARI = {ari:.4f} | "
                      f"Best ARI = {best_ari:.4f} (Epoch {best_epoch})")


            # np.save(os.path.join(out_dir, f"h_fuse_epoch{epoch:03d}.npy"), h_np)
            # np.save(os.path.join(out_dir, f"labels_k{k}_epoch{epoch:03d}_h_fuse.npy"), labels)

            # 分片分别绘制，但颜色映射按全局 labels 一致
            for sid in np.unique(slice_id):
                idx = (slice_id == sid)
                coords_s = coords[idx]
                labels_s = labels[idx]
                save_path = os.path.join(out_dir, f"slice{int(sid)}_k{k}_epoch{epoch:03d}.png")
                # save_path = os.path.join(out_dir, f"slice{int(sid)}_k{k}_epoch{epoch:03d}_h_fuse.png")
                _plot_clusters_points(
                    coords_s, labels_s, save_path,
                    palette=TAB10, point_size=25, invert_y=True,
                    title=f"slice {int(sid)} | k={k} | epoch {epoch}"
                )




# def info_nce_loss_fuse_self(
#         h_fuse: torch.Tensor,  # (N, h_dim)
#         nbr_idx_t: torch.Tensor,  # (N, Kmax)  # 同切片邻居索引
#         nbr_mask: torch.Tensor,  # (N, Kmax)  # 同切片邻居有效掩码
#         temperature: float = 0.07,
#         eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     修改后的对比学习损失（`h_fuse` 和 `h_fuse`，邻居为正样本）
#     - 正样本为每个样本的邻居
#     - 负样本为其他样本，去除正样本和对角线
#     """
#     device = h_fuse.device
#     N = h_fuse.shape[0]
#
#     h_fuse_norm = h_fuse / (h_fuse.norm(dim=-1, keepdim=True) + eps)
#
#     sim_matrix = torch.mm(h_fuse_norm, h_fuse_norm.T)  # (N, N)
#
#     sim_matrix = torch.sigmoid(sim_matrix)
#
#     valid_nbr_mask = nbr_mask.bool()  # (N, Kmax), 转为布尔型
#     pos_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
#     pos_mask[torch.arange(N).unsqueeze(1), nbr_idx_t] = valid_nbr_mask
#
#     # 计算正样本的相似度
#     pos_sim = sim_matrix * pos_mask  # 只保留正样本的相似度
#
#     # 计算每个样本的正邻居数量（即 pos_mask 中每行的和）
#     pos_neighbors_count = pos_mask.sum(dim=1).float()
#     # 计算正样本的损失
#     weighted_pos_loss = -torch.log(pos_sim + eps) / pos_neighbors_count
#
#     # 计算负样本掩码：去掉正样本和对角线
#     neg_mask = ~pos_mask  # 首先排除正样本
#     neg_mask = neg_mask * (~torch.eye(N, device=device).bool())  # 排除对角线（即每个点自己）
#
#     # 计算负样本相似度
#     neg_sim = sim_matrix * neg_mask  # 只保留负样本的相似度
#
#     # 计算负样本的损失
#     neg_loss = -torch.log(1 - neg_sim + eps)
#
#     # 计算每个样本的负邻居数量（即 neg_mask 中每行的和）
#     neg_neighbors_count = neg_mask.sum(dim=1).float()
#
#     # 对负样本损失进行加权平均：每个样本的负样本损失除以它的负邻居数量
#     weighted_neg_loss = neg_loss.sum(dim=1) / neg_neighbors_count
#
#     # 最终损失：正负样本损失加和
#     total_loss = weighted_pos_loss.mean() + weighted_neg_loss.mean()
#
#     return total_loss
