import os
import sys
import glob
import math
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import matplotlib
import matplotlib.pyplot as plt

# ======================
# 中文字体 & 基本显示设置
# ======================
def setup_chinese_font():
    """
    自动设置中文字体；优先尝试：思源黑体、黑体、微软雅黑。
    并修复负号显示。
    """
    matplotlib.rcParams['axes.unicode_minus'] = False
    candidates = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "Source Han Sans SC"]
    # 尝试按顺序设置可用字体
    try:
        from matplotlib import font_manager
        installed = set(f.name for f in font_manager.fontManager.ttflist)
        for name in candidates:
            if name in installed:
                matplotlib.rcParams['font.family'] = name
                break
        else:
            # 若一个都没命中，也先不报错，仍按默认英文字体，但负号可用
            pass
    except Exception:
        pass

setup_chinese_font()

# ======================
# 工具函数
# ======================

def sanitize_argv():
    """清理 Notebook 注入的 -f <kernel.json> 等参数。"""
    keep_prefixes = (
        "--data_dir", "--out_dir", "--fs", "--include_source_like",
        "--assume_duration_s", "--plot_seconds", "--max_points_plot",
        "--zscore_norm", "--save_psd_npz", "--help"
    )
    argv = sys.argv[:]
    cleaned = [argv[0]]
    i, skip_next = 1, False
    while i < len(argv):
        a = argv[i]
        if skip_next:
            skip_next = False
            i += 1
            continue
        if a == "-f":
            skip_next = True
            i += 1
            continue
        if any(a.startswith(p) for p in keep_prefixes):
            cleaned.append(a)
            if ("=" not in a) and i + 1 < len(argv) and not argv[i+1].startswith("-"):
                cleaned.append(argv[i+1]); i += 1
        i += 1
    sys.argv = cleaned

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def find_signal_variables(mat_dict: Dict) -> List[str]:
    return [k for k, v in mat_dict.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)]

def squeeze_1d(x: np.ndarray) -> np.ndarray:
    x = np.array(x)
    if np.iscomplexobj(x):
        x = np.real(x)
    x = np.squeeze(x)
    if x.ndim == 1: return x
    if x.ndim == 2:
        if 1 in x.shape and max(x.shape) > 1: return x.reshape(-1)
        return (x[:, 0] if x.shape[0] >= x.shape[1] else x[0, :]).reshape(-1)
    return x.reshape(-1)

def infer_fs_from_duration(sig_len: int, assume_duration_s: Optional[float]) -> Optional[float]:
    if not assume_duration_s or assume_duration_s <= 0 or sig_len <= 0:
        return None
    return float(sig_len) / float(assume_duration_s)

def compute_basic_stats(sig: np.ndarray) -> Dict[str, float]:
    if sig.size == 0:
        return dict(均值=np.nan, 标准差=np.nan, RMS=np.nan, 峭度=np.nan,
                    偏度=np.nan, 峰值=np.nan, 峰值因子=np.nan)
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    krt = float(kurtosis(sig, fisher=True, bias=False))
    skw = float(skew(sig, bias=False))
    peak = float(np.max(np.abs(sig)))
    crest = float(peak / rms) if rms > 0 else np.nan
    return dict(均值=mean, 标准差=std, RMS=rms, 峭度=krt, 偏度=skw, 峰值=peak, 峰值因子=crest)

def maybe_downsample(sig: np.ndarray, fs: Optional[float], max_points: Optional[int]) -> Tuple[np.ndarray, Optional[float]]:
    if not max_points or max_points <= 0 or sig.size <= max_points:
        return sig, fs
    step = int(math.ceil(sig.size / max_points))
    sig_ds = sig[::step]
    fs_ds = (fs / step) if (fs and fs > 0) else fs
    return sig_ds, fs_ds

# ======================
# 读取 .mat（v7 & v7.3）
# ======================

def load_mat_auto(file_path: str) -> Dict[str, np.ndarray]:
    try:
        md = sio.loadmat(file_path)
        return {k: v for k, v in md.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)}
    except Exception:
        pass
    try:
        import h5py
        out = {}
        with h5py.File(file_path, "r") as f:
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = np.array(obj)
                        if np.issubdtype(arr.dtype, np.number):
                            out[name.split("/")[-1]] = arr
                    except Exception:
                        pass
            f.visititems(_collect)
        return out
    except Exception as e:
        raise RuntimeError(f"无法读取 MAT（scipy/h5py 均失败）：{e}")

def read_target_mat(file_path: str) -> Dict[str, np.ndarray]:
    md = load_mat_auto(file_path)
    return {k: squeeze_1d(md[k]) for k in find_signal_variables(md)}

def read_source_mat(file_path: str) -> Dict[str, np.ndarray]:
    md = load_mat_auto(file_path)
    out = {}
    for k, v in md.items():
        ks = k.lower()
        if "de" in ks and "time" in ks: out["DE"] = squeeze_1d(v)
        elif "fe" in ks and "time" in ks: out["FE"] = squeeze_1d(v)
        elif "ba" in ks and "time" in ks: out["BA"] = squeeze_1d(v)
        elif "rpm" in ks: out["RPM"] = squeeze_1d(v)
    if not out:
        for k in find_signal_variables(md):
            out[k] = squeeze_1d(md[k])
    return out

# ======================
# 单文件作图（保存到磁盘，中文标题/坐标）
# ======================

def save_time_series(sig: np.ndarray, fs: Optional[float], title: str, out_png: str):
    plt.figure()
    if fs and fs > 0:
        t = np.arange(len(sig)) / fs
        plt.plot(t, sig); plt.xlabel("时间 (秒)")
    else:
        plt.plot(sig);     plt.xlabel("样本索引")
    plt.ylabel("幅值"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def save_psd(sig: np.ndarray, fs: Optional[float], title: str, out_png: str, return_arrays: bool=False):
    plt.figure()
    nperseg = min(len(sig), 4096);  nperseg = nperseg if nperseg >= 8 else len(sig)
    if fs and fs > 0:
        f, Pxx = welch(sig, fs=fs, nperseg=nperseg); plt.xlabel("频率 (Hz)")
    else:
        f, Pxx = welch(sig, fs=1.0, nperseg=nperseg); plt.xlabel("归一化频率 (cycles/sample)")
    plt.semilogy(f, Pxx); plt.ylabel("PSD"); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
    if return_arrays:
        return f, Pxx
    return None

def save_spectrogram(sig: np.ndarray, fs: Optional[float], title: str, out_png: str):
    plt.figure()
    fs_used = fs if (fs and fs > 0) else 1.0
    if len(sig) >= 4096:   NFFT = 1024
    elif len(sig) >= 2048: NFFT = 512
    else:                  NFFT = 256
    noverlap = NFFT // 2
    plt.specgram(sig, NFFT=NFFT, Fs=fs_used, noverlap=noverlap)
    plt.xlabel("时间 (秒)" if (fs and fs > 0) else "时间 (样本)"); plt.ylabel("频率 (Hz)" if (fs and fs > 0) else "频率 (cycles/sample)")
    plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

# ======================
# 综合性图（保存 + 显示）
# ======================

def show_and_save_features_scatter(df: pd.DataFrame, out_png_prefix: str):
    # 图1：峭度 vs 峰值因子
    plt.figure()
    ok = df[['峭度','峰值因子']].dropna()
    plt.scatter(ok['峭度'], ok['峰值因子'])
    plt.xlabel("峭度"); plt.ylabel("峰值因子"); plt.title("综合图：峭度 vs 峰值因子")
    plt.tight_layout(); plt.savefig(out_png_prefix + "_kurtosis_crest.png", dpi=150); plt.show()

    # 图2：RMS vs 峰值
    plt.figure()
    ok = df[['RMS','峰值']].dropna()
    plt.scatter(ok['RMS'], ok['峰值'])
    plt.xlabel("RMS"); plt.ylabel("峰值"); plt.title("综合图：RMS vs 峰值")
    plt.tight_layout(); plt.savefig(out_png_prefix + "_rms_peak.png", dpi=150); plt.show()

def show_and_save_psd_overlay(psd_bank: List[Tuple[str, np.ndarray, np.ndarray]], out_png: str, max_curves: int=12):
    """
    多文件 PSD 叠加对比（前 max_curves 条），用于宏观观察差异。
    """
    if len(psd_bank) == 0: return
    plt.figure()
    for i, (label, f, Pxx) in enumerate(psd_bank[:max_curves]):
        plt.semilogy(f, Pxx, label=label)
    plt.xlabel("频率 (Hz)"); plt.ylabel("PSD"); plt.title("综合图：多文件 PSD 叠加对比（前若干条）")
    # 不强制显示图例（避免中文字体不全触发问题）；如需可手动 plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show()

def show_and_save_psd_heatmap(psd_bank: List[Tuple[str, np.ndarray, np.ndarray]], out_png: str, freq_max: Optional[float]=None, n_bins: int=256):
    """
    将各文件 PSD 投到统一频率轴，堆叠为热力图（文件×频率）。
    """
    if len(psd_bank) == 0: return
    # 统一频率轴
    f_common = psd_bank[0][1]
    if freq_max is not None:
        mask = f_common <= freq_max
        f_common = f_common[mask]
    X = []
    for _, f, Pxx in psd_bank:
        if freq_max is not None:
            Pxx = Pxx[f <= freq_max]
            f_use = f[f <= freq_max]
        else:
            f_use = f
        # 简单插值到 f_common
        if len(f_use) != len(f_common) or np.any(f_use != f_common):
            Pxx_i = np.interp(f_common, f_use, Pxx)
        else:
            Pxx_i = Pxx
        # 对数尺度（防止量级差异过大）
        X.append(np.log10(Pxx_i + 1e-12))
    X = np.vstack(X)  # shape: [num_files, num_freq]

    # 可选再次降维到 n_bins（频率向量太长时）
    if X.shape[1] > n_bins:
        idx = np.linspace(0, X.shape[1]-1, n_bins).astype(int)
        X = X[:, idx]
        f_disp = f_common[idx]
    else:
        f_disp = f_common

    plt.figure()
    plt.imshow(X, aspect='auto', origin='lower', extent=[f_disp[0], f_disp[-1], 0, X.shape[0]])
    plt.xlabel("频率 (Hz)"); plt.ylabel("文件索引"); plt.title("综合图：PSD 热力图（文件 × 频率）")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show()

def show_and_save_pca(df: pd.DataFrame, out_png: str):
    """
    用 numpy 做一个简易 PCA（特征标准化后 SVD），展示二维散点。
    """
    feat_cols = ["均值","标准差","RMS","峭度","偏度","峰值","峰值因子"]
    data = df[feat_cols].values.astype(float)
    mask = np.all(np.isfinite(data), axis=1)
    X = data[mask]
    if X.shape[0] < 2:
        return
    # 标准化
    Xs = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    # SVD
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    Z = Xs @ Vt[:2].T  # 投影到前两主成分
    plt.figure()
    plt.scatter(Z[:,0], Z[:,1])
    plt.xlabel("主成分1"); plt.ylabel("主成分2"); plt.title("综合图：统计特征 PCA 二维散点")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show()

# ======================
# 主流程
# ======================

def process_all(
    data_dir: str = "./目标域数据集",
    out_dir: str = "./目标域输出",
    fs: Optional[float] = 32000.0,
    include_source_like: bool = False,
    assume_duration_s: Optional[float] = 8.0,
    plot_seconds: Optional[float] = 3.0,
    max_points_plot: Optional[int] = 200000,
    zscore_norm: bool = False,
    save_psd_npz: bool = True
):
    """
    - 单文件图：保存到 out_dir/plots（中文标题/坐标）
    - 统计参数：保存 out_dir/summary.csv + out_dir/meta/*.json
    - PSD 数值：保存 out_dir/npz/*.npz（可关）
    - 综合性图：保存到 out_dir/aggregates，并显示
    """
    # 目录
    plots_dir = os.path.join(out_dir, "plots");        safe_mkdir(plots_dir)
    meta_dir  = os.path.join(out_dir, "meta");         safe_mkdir(meta_dir)
    npz_dir   = os.path.join(out_dir, "npz");          safe_mkdir(npz_dir)
    aggr_dir  = os.path.join(out_dir, "aggregates");   safe_mkdir(aggr_dir)

    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        print(f"[WARN] 未在 {os.path.abspath(data_dir)} 找到 .mat 文件"); return

    print(f"[INFO] 发现 {len(mat_files)} 个 .mat；读取目录：{os.path.abspath(data_dir)}")
    rows = []
    psd_bank = []  # for aggregates: [(label, f, Pxx), ...]

    for fp in mat_files:
        fname = os.path.basename(fp); stem = os.path.splitext(fname)[0]
        try:
            data_map = read_source_mat(fp) if include_source_like else read_target_mat(fp)
            if not data_map:
                rows.append({"文件": fname, "变量": "", "长度": 0,
                             "使用采样率": np.nan, "时长(秒)": np.nan, "备注": "EMPTY"})
                print(f"[WARN] {fname} 未发现有效数值变量"); continue

            for var, raw_sig in data_map.items():
                if var.upper() == "RPM":
                    rpm_val = float(np.median(raw_sig)) if raw_sig.size else np.nan
                    meta = {"文件": fname, "变量": var, "类型": "RPM", "中位数": rpm_val}
                    with open(os.path.join(meta_dir, f"{stem}_{var}.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    rows.append({"文件": fname, "变量": var, "长度": int(raw_sig.size),
                                 "使用采样率": np.nan, "时长(秒)": np.nan,
                                 "均值": np.nan, "标准差": np.nan, "RMS": np.nan, "峭度": np.nan,
                                 "偏度": np.nan, "峰值": np.nan, "峰值因子": np.nan,
                                 "备注": "RPM通道"})
                    continue

                sig = raw_sig.astype(float)

                # 采样率推断与截取
                fs_used = fs if (fs and fs > 0) else infer_fs_from_duration(len(sig), assume_duration_s)
                if plot_seconds and (plot_seconds > 0) and fs_used and fs_used > 0:
                    sig = sig[: int(min(len(sig), plot_seconds * fs_used))]

                # 统计（基于原始片段）；绘图可 z-score
                stats_input = sig.copy()
                if zscore_norm and sig.size > 1:
                    m, s = float(np.mean(sig)), float(np.std(sig))
                    if s > 0: sig = (sig - m) / s

                stats = compute_basic_stats(stats_input)
                duration_s = (len(sig) / fs_used) if (fs_used and fs_used > 0) else np.nan

                # 绘图降采样
                sig_plot, fs_plot = maybe_downsample(sig, fs_used, max_points_plot)

                # —— 保存单文件三图（中文）——
                base = f"{stem}_{var}"
                save_time_series(sig_plot, fs_plot, f"{fname} - {var} 时域波形",
                                 os.path.join(plots_dir, f"{base}_时域.png"))
                fP = save_psd(sig_plot, fs_plot, f"{fname} - {var} PSD 功率谱密度",
                              os.path.join(plots_dir, f"{base}_PSD.png"), return_arrays=True)
                save_spectrogram(sig_plot, fs_plot, f"{fname} - {var} 频谱图（Spectrogram）",
                                 os.path.join(plots_dir, f"{base}_谱图.png"))

                # —— 保存单条 meta（JSON）——
                meta = {
                    "文件": fname, "变量": var,
                    "使用采样率": float(fs_used) if (fs_used and fs_used > 0) else None,
                    "时长(秒)": float(duration_s) if not np.isnan(duration_s) else None,
                    **stats
                }
                with open(os.path.join(meta_dir, f"{stem}_{var}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # —— 保存 PSD 数值（可选）——
                if save_psd_npz and fP is not None:
                    f_arr, Pxx_arr = fP
                    np.savez(os.path.join(npz_dir, f"{stem}_{var}_psd.npz"), f=f_arr, Pxx=Pxx_arr)

                # —— 行汇总（供 summary.csv & 综合图）——
                rows.append({
                    "文件": fname, "变量": var, "长度": int(len(stats_input)),
                    "使用采样率": float(fs_used) if (fs_used and fs_used > 0) else np.nan,
                    "时长(秒)": float(duration_s) if not np.isnan(duration_s) else np.nan,
                    **stats, "备注": ""
                })

                # —— 供综合 PSD 叠加/热力图 ——（仅选取首个变量或全部？这里收集全部变量）
                if fP is not None and fs_plot and fs_plot > 0:
                    label = f"{stem}-{var}"
                    psd_bank.append((label, fP[0], fP[1]))

            print(f"[OK] 处理完成：{fname}")

        except Exception as e:
            rows.append({
                "文件": fname, "变量": "", "长度": 0,
                "使用采样率": np.nan, "时长(秒)": np.nan,
                "均值": np.nan, "标准差": np.nan, "RMS": np.nan, "峭度": np.nan,
                "偏度": np.nan, "峰值": np.nan, "峰值因子": np.nan,
                "备注": f"ERROR: {e}"
            })
            print(f"[ERROR] 处理 {fname} 失败：{e}")

    # —— 汇总表 —— 
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[SUMMARY] 汇总：{csv_path}")
    print(f"[SUMMARY] 单文件图片目录：{os.path.join(out_dir, 'plots')}")
    print(f"[SUMMARY] 元数据目录：{os.path.join(out_dir, 'meta')}")
    if save_psd_npz:
        print(f"[SUMMARY] PSD 数值目录：{os.path.join(out_dir, 'npz')}")

    # —— 综合性大图：保存 + 显示 —— 
    show_and_save_features_scatter(df, os.path.join(aggr_dir, "features_scatter"))
    show_and_save_psd_overlay(psd_bank, os.path.join(aggr_dir, "psd_overlay.png"), max_curves=12)
    # 只展示到 Nyquist 的一部分，如 0~8kHz（若需要可改为 None）
    show_and_save_psd_heatmap(psd_bank, os.path.join(aggr_dir, "psd_heatmap.png"), freq_max=None, n_bins=256)
    show_and_save_pca(df, os.path.join(aggr_dir, "pca_scatter.png"))

    print(f"[SUMMARY] 综合性图目录：{aggr_dir}（同时已在屏幕显示）")


# ======================
# 简易 CLI（--key=value），Notebook 也可直接 %run
# ======================
def main_cli():
    sanitize_argv()
    args = {
        "data_dir": "./目标域数据集",
        "out_dir": "./目标域输出",
        "fs": "32000",
        "include_source_like": "false",
        "assume_duration_s": "8",
        "plot_seconds": "3",
        "max_points_plot": "200000",
        "zscore_norm": "false",
        "save_psd_npz": "true",
    }
    for a in sys.argv[1:]:
        if not a.startswith("--"): continue
        if "=" in a:
            k, v = a[2:].split("=", 1); args[k] = v
        else:
            args[a[2:]] = "true"

    def to_bool(s: str) -> bool:
        return str(s).strip().lower() in ("1", "true", "yes", "y", "on")

    data_dir = args.get("data_dir", "./目标域数据集")
    out_dir  = args.get("out_dir", "./目标域输出")

    try:
        fs = float(args.get("fs", "32000")); fs = None if fs <= 0 else fs
    except Exception:
        fs = 32000.0

    include_source_like = to_bool(args.get("include_source_like", "false"))

    try:
        assume_duration_s = float(args.get("assume_duration_s", "8")); 
        assume_duration_s = None if assume_duration_s <= 0 else assume_duration_s
    except Exception:
        assume_duration_s = 8.0

    try:
        plot_seconds = float(args.get("plot_seconds", "3")); 
        plot_seconds = None if plot_seconds <= 0 else plot_seconds
    except Exception:
        plot_seconds = 3.0

    try:
        max_points_plot = int(args.get("max_points_plot", "200000"))
        max_points_plot = None if max_points_plot <= 0 else max_points_plot
    except Exception:
        max_points_plot = 200000

    zscore_norm = to_bool(args.get("zscore_norm", "false"))
    save_psd_npz = to_bool(args.get("save_psd_npz", "true"))

    process_all(
        data_dir=data_dir,
        out_dir=out_dir,
        fs=fs,
        include_source_like=include_source_like,
        assume_duration_s=assume_duration_s,
        plot_seconds=plot_seconds,
        max_points_plot=max_points_plot,
        zscore_norm=zscore_norm,
        save_psd_npz=save_psd_npz
    )

if __name__ == "__main__":
    main_cli()
