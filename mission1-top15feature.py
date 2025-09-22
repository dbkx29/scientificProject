import os, re, glob, math, sys, argparse, itertools
import numpy as np
import scipy.io as sio
from scipy.signal import welch, stft
from scipy.stats import kurtosis, skew, entropy, f_oneway
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd

# ================== 中文 & 基础 ==================
def setup_chinese():
    matplotlib.rcParams['axes.unicode_minus'] = False
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in ["Noto Sans CJK SC","SimHei","Microsoft YaHei","Source Han Sans SC"]:
            if name in installed:
                matplotlib.rcParams['font.family'] = name
                break
    except Exception:
        pass
setup_chinese()

def sanitize_argv():
    keep = ("--data_root","--out_dir","--fs","--win_sec","--overlap","--zscore",
            "--bands","--tsne","--topk","--seed","--help")
    argv = sys.argv[:]; out=[argv[0]]; i=1; skip=False
    while i < len(argv):
        a = argv[i]
        if skip: skip=False; i+=1; continue
        if a == "-f": skip=True; i+=1; continue
        if any(a.startswith(k) for k in keep):
            out.append(a)
            if ("=" not in a) and (i+1<len(argv)) and (not argv[i+1].startswith("-")):
                out.append(argv[i+1]); i+=1
        i+=1
    sys.argv = out

def set_seed(seed=42):
    np.random.seed(seed)

# ================== I/O 与数据管线 ==================
FN_PAT = re.compile(r'^(?P<cls>IR|OR|B|N)', re.IGNORECASE)

def load_mat(fp):
    md = sio.loadmat(fp)
    return {k:v for k,v in md.items() if not k.startswith("__") and isinstance(v, np.ndarray)}

def squeeze1d(x):
    x = np.array(x).squeeze()
    if np.iscomplexobj(x): x = np.real(x)
    if x.ndim==1: return x
    if x.ndim==2 and 1 in x.shape: return x.reshape(-1)
    return x.flatten()

def choose_signal(md):
    for k,v in md.items():
        kl = k.lower()
        if "de" in kl and "time" in kl: return "DE", squeeze1d(v)
        if "fe" in kl and "time" in kl: return "FE", squeeze1d(v)
        if "ba" in kl and "time" in kl: return "BA", squeeze1d(v)
    k0=list(md.keys())[0]
    return k0, squeeze1d(md[k0])

def parse_cls_from_name(fname: str):
    m = FN_PAT.match(os.path.basename(fname))
    if m: return m.group("cls").upper()
    if "normal" in fname.lower(): return "N"
    return "UNK"

def sliding_windows(x, win, hop):
    if x.size < win: return np.empty((0,win), dtype=np.float32)
    n = 1 + (x.size - win)//hop
    out = np.zeros((n,win), dtype=np.float32)
    for i in range(n): out[i,:] = x[i*hop:i*hop+win]
    return out

def zscore_norm(X):
    m = X.mean(axis=1, keepdims=True); s = X.std(axis=1, keepdims=True) + 1e-12
    return (X - m) / s

# ================== 特征工程 ==================
def spectral_features(x, fs):
    """返回：谱质心、谱带宽、谱滚降95%、主峰频率、谱熵"""
    nperseg = min(len(x), 4096); nperseg = nperseg if nperseg>=8 else len(x)
    f, Pxx = welch(x, fs=fs if fs>0 else 1.0, nperseg=nperseg)
    P = Pxx / (np.sum(Pxx) + 1e-12)
    # 质心
    sc = np.sum(f * P)
    # 带宽（二阶矩）
    bw = np.sqrt(np.sum(((f - sc) ** 2) * P))
    # 滚降95%
    cdf = np.cumsum(P)
    roll95 = f[np.searchsorted(cdf, 0.95)]
    # 主峰频率
    pfreq = f[np.argmax(Pxx)]
    # 谱熵
    sent = entropy(P + 1e-12, base=2)
    return dict(spec_centroid=sc, spec_bw=bw, spec_rolloff95=roll95, peak_freq=pfreq, spec_entropy=sent)

def band_energies(x, fs, bands_hz):
    """返回每个频带的能量占比（使用 Welch）"""
    nperseg = min(len(x), 4096); nperseg = nperseg if nperseg>=8 else len(x)
    f, Pxx = welch(x, fs=fs if fs>0 else 1.0, nperseg=nperseg)
    total = np.trapz(Pxx, f) + 1e-12
    out = {}
    for (lo, hi) in bands_hz:
        lo_ = max(lo, 0.0); hi_ = min(hi, fs/2 if fs>0 else 0.5)
        mask = (f >= lo_) & (f < hi_)
        if not np.any(mask):
            out[f"band_{int(lo)}_{int(hi)}"] = 0.0
        else:
            e = np.trapz(Pxx[mask], f[mask]) / total
            out[f"band_{int(lo)}_{int(hi)}"] = float(e)
    return out

def stft_texture(x, fs):
    """简单的时频纹理特征：STFT 幅度图的统计"""
    fs_used = fs if fs>0 else 1.0
    nperseg = 1024 if len(x) >= 2048 else 256
    noverlap = nperseg // 2
    f, t, Z = stft(x, fs=fs_used, nperseg=nperseg, noverlap=noverlap)
    A = np.abs(Z)
    # 统计量
    return dict(
        stft_mean = float(np.mean(A)),
        stft_std  = float(np.std(A)),
        stft_skew = float(skew(A.reshape(-1), bias=False)),
        stft_kurt = float(kurtosis(A.reshape(-1), fisher=True, bias=False))
    )

def time_features(x):
    """时域统计 & 形态学"""
    x = x.astype(np.float64)
    mean = float(np.mean(x))
    std  = float(np.std(x))
    rms  = float(np.sqrt(np.mean(x**2)))
    krt  = float(kurtosis(x, fisher=True, bias=False))
    skw  = float(skew(x, bias=False))
    peak = float(np.max(np.abs(x)))
    crest= float(peak / (rms + 1e-12))
    zcr  = float(np.mean(np.abs(np.diff(np.sign(x))))/2.0)  # 零交叉率（近似）
    pp_amp = float(np.max(x) - np.min(x))                   # 峰-峰值
    impulse = float(peak / (np.mean(np.abs(x)) + 1e-12))    # 脉冲因子
    shape   = float(rms / (np.mean(np.abs(x)) + 1e-12))     # 形状因子
    return dict(mean=mean, std=std, rms=rms, kurtosis=krt, skew=skw,
                peak=peak, crest_factor=crest, zcr=zcr, pp_amp=pp_amp,
                impulse_factor=impulse, shape_factor=shape)

def extract_features_of_segment(seg, fs, bands_hz):
    feats = {}
    feats.update(time_features(seg))
    feats.update(spectral_features(seg, fs))
    feats.update(band_energies(seg, fs, bands_hz))
    feats.update(stft_texture(seg, fs))
    return feats

# ================== 数据→特征主流程 ==================
def build_feature_table(
    data_root: str,
    fs: int = 12000,
    win_sec: float = 1.0,
    overlap: float = 0.5,
    zscore: bool = True,
    bands_hz = None
):
    if bands_hz is None:
        # 默认 1 kHz 步进到 Nyquist
        nyq = fs/2
        edges = list(range(0, int(nyq)+1000, 1000))
        bands_hz = list(zip(edges[:-1], edges[1:]))
        if not bands_hz: bands_hz = [(0, int(nyq))]

    mats = sorted(glob.glob(os.path.join(data_root, "**", "*.mat"), recursive=True))
    assert mats, f"未在 {data_root} 找到 .mat 文件"

    rows = []
    for fp in mats:
        md = load_mat(fp)
        _, sig = choose_signal(md)
        cls = parse_cls_from_name(fp)
        if cls == "UNK": 
            continue
        win = int(win_sec*fs); hop = max(1, int(win*(1-overlap)))
        segs = sliding_windows(sig, win, hop)
        if segs.shape[0]==0: 
            continue
        if zscore:
            segs = zscore_norm(segs)

        for i in range(segs.shape[0]):
            feats = extract_features_of_segment(segs[i], fs, bands_hz)
            feats.update(dict(file=os.path.relpath(fp, data_root).replace("\\","/"),
                              seg_idx=i, cls=cls))
            rows.append(feats)

    df = pd.DataFrame(rows)
    # 合理列排序：标签元信息在前
    cols_front = ["file","seg_idx","cls"]
    other = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + sorted(other)]
    return df

# ================== 统计与可视化 ==================
def save_tables(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    seg_csv = os.path.join(out_dir, "features_segments.csv")
    df.to_csv(seg_csv, index=False, encoding="utf-8-sig")

    # 按类别统计均值与标准差
    grp = df.groupby("cls")
    mean_tbl = grp.mean(numeric_only=True).add_suffix("_mean")
    std_tbl  = grp.std(numeric_only=True).add_suffix("_std")
    agg = pd.concat([mean_tbl, std_tbl], axis=1).reset_index()
    agg_csv = os.path.join(out_dir, "features_class_agg.csv")
    agg.to_csv(agg_csv, index=False, encoding="utf-8-sig")
    print(f"[导出] 片段特征：{seg_csv}")
    print(f"[导出] 类别汇总：{agg_csv}")
    return agg

def plot_feature_box_by_class(df: pd.DataFrame, features, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(df["cls"].unique().tolist())
    for feat in features:
        if feat not in df.columns: 
            continue
        plt.figure()
        data = [df.loc[df["cls"]==c, feat].values for c in classes]
        plt.boxplot(data, labels=classes, showfliers=False)
        plt.title(f"{feat} 按类别箱线图"); plt.xlabel("类别"); plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{feat}.png"), dpi=150)
        plt.show()

def plot_corr_heatmap(df: pd.DataFrame, out_path: str):
    num_df = df.select_dtypes(include=[np.number])
    C = num_df.corr().values
    plt.figure()
    plt.imshow(C, aspect='auto')
    plt.title("特征相关性热力图")
    plt.xlabel("特征索引"); plt.ylabel("特征索引")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

def plot_tsne(df: pd.DataFrame, out_path: str, max_n=4000, seed=42):
    # 只取数值列
    X = df.select_dtypes(include=[np.number]).values
    y = df["cls"].values
    if len(X) > max_n:
        idx = np.random.RandomState(seed).choice(len(X), size=max_n, replace=False)
        X = X[idx]; y = y[idx]
    Xs = StandardScaler().fit_transform(X)
    Z = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random", n_iter=1000, verbose=0, random_state=seed).fit_transform(Xs)
    classes = sorted(np.unique(y).tolist())
    plt.figure()
    for c in classes:
        Zi = Z[y==c]
        if Zi.size==0: continue
        plt.scatter(Zi[:,0], Zi[:,1], label=c, s=10)
    plt.title("t-SNE（特征空间分布）"); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()

def plot_avg_psd_by_class(df: pd.DataFrame, fs: int, out_path: str, nperseg=4096):
    # 由于 df 只有统计特征，这里再从 df 的均值频带能量复原趋势 → 简化为绘制“频带能量条形图”
    band_cols = [c for c in df.columns if c.startswith("band_")]
    if not band_cols: 
        print("[提示] 未找到频带能量特征列（band_*），跳过 PSD 类图。"); return
    classes = sorted(df["cls"].unique().tolist())
    # 画每个类别一张雷达更直观，这里先画整体平均条形图
    m = df.groupby("cls")[band_cols].mean()
    # 按频带顺序排序
    order = sorted(band_cols, key=lambda s: int(s.split("_")[1]))
    m = m[order]
    # 每类一张雷达图
    for c in classes:
        vals = m.loc[c].values
        labels = order
        ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        vals = np.r_[vals, vals[0]]
        ang  = np.r_[ang,  ang[0]]
        plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.plot(ang, vals)
        ax.fill(ang, vals, alpha=0.1)
        ax.set_thetagrids(angles=ang[:-1]*180/np.pi, labels=labels)
        ax.set_title(f"类别 {c} 频带能量雷达图")
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(out_path), f"radar_bands_{c}.png"), dpi=150); plt.show()

    # 也导出一个整体条形图（每个频带的全局均值）
    g = df[band_cols].mean().values
    plt.figure()
    plt.bar(range(len(order)), g)
    plt.xticks(range(len(order)), order, rotation=45)
    plt.ylabel("能量占比"); plt.title("全局平均频带能量分布")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()

def compute_feature_importance_variance(df: pd.DataFrame, k=15):
    """用类别均值方差 + ANOVA 简单估算可分性，返回 Top-K 特征名与分数"""
    num_cols = [c for c in df.columns if c not in ("file","seg_idx","cls")]
    # 1) 按类别均值的方差
    g = df.groupby("cls")[num_cols].mean()
    var_across_cls = g.var(axis=0)
    score1 = var_across_cls / (df[num_cols].var(axis=0) + 1e-12)

    # 2) ANOVA F-score
    y = df["cls"].values
    groups = [df.loc[y==c, num_cols].values for c in sorted(df["cls"].unique())]
    # 对每列做单因素方差分析（简化实现）
    f_scores = []
    for j,_ in enumerate(num_cols):
        col_groups = [gi[:,j] for gi in groups if gi.shape[0]>1]
        if len(col_groups) <= 1:
            f_scores.append(0.0)
        else:
            try:
                F, p = f_oneway(*col_groups)
                f_scores.append(float(F))
            except Exception:
                f_scores.append(0.0)
    score2 = pd.Series(f_scores, index=num_cols)

    # 综合分数（归一到 0-1 后求和）
    s1 = (score1 - score1.min()) / (score1.max() - score1.min() + 1e-12)
    s2 = (score2 - score2.min()) / (score2.max() - score2.min() + 1e-12)
    score = s1 + s2
    topk = score.sort_values(ascending=False).head(k)
    return topk

def plot_topk_importance(df: pd.DataFrame, k: int, out_path: str):
    topk = compute_feature_importance_variance(df, k=k)
    plt.figure()
    idx = np.arange(len(topk))
    plt.bar(idx, topk.values)
    plt.xticks(idx, topk.index, rotation=45)
    plt.ylabel("重要性评分（方差+ANOVA）")
    plt.title(f"Top-{k} 重要特征")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()
    return topk

# ================== 一键主函数 ==================
def run_feature_analysis(
    data_root: str,
    out_dir: str = "./q1_feat_out",
    fs: int = 12000,
    win_sec: float = 1.0,
    overlap: float = 0.5,
    zscore: bool = True,
    bands: str = "",     # 形如 "0-1000,1000-2000,2000-4000"
    tsne_plot: bool = True,
    topk: int = 15,
    seed: int = 42
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figs"); os.makedirs(fig_dir, exist_ok=True)

    # 解析频带
    if bands:
        bands_hz = []
        for token in bands.split(","):
            lo,hi = token.strip().split("-")
            bands_hz.append((float(lo), float(hi)))
    else:
        # 默认每 1kHz 一个带
        nyq = fs/2
        edges = list(range(0, int(nyq)+1000, 1000))
        bands_hz = list(zip(edges[:-1], edges[1:]))
        if not bands_hz: bands_hz = [(0, int(nyq))]

    print(f"[信息] 数据根目录: {data_root}")
    print(f"[信息] fs={fs}, win_sec={win_sec}, overlap={overlap}, zscore={zscore}")
    print(f"[信息] 频带: {bands_hz}")

    # 1) 提取特征
    df = build_feature_table(
        data_root=data_root, fs=fs, win_sec=win_sec, overlap=overlap, zscore=zscore, bands_hz=bands_hz
    )

    # 2) 保存表格（片段级 & 类别聚合）
    agg = save_tables(df, out_dir)

    # 3) 可视化
    # 3.1 核心特征箱线图
    core_feats = [
        "rms","kurtosis","skew","crest_factor","pp_amp",
        "spec_centroid","spec_bw","spec_rolloff95","peak_freq","spec_entropy"
    ]
    plot_feature_box_by_class(df, core_feats, os.path.join(fig_dir, "box"))

    # 3.2 相关性热力图（片段级数值特征）
    plot_corr_heatmap(df, os.path.join(fig_dir, "corr_heatmap.png"))

    # 3.3 t-SNE（标准化后）
    if tsne_plot:
        plot_tsne(df, os.path.join(fig_dir, "tsne_features.png"), max_n=4000)

    # 3.4 平均频带能量（雷达 + 条形）
    plot_avg_psd_by_class(df, fs, os.path.join(fig_dir, "bands_global_bar.png"))

    # 3.5 Top-K 重要特征
    topk_series = plot_topk_importance(df, k=topk, out_path=os.path.join(fig_dir, f"top{topk}_importance.png"))

    # 4) 导出一个“特征清单说明”
    with open(os.path.join(out_dir, "feature_definitions.txt"), "w", encoding="utf-8") as f:
        f.write(
            "【时域】\n"
            "- rms：均方根\n- kurtosis：峭度（Fisher）\n- skew：偏度\n- crest_factor：峰值/均方根\n"
            "- pp_amp：峰-峰值\n- impulse_factor：峰值/平均幅值\n- shape_factor：均方根/平均幅值\n- zcr：零交叉率\n\n"
            "【频域（Welch）】\n"
            "- spec_centroid：谱质心\n- spec_bw：谱带宽（围绕质心的二阶矩）\n- spec_rolloff95：95%能量滚降点频率\n"
            "- peak_freq：主峰频率\n- spec_entropy：谱熵\n\n"
            "【频带能量】\n"
            "- band_{lo}_{hi}：频带[lo,hi) Hz能量占比（对Pxx积分，再除以总能量）\n\n"
            "【时频】\n"
            "- stft_mean/std/skew/kurt：STFT幅度统计\n"
        )
    print(f"[完成] 可视化图片已保存到：{fig_dir}")
    return df, agg

# ================== CLI ==================
def build_args():
    p = argparse.ArgumentParser("第一问-特征分析（源域）", allow_abbrev=False)
    p.add_argument("--data_root", type=str, default="./源域数据集")
    p.add_argument("--out_dir",  type=str, default="./q1_feat_out")
    p.add_argument("--fs",       type=int, default=12000)
    p.add_argument("--win_sec",  type=float, default=1.0)
    p.add_argument("--overlap",  type=float, default=0.5)
    p.add_argument("--zscore",   action="store_true")
    p.add_argument("--bands",    type=str, default="", help='自定义频带，如 "0-800,800-1600,1600-3200"')
    p.add_argument("--tsne",     action="store_true")
    p.add_argument("--topk",     type=int, default=15)
    p.add_argument("--seed",     type=int, default=42)
    return p

if __name__ == "__main__":
    sanitize_argv()
    args = build_args().parse_args()
    run_feature_analysis(
        data_root=args.data_root, out_dir=args.out_dir, fs=args.fs,
        win_sec=args.win_sec, overlap=args.overlap, zscore=bool(args.zscore),
        bands=args.bands, tsne_plot=bool(args.tsne), topk=args.topk, seed=args.seed
    )
