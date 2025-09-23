import os, re, glob, sys, argparse
import numpy as np
import scipy.io as sio
from scipy.signal import welch, stft
from scipy.stats import kurtosis, skew, entropy, f_oneway
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd

# ================== 中文 & 风格 ==================
def setup_plot_style():
    matplotlib.rcParams['axes.unicode_minus'] = False
    sns.set(style="whitegrid", palette="muted")
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in ["Noto Sans CJK SC","SimHei","Microsoft YaHei","Source Han Sans SC"]:
            if name in installed:
                matplotlib.rcParams['font.family'] = name
                break
    except Exception:
        pass
setup_plot_style()

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

# ================== I/O ==================
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

def sliding_windows(signal, win_len, hop_len):
    if signal.size < win_len: return np.empty((0,win_len), dtype=np.float32)
    n = 1 + (signal.size - win_len)//hop_len
    out = np.zeros((n, win_len), dtype=np.float32)
    for i in range(n): out[i,:] = signal[i*hop_len:i*hop_len+win_len]
    return out

def zscore_norm(X):
    m = X.mean(axis=1, keepdims=True); s = X.std(axis=1, keepdims=True) + 1e-12
    return (X - m) / s

# ================== 特征工程 ==================
def spectral_features(x, fs):
    nperseg = min(len(x), 4096); nperseg = max(nperseg, 8)
    f, Pxx = welch(x, fs=fs if fs>0 else 1.0, nperseg=nperseg)
    P = Pxx / (np.sum(Pxx) + 1e-12)
    sc = np.sum(f * P)
    bw = np.sqrt(np.sum(((f - sc) ** 2) * P))
    cdf = np.cumsum(P)
    roll95 = f[np.searchsorted(cdf, 0.95)]
    pfreq = f[np.argmax(Pxx)]
    sent = entropy(P + 1e-12, base=2)
    return dict(spec_centroid=sc, spec_bw=bw, spec_rolloff95=roll95, peak_freq=pfreq, spec_entropy=sent)

def band_energies(x, fs, bands_hz):
    nperseg = min(len(x), 4096); nperseg = max(nperseg, 8)
    f, Pxx = welch(x, fs=fs if fs>0 else 1.0, nperseg=nperseg)
    total = np.trapz(Pxx, f) + 1e-12
    out = {}
    for (lo, hi) in bands_hz:
        lo_ = max(lo, 0.0); hi_ = min(hi, fs/2 if fs>0 else 0.5)
        mask = (f >= lo_) & (f < hi_)
        out[f"band_{int(lo)}_{int(hi)}"] = float(np.trapz(Pxx[mask], f[mask])/total) if np.any(mask) else 0.0
    return out

def stft_texture(x, fs):
    fs_used = fs if fs>0 else 1.0
    nperseg = 1024 if len(x) >= 2048 else 256
    f, t, Z = stft(x, fs=fs_used, nperseg=nperseg, noverlap=nperseg//2)
    A = np.abs(Z)
    return dict(
        stft_mean=float(np.mean(A)),
        stft_std=float(np.std(A)),
        stft_skew=float(skew(A.reshape(-1), bias=False)),
        stft_kurt=float(kurtosis(A.reshape(-1), fisher=True, bias=False))
    )

def time_features(x):
    x = x.astype(np.float64)
    mean = float(np.mean(x)); std=float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2))); krt=float(kurtosis(x, fisher=True, bias=False))
    skw=float(skew(x, bias=False)); peak=float(np.max(np.abs(x)))
    crest=float(peak/(rms+1e-12)); zcr=float(np.mean(np.abs(np.diff(np.sign(x))))/2.0)
    pp_amp=float(np.max(x)-np.min(x)); impulse=float(peak/(np.mean(np.abs(x))+1e-12))
    shape=float(rms/(np.mean(np.abs(x))+1e-12))
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

# ================== 数据→特征表 ==================
def build_feature_table(
    data_root: str,
    fs: int = 12000,
    win_sec: float = 1.0,
    overlap: float = 0.5,
    zscore: bool = True,
    bands_hz = None
):
    if bands_hz is None:
        nyq = fs/2
        edges = list(range(0, int(nyq)+1000, 1000))
        bands_hz = list(zip(edges[:-1], edges[1:])) or [(0,int(nyq))]

    mats = sorted(glob.glob(os.path.join(data_root,"**","*.mat"), recursive=True))
    assert mats, f"未在 {data_root} 找到 .mat 文件"
    rows=[]
    for fp in mats:
        md = load_mat(fp)
        _, sig = choose_signal(md)
        cls = parse_cls_from_name(fp)
        if cls=="UNK": continue
        win_len = int(win_sec*fs); hop_len = max(1,int(win_len*(1-overlap)))
        segments = sliding_windows(sig, win_len, hop_len)
        if segments.shape[0]==0: continue
        if zscore: segments = zscore_norm(segments)
        for i in range(segments.shape[0]):
            feats = extract_features_of_segment(segments[i], fs, bands_hz)
            feats.update(dict(file=os.path.relpath(fp, data_root).replace("\\","/"), seg_idx=i, cls=cls))
            rows.append(feats)
    df = pd.DataFrame(rows)
    cols_front = ["file","seg_idx","cls"]
    other = [c for c in df.columns if c not in cols_front]
    return df[cols_front + sorted(other)]

# ================== 可视化 ==================
def save_tables(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir,"features_segments.csv"), index=False, encoding="utf-8-sig")
    grp = df.groupby("cls")
    agg = pd.concat([grp.mean(numeric_only=True).add_suffix("_mean"),
                     grp.std(numeric_only=True).add_suffix("_std")], axis=1).reset_index()
    agg.to_csv(os.path.join(out_dir,"features_class_agg.csv"), index=False, encoding="utf-8-sig")
    return agg

def plot_box_by_class(df, features, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(df["cls"].unique())
    for feat in features:
        if feat not in df.columns: continue
        plt.figure()
        data=[df.loc[df["cls"]==c, feat].values for c in classes]
        sns.boxplot(data=data)
        plt.xticks(range(len(classes)), classes)
        plt.title(f"{feat} 按类别箱线图")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"box_{feat}.png"), dpi=150); plt.show()

def plot_corr_heatmap(df, out_path):
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10,8))
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
    plt.title("特征相关性热力图")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()

def plot_tsne(df, out_path, max_n=4000, seed=42):
    X = df.select_dtypes(include=[np.number]).values
    y = df["cls"].values
    if len(X) > max_n:
        idx = np.random.RandomState(seed).choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    Xs = StandardScaler().fit_transform(X)
    Z = TSNE(n_components=2, perplexity=30, learning_rate="auto",
             init="random", n_iter=1000, random_state=seed).fit_transform(Xs)
    plt.figure(figsize=(8,6))
    for c in sorted(np.unique(y)):
        Zi = Z[y==c]
        if Zi.size==0: continue
        plt.scatter(Zi[:,0], Zi[:,1], label=c, s=15, alpha=0.7)
    plt.title("t-SNE 特征空间分布"); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()

def plot_topk_importance(df, k, out_path):
    num_cols = [c for c in df.columns if c not in ("file","seg_idx","cls")]
    g = df.groupby("cls")[num_cols].mean()
    score1 = g.var(axis=0) / (df[num_cols].var(axis=0)+1e-12)
    y = df["cls"].values
    groups = [df.loc[y==c, num_cols].values for c in sorted(df["cls"].unique())]
    f_scores=[]
    for j,_ in enumerate(num_cols):
        col_groups=[gi[:,j] for gi in groups if gi.shape[0]>1]
        if len(col_groups)<=1: f_scores.append(0.0)
        else:
            try: F,_ = f_oneway(*col_groups); f_scores.append(float(F))
            except: f_scores.append(0.0)
    score2 = pd.Series(f_scores, index=num_cols)
    s1 = (score1-score1.min())/(score1.max()-score1.min()+1e-12)
    s2 = (score2-score2.min())/(score2.max()-score2.min()+1e-12)
    score = s1+s2
    topk = score.sort_values(ascending=False).head(k)
    plt.figure(figsize=(8,5))
    sns.barplot(x=topk.index, y=topk.values)
    plt.xticks(rotation=45)
    plt.ylabel("重要性评分")
    plt.title(f"Top-{k} 特征重要性")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()
    return topk

# ================== 主函数 ==================
def run_feature_analysis(
    data_root, out_dir="./q1_feat_out", fs=12000, win_sec=1.0, overlap=0.5,
    zscore=True, bands="", tsne_plot=True, topk=15, seed=42
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir,"figs"); os.makedirs(fig_dir, exist_ok=True)

    if bands:
        bands_hz = [(float(lo), float(hi)) for token in bands.split(",") for lo,hi in [token.strip().split("-")]]
    else:
        nyq = fs/2; edges=list(range(0,int(nyq)+1000,1000))
        bands_hz = list(zip(edges[:-1], edges[1:])) or [(0,int(nyq))]

    print(f"[信息] 数据根目录: {data_root} 采样率: {fs}Hz 频带: {bands_hz}")
    df = build_feature_table(data_root, fs, win_sec, overlap, zscore, bands_hz)
    agg = save_tables(df, out_dir)

    core_feats = ["rms","kurtosis","skew","crest_factor","pp_amp",
                  "spec_centroid","spec_bw","spec_rolloff95","peak_freq","spec_entropy"]
    plot_box_by_class(df, core_feats, os.path.join(fig_dir,"box"))
    plot_corr_heatmap(df, os.path.join(fig_dir,"corr_heatmap.png"))
    if tsne_plot: plot_tsne(df, os.path.join(fig_dir,"tsne_features.png"))
    plot_topk_importance(df, topk, os.path.join(fig_dir,f"top{topk}_importance.png"))
    print(f"[完成] 可视化已保存至 {fig_dir}")
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
    p.add_argument("--bands",    type=str, default="2500-4300", help='自定义频带，如 "0-800,800-1600,1600-3200"')
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
