import os, re, glob, sys, argparse, random
import numpy as np
import scipy.io as sio
from scipy.signal import welch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============ 中文 & 随机性 & argv ============
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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)

def sanitize_argv():
    keep = ("--source_root","--target_unlabeled_root","--target_labeled_root",
            "--out_dir","--fs_source","--fs_target","--win_sec","--overlap",
            "--zscore","--align_mode","--clf","--self_train","--self_thr",
            "--max_pl_per_class","--tsne","--pca_dim","--help")
    argv = sys.argv[:]; out=[argv[0]]; i=1; skip=False
    while i < len(argv):
        a = argv[i]
        if skip: skip=False; i+=1; continue
        if a == "-f": skip=True; i+=1; continue
        if any(a.startswith(k) for k in keep):
            out.append(a)
            if ("=" not in a) and i+1<len(argv) and not argv[i+1].startswith("-"):
                out.append(argv[i+1]); i+=1
        i+=1
    sys.argv = out

# ============ I/O & 工具 ============
FN_PAT = re.compile(r'^(?P<cls>IR|OR|B|N)', re.IGNORECASE)
CLS_ORDER = ["B","IR","OR","N"]

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
    k0 = list(md.keys())[0]; return k0, squeeze1d(md[k0])

def sliding_windows(x, win, hop):
    if x.size < win: return np.empty((0,win), dtype=np.float32)
    n = 1 + (x.size - win)//hop
    out = np.zeros((n,win), dtype=np.float32)
    for i in range(n): out[i] = x[i*hop:i*hop+win]
    return out

def zscore_rows(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-12
    return (X - m)/s

def align_segments(X, target_len, mode="resample"):
    N, L = X.shape
    if L == target_len: return X.astype(np.float32, copy=False)
    if mode == "resample":
        xp = np.linspace(0,1,L); xq = np.linspace(0,1,target_len)
        Y = np.empty((N,target_len), dtype=np.float32)
        for i in range(N): Y[i] = np.interp(xq, xp, X[i]).astype(np.float32)
        return Y
    elif mode == "crop":
        if L > target_len:
            s = (L-target_len)//2; return X[:, s:s+target_len].astype(np.float32)
        pad = target_len - L; left = pad//2; right = pad-left
        return np.pad(X, ((0,0),(left,right)), mode="constant").astype(np.float32)
    else:  # pad
        if L >= target_len: return X[:,:target_len].astype(np.float32)
        pad = target_len - L
        return np.pad(X, ((0,0),(0,pad)), mode="constant").astype(np.float32)

def parse_cls_from_name(fname):
    m = FN_PAT.match(os.path.basename(fname))
    if m: return m.group("cls").upper()
    if "normal" in fname.lower(): return "N"
    return "UNK"

# ============ 轻量特征 ============
def time_stats(x):
    mean = np.mean(x); std = np.std(x); rms = np.sqrt(np.mean(x**2) + 1e-12)
    krt = float(np.mean(((x-mean)/(std+1e-12))**4) - 3.0)
    skw = float(np.mean(((x-mean)/(std+1e-12))**3))
    peak = float(np.max(np.abs(x))); crest = float(peak/(rms+1e-12))
    return np.array([mean, std, rms, krt, skw, peak, crest], dtype=np.float32)

def spectral_stats(x, fs):
    nperseg = min(len(x), 2048); nperseg = nperseg if nperseg>=32 else len(x)
    f, Pxx = welch(x, fs=fs if fs>0 else 1.0, nperseg=nperseg)
    P = Pxx/(np.sum(Pxx)+1e-12)
    cen = np.sum(f*P); bw = np.sqrt(np.sum(((f-cen)**2)*P))
    dom = f[np.argmax(Pxx)]
    # 分段能量（最多到8kHz，避免维度过大）
    edges = [0,500,1000,2000,4000,8000,16000,32000]
    bands=[]
    for i in range(len(edges)-1):
        lo,hi = edges[i], edges[i+1]
        mask = (f>=lo) & (f<hi)
        bands.append(np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0)
    bands = np.array(bands, dtype=np.float32)
    total = np.trapz(Pxx, f) + 1e-12
    bands = bands/total
    return np.r_[cen, bw, dom, bands].astype(np.float32)

def build_feature_for_segments(X, fs):
    # 每个切片 -> [时域7维 + 频域(3+分段能量)]
    feats=[]
    for i in range(len(X)):
        t = time_stats(X[i])
        s = spectral_stats(X[i], fs)
        feats.append(np.r_[t, s])
    return np.vstack(feats).astype(np.float32)

# ============ 读取数据集 ============
def read_source(source_root, fs, win_sec, overlap, zscore, align_mode):
    # 支持：source_root/IR|OR|B|N/*.mat  或  source_root/*.mat(按文件名前缀解析)
    files = sorted(glob.glob(os.path.join(source_root, "**", "*.mat"), recursive=True))
    assert files, f"源域空目录：{source_root}"
    Xs=[]; ys=[]; file_ids=[]
    for fp in files:
        md = load_mat(fp); _, sig = choose_signal(md)
        win=int(win_sec*fs); hop=max(1, int(win*(1-overlap)))
        segs = sliding_windows(sig, win, hop)
        if segs.size==0: continue
        if zscore: segs = zscore_rows(segs)
        # 长度统一（以 win 为基准）
        if segs.shape[1] != win:
            segs = align_segments(segs, win, mode=align_mode)
        cls = parse_cls_from_name(fp)
        if cls == "UNK":
            # 尝试从父文件夹名判断
            parent = os.path.basename(os.path.dirname(fp)).upper()
            cls = parent if parent in CLS_ORDER else "UNK"
        if cls == "UNK":  # 忽略未知
            continue
        Xs.append(segs); ys.append(np.full(len(segs), cls)); file_ids.extend([fp]*len(segs))
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    cls_used = sorted({c for c in y})
    cls2id = {c:i for i,c in enumerate(cls_used)}
    yid = np.array([cls2id[c] for c in y], dtype=np.int64)
    print(f"[源域] {source_root} → 切片 {X.shape}, 类别映射 {cls2id}")
    return X, yid, cls2id, np.array(file_ids)

def read_target_unlabeled(target_unlabeled_root, fs, win_sec, overlap, zscore, align_mode):
    mats = sorted(glob.glob(os.path.join(target_unlabeled_root, "*.mat")))
    assert mats, f"目标域无标注空目录：{target_unlabeled_root}"
    Xs=[]; file_ids=[]
    win=int(win_sec*fs); hop=max(1, int(win*(1-overlap)))
    for fp in mats:
        md = load_mat(fp); _, sig = choose_signal(md)
        segs = sliding_windows(sig, win, hop)
        if segs.size==0: continue
        if zscore: segs = zscore_rows(segs)
        if segs.shape[1] != win:
            segs = align_segments(segs, win, mode=align_mode)
        Xs.append(segs); file_ids.extend([fp]*len(segs))
    X = np.vstack(Xs).astype(np.float32)
    print(f"[目标-无标注] {target_unlabeled_root} → 切片 {X.shape}")
    return X, np.array(file_ids)

def read_target_labeled_folder(target_labeled_root, fs, win_sec, overlap, zscore, align_mode):
    if not (target_labeled_root and os.path.isdir(target_labeled_root)):
        return None, None, None, None
    Xs=[]; ys=[]; file_ids=[]
    win=int(win_sec*fs); hop=max(1, int(win*(1-overlap)))
    found=False
    for c in CLS_ORDER:
        mats = sorted(glob.glob(os.path.join(target_labeled_root, c, "*.mat")))
        if not mats: continue
        found=True
        for fp in mats:
            md = load_mat(fp); _, sig = choose_signal(md)
            segs = sliding_windows(sig, win, hop)
            if segs.size==0: continue
            if zscore: segs = zscore_rows(segs)
            if segs.shape[1] != win:
                segs = align_segments(segs, win, mode=align_mode)
            Xs.append(segs); ys.append(np.full(len(segs), c)); file_ids.extend([fp]*len(segs))
    if not found: return None, None, None, None
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    cls_used = [c for c in CLS_ORDER if np.any(y==c)]
    cls2id = {c:i for i,c in enumerate(cls_used)}
    yid = np.array([cls2id[c] for c in y], dtype=np.int64)
    print(f"[目标-有标注] {target_labeled_root} → 切片 {X.shape}, 类别映射 {cls2id}")
    return X, yid, cls2id, np.array(file_ids)

# ============ CORAL 特征对齐 ============
def coral_fit(Xs, Xt, eps=1e-5):
    # 学习仿射：Xs_whiten -> recolor to Xt
    ms = np.mean(Xs, axis=0, keepdims=True)
    mt = np.mean(Xt, axis=0, keepdims=True)
    Cs = np.cov((Xs-ms).T) + eps*np.eye(Xs.shape[1])
    Ct = np.cov((Xt-mt).T) + eps*np.eye(Xt.shape[1])
    # 对称平方根与逆平方根
    # 为了速度与鲁棒性，用SVD
    Us, Ss, _ = np.linalg.svd(Cs)
    Ut, St, _ = np.linalg.svd(Ct)
    Cs_m12 = Us @ np.diag(1.0/np.sqrt(Ss+eps)) @ Us.T
    Ct_p12 = Ut @ np.diag(np.sqrt(St+eps)) @ Ut.T
    A = Cs_m12 @ Ct_p12
    return ms, mt, A

def coral_transform(X, ms, mt, A):
    return (X - ms) @ A + mt

# ============ 分类器 ============
def build_clf(name="lr"):
    if name.lower() == "svm":
        # 线性 SVM，速度快；probability 无法直接取，用决策函数近似置信度
        clf = LinearSVC(C=1.0, max_iter=5000)
        conf_mode = "decision"
    else:
        # 逻辑回归（默认），可输出概率
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="auto")
        conf_mode = "proba"
    return clf, conf_mode

# ============ 文件级投票 ============
def file_vote(file_ids, yhat, conf=None):
    # 统计每个文件内的切片预测，投票为文件标签；若给 conf，做加权投票
    files = np.unique(file_ids)
    file_label = {}
    for f in files:
        idx = np.where(file_ids == f)[0]
        labels = yhat[idx]
        if conf is None:
            vals, cnts = np.unique(labels, return_counts=True)
            lab = vals[np.argmax(cnts)]
        else:
            labs = np.unique(labels)
            scores=[]
            for L in labs:
                scores.append(conf[idx][labels==L].sum())
            lab = labs[np.argmax(scores)]
        file_label[f] = int(lab)
    return file_label

# ============ 主流程 ============
def run_q4(args):
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    figs = os.path.join(args.out_dir, "figs"); os.makedirs(figs, exist_ok=True)

    # 1) 读取源/目标数据，切片 & 轻量特征
    Xs_seg, ys, cls2id_s, src_file_ids = read_source(
        args.source_root, args.fs_source, args.win_sec, args.overlap, bool(args.zscore), args.align_mode
    )
    Xt_u_seg, tgt_u_files = read_target_unlabeled(
        args.target_unlabeled_root, args.fs_target, args.win_sec, args.overlap, bool(args.zscore), args.align_mode
    )
    # 统一特征尺度：源/目标各自按各自采样率提取
    Xs_feat = build_feature_for_segments(Xs_seg, fs=args.fs_source)
    Xt_u_feat = build_feature_for_segments(Xt_u_seg, fs=args.fs_target)

    # 可选：目标域有标注（用于评测）
    Xtl_seg = ytl = cls2id_t = tl_files = None
    if args.target_labeled_root and os.path.isdir(args.target_labeled_root):
        Xtl_seg, ytl, cls2id_t, tl_files = read_target_labeled_folder(
            args.target_labeled_root, args.fs_target, args.win_sec, args.overlap, bool(args.zscore), args.align_mode
        )
        if Xtl_seg is not None:
            Xtl_feat = build_feature_for_segments(Xtl_seg, fs=args.fs_target)
        else:
            Xtl_feat = None
    else:
        Xtl_feat = None

    # 2) 标准化（源+目标联合 fit，避免偏移）
    scaler = StandardScaler().fit(np.vstack([Xs_feat, Xt_u_feat]))
    Xs_z = scaler.transform(Xs_feat)
    Xt_u_z = scaler.transform(Xt_u_feat)
    if Xtl_feat is not None:
        Xtl_z = scaler.transform(Xtl_feat)

    # 3) CORAL：用目标无标注统计对齐源域
    ms, mt, A = coral_fit(Xs_z, Xt_u_z)
    Xs_c = coral_transform(Xs_z, ms, mt, A)

    # 4) 轻量分类器（源域对齐后训练）
    clf, conf_mode = build_clf(args.clf)
    clf.fit(Xs_c, ys)

    # 5) 直接在目标无标注上推理，得到伪标签与置信度
    if conf_mode == "proba":
        proba_u = clf.predict_proba(Xt_u_z)
        yhat_u = np.argmax(proba_u, axis=1)
        conf_u = np.max(proba_u, axis=1)
    else:
        dec = clf.decision_function(Xt_u_z)
        if dec.ndim == 1:  # 二分类退化
            proba_u = 1/(1+np.exp(-dec))
            yhat_u = (proba_u >= 0.5).astype(int)
            conf_u = np.maximum(proba_u, 1-proba_u)
        else:
            # 线性SVM的“伪概率”——归一化决策值
            dec_shift = dec - dec.min(axis=1, keepdims=True)
            proba_u = dec_shift / (dec_shift.sum(axis=1, keepdims=True)+1e-12)
            yhat_u = np.argmax(proba_u, axis=1)
            conf_u = np.max(proba_u, axis=1)

    # 6) 一次性自训练（可选）：挑高置信度目标样本拼回训练
    if args.self_train:
        thr = float(args.self_thr)
        idx = np.where(conf_u >= thr)[0]
        if idx.size > 0:
            # 每类限量，防止倾斜
            max_k = int(args.max_pl_per_class)
            take_idx=[]
            for c in np.unique(yhat_u[idx]):
                ic = idx[yhat_u[idx]==c]
                take_idx.extend(ic[:max_k])
            take_idx = np.array(take_idx, dtype=int)
            X_pl = Xt_u_z[take_idx]  # 注意：已经在目标域标准化空间
            y_pl = yhat_u[take_idx]
            # 直接在目标域空间再 fit 一次（温和做法：只微调，不清空）
            X_aug = np.vstack([Xs_c, X_pl])
            y_aug = np.concatenate([ys, y_pl])
            clf.fit(X_aug, y_aug)
            print(f"[自训练] 采用伪标签样本 {len(take_idx)} 条，阈值 {thr}，每类≤{args.max_pl_per_class}")
        else:
            print(f"[自训练] 阈值 {thr} 下无高置信度样本，跳过。")

    # 7) 若有目标域标注，评测
    if Xtl_feat is not None:
        # 注意：目标标注是在目标标准化空间，不需要 CORAL
        if conf_mode == "proba":
            proba_tl = clf.predict_proba(Xtl_z)
            yhat_tl = np.argmax(proba_tl, axis=1)
        else:
            dec = clf.decision_function(Xtl_z)
            if dec.ndim == 1:
                proba_tl = 1/(1+np.exp(-dec))
                yhat_tl = (proba_tl >= 0.5).astype(int)
            else:
                dec_shift = dec - dec.min(axis=1, keepdims=True)
                proba_tl = dec_shift / (dec_shift.sum(axis=1, keepdims=True)+1e-12)
                yhat_tl = np.argmax(proba_tl, axis=1)

        id2cls_t = {i:c for c,i in cls2id_t.items()}
        print("\n[目标-标注] 分类报告：")
        print(classification_report(ytl, yhat_tl, target_names=[id2cls_t[i] for i in range(len(id2cls_t))]))
        cm = confusion_matrix(ytl, yhat_tl)
        plt.figure(); plt.imshow(cm, aspect='auto')
        plt.title("目标域有标注：混淆矩阵"); plt.xlabel("预测类别"); plt.ylabel("真实类别")
        plt.xticks(range(len(id2cls_t)), [id2cls_t[i] for i in range(len(id2cls_t))], rotation=45)
        plt.yticks(range(len(id2cls_t)), [id2cls_t[i] for i in range(len(id2cls_t))])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
        plt.tight_layout(); plt.savefig(os.path.join(figs, "cm_target_labeled.png"), dpi=150); plt.show()

    # 8) 目标无标注：文件级投票 & 导出
    file_pred = file_vote(tgt_u_files, yhat_u, conf=conf_u)
    # 导出切片级结果
    np.savetxt(os.path.join(args.out_dir, "target_unlabeled_slice_pred.csv"),
               np.c_[tgt_u_files, yhat_u, conf_u], fmt="%s", delimiter=",",
               header="file,yhat,conf", comments="")
    # 导出文件级结果
    with open(os.path.join(args.out_dir, "target_unlabeled_file_vote.csv"), "w", encoding="utf-8") as f:
        f.write("file,yhat\n")
        for k,v in file_pred.items():
            f.write(f"{k},{v}\n")
    print(f"[输出] 切片预测与文件投票已保存至：{args.out_dir}")

    # 9) 可视化：目标无标注的置信度直方图 + 可选 TSNE/PCA
    plt.figure(); plt.hist(conf_u, bins=50)
    plt.xlabel("预测置信度"); plt.ylabel("计数"); plt.title("目标无标注：置信度分布")
    plt.tight_layout(); plt.savefig(os.path.join(figs, "conf_u_hist.png"), dpi=150); plt.show()

    if args.tsne:
        # 为速度，仅采样部分目标样本 + 源样本
        n_s = min(2000, len(Xs_c)); n_u = min(2000, len(Xt_u_z))
        sel_s = np.random.choice(len(Xs_c), n_s, replace=False)
        sel_u = np.random.choice(len(Xt_u_z), n_u, replace=False)
        Z = np.vstack([Xs_c[sel_s], Xt_u_z[sel_u]])
        lab = np.r_[ys[sel_s], np.full(n_u, -1)]
        # 降维（PCA->TSNE 更快更稳）
        pca = PCA(n_components=min(30, Z.shape[1]))
        Zp = pca.fit_transform(Z)
        Zt = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto").fit_transform(Zp)
        plt.figure()
        plt.scatter(Zt[:n_s,0], Zt[:n_s,1], s=5, label="源(对齐)")
        plt.scatter(Zt[n_s:,0], Zt[n_s:,1], s=5, label="目标")
        plt.legend(); plt.title("t-SNE：源(对齐) vs 目标")
        plt.tight_layout(); plt.savefig(os.path.join(figs, "tsne_src_tgt.png"), dpi=150); plt.show()

# ============ CLI ============
def build_args():
    p = argparse.ArgumentParser("第四问：快速域适配（CORAL + 轻量分类器 + 一次性伪标签）", allow_abbrev=False)
    p.add_argument("--source_root", type=str, default="./源域数据集", help="源域根目录（支持 B/IR/OR/N 子文件夹或前缀）")
    p.add_argument("--target_unlabeled_root", type=str, default="./目标域数据集", help="目标域无标注 .mat 所在目录")
    p.add_argument("--target_labeled_root", type=str, default="./目标域有标注", help="可选：目标域有标注目录（B/IR/OR/N 子文件夹）")
    p.add_argument("--out_dir", type=str, default="./q4_out")
    p.add_argument("--fs_source", type=int, default=12000)
    p.add_argument("--fs_target", type=int, default=32000)
    p.add_argument("--win_sec", type=float, default=1.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--zscore", action="store_true")
    p.add_argument("--align_mode", type=str, default="resample", choices=["resample","crop","pad"])
    p.add_argument("--clf", type=str, default="lr", choices=["lr","svm"], help="分类器：lr=逻辑回归；svm=线性SVM")
    p.add_argument("--self_train", action="store_true", help="是否启用一次性伪标签自训练")
    p.add_argument("--self_thr", type=float, default=0.9, help="伪标签置信度阈值")
    p.add_argument("--max_pl_per_class", type=int, default=500, help="每类最多加入的伪标签数量")
    p.add_argument("--tsne", action="store_true", help="是否绘制 t-SNE 快速可视化")
    p.add_argument("--pca_dim", type=int, default=30)  # 预留
    return p

if __name__ == "__main__":
    sanitize_argv()
    args = build_args().parse_args()
    run_q4(args)
