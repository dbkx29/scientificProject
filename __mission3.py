import os, re, glob, math, random, argparse, sys
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras import layers, models, losses
from scipy.signal import welch
from scipy.stats import entropy as sp_entropy

# ========== 关键修复点 1：使用 legacy.Adam，避免解冻后 KeyError ==========
try:
    AdamLegacy = tf.keras.optimizers.legacy.Adam
except Exception:
    # 极少环境没有 legacy，回退到普通 Adam（一般也能跑）
    from tensorflow.keras import optimizers
    AdamLegacy = optimizers.Adam

# ================== 中文 & 随机性 & argv ==================
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
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def sanitize_argv():
    keep = ("--target_labeled_root","--target_unlabeled_root","--out_dir",
            "--fs_target","--win_sec","--overlap","--zscore",
            "--align_mode",
            "--fe_path","--seed_model","--seed_thr","--seed_per_class","--seed_total",
            "--k_classes",
            "--batch_size","--epochs","--lr",
            "--tau","--lambda_u","--tsne_n",
            "--strong_jitter","--strong_scale","--strong_drop",
            "--help")
    argv = sys.argv[:]; out=[argv[0]]; i=1; skip=False
    while i < len(argv):
        a = argv[i]
        if skip: skip=False; i+=1; continue
        if a == "-f": skip=True; i+=1; continue
        if any(a.startswith(k) for k in keep):
            out.append(a)
            if ("=" not in a) and (i+1 < len(argv)) and (not argv[i+1].startswith("-")):
                out.append(argv[i+1]); i+=1
        i+=1
    sys.argv = out

# ================== 数据与工具 ==================
FN_PAT = re.compile(r'^(?P<cls>IR|OR|B|N)', re.IGNORECASE)
CLS_LIST = ["B","IR","OR","N"]  # 文件夹标注时的常见四类

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
    k0=list(md.keys())[0]; return k0, squeeze1d(md[k0])

def sliding_windows(x, win, hop):
    if x.size < win: return np.empty((0,win), dtype=np.float32)
    n = 1 + (x.size - win)//hop
    out = np.zeros((n,win), dtype=np.float32)
    for i in range(n): out[i,:] = x[i*hop:i*hop+win]
    return out

def zscore_norm(X):
    m = X.mean(axis=1, keepdims=True); s = X.std(axis=1, keepdims=True) + 1e-12
    return (X - m) / s

def align_segments(X: np.ndarray, target_len: int, mode: str = "resample") -> np.ndarray:
    N, Lin = X.shape
    if Lin == target_len: return X.astype(np.float32, copy=False)
    if mode == "resample":
        xp = np.linspace(0.0, 1.0, Lin, dtype=np.float64)
        xq = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        Y = np.empty((N, target_len), dtype=np.float32)
        for i in range(N): Y[i] = np.interp(xq, xp, X[i]).astype(np.float32)
        return Y
    elif mode == "crop":
        if Lin > target_len:
            s = (Lin - target_len)//2
            return X[:, s:s+target_len].astype(np.float32)
        else:
            pad = target_len - Lin
            left = pad//2; right = pad - left
            return np.pad(X, ((0,0),(left,right)), mode="constant").astype(np.float32)
    else:  # pad
        if Lin >= target_len:
            return X[:,:target_len].astype(np.float32)
        else:
            pad = target_len - Lin
            return np.pad(X, ((0,0),(0,pad)), mode="constant").astype(np.float32)

def parse_cls_from_name(fname: str):
    m = FN_PAT.match(os.path.basename(fname))
    if m: return m.group("cls").upper()
    if "normal" in fname.lower(): return "N"
    return "UNK"

# ---------- 读取有标注/无标注 ----------
def build_target_labeled_folder(root, fs, win_sec, overlap, zscore):
    Xs, ys = [], []
    if not (root and os.path.isdir(root)):
        return None, None, None  # 没有标注文件夹
    found_any = False
    for sub in sorted(os.listdir(root)):
        c = sub.upper()
        if c not in CLS_LIST: continue
        mat_files = sorted(glob.glob(os.path.join(root, sub, "*.mat")))
        if not mat_files: continue
        found_any = True
        for fp in mat_files:
            md = load_mat(fp); _, sig = choose_signal(md)
            win = int(win_sec*fs); hop = max(1, int(win*(1-overlap)))
            segs = sliding_windows(sig, win, hop)
            if segs.shape[0]==0: continue
            if zscore: segs = zscore_norm(segs)
            Xs.append(segs); ys.append(np.full(segs.shape[0], c))
    if not found_any:
        return None, None, None
    X = np.vstack(Xs).astype(np.float32); y = np.concatenate(ys)
    classes = [c for c in CLS_LIST if np.any(y==c)]
    cls2id = {c:i for i,c in enumerate(classes)}
    yid = np.array([cls2id[c] for c in y], dtype=np.int64)
    print(f"[有标注] 文件夹：{root}  → {X.shape}，类别映射：{cls2id}")
    return X, yid, cls2id

def build_target_unlabeled(root, fs, win_sec, overlap, zscore):
    mats = sorted(glob.glob(os.path.join(root, "*.mat")))
    assert mats, f"未在 {root} 找到无标注 .mat"
    Xs, files = [], []
    for fp in mats:
        md = load_mat(fp); _, sig = choose_signal(md)
        win = int(win_sec*fs); hop = max(1, int(win*(1-overlap)))
        segs = sliding_windows(sig, win, hop)
        if segs.shape[0]==0: continue
        if zscore: segs = zscore_norm(segs)
        Xs.append(segs); files.extend([fp]*segs.shape[0])
    X = np.vstack(Xs).astype(np.float32)
    files = np.array(files)
    print(f"[无标注] {root}  → {X.shape}")
    return X, files

# ================== 模型（ResNet1D + 线性头） ==================
def resnet1d_backbone(input_len):
    inp = layers.Input(shape=(input_len,1))
    x = layers.Conv1D(32,7,strides=2,padding="same",use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool1D(3,strides=2,padding="same")(x)
    def block(x, filters, stride):
        sc = x
        y = layers.Conv1D(filters,7,strides=stride,padding="same",use_bias=False)(x)
        y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
        y = layers.Conv1D(filters,7,strides=1,padding="same",use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        if stride!=1 or sc.shape[-1]!=filters:
            sc = layers.Conv1D(filters,1,strides=stride,padding="same",use_bias=False)(sc)
            sc = layers.BatchNormalization()(sc)
        y = layers.add([y, sc]); y = layers.ReLU()(y)
        return y
    x = block(x,64,1)
    x = block(x,128,2)
    x = block(x,256,2)
    feat = layers.GlobalAveragePooling1D(name="feat_gap")(x)  # (B,256)
    return models.Model(inp, feat, name="FeatureExtractor")

def build_classifier(fe, n_classes):
    inp = layers.Input(shape=fe.input_shape[1:])
    z = fe(inp)
    out = layers.Dense(n_classes, activation="softmax", name="cls_head")(z)
    return models.Model(inp, out, name="TargetClassifier")

# ================== 1D 增强（弱/强） ==================
@tf.function
def aug_weak(x):
    L = tf.shape(x)[1]
    shift = tf.random.uniform([], minval=-tf.cast(0.02*tf.cast(L,tf.float32), tf.int32),
                              maxval= tf.cast(0.02*tf.cast(L,tf.float32), tf.int32), dtype=tf.int32)
    x = tf.roll(x, shift=shift, axis=1)
    noise = tf.random.normal(tf.shape(x), stddev=0.01)
    return x + noise

@tf.function
def aug_strong(x, jitter=0.03, scale=0.1, drop=0.1):
    noise = tf.random.normal(tf.shape(x), stddev=jitter)
    gain  = tf.random.uniform([tf.shape(x)[0],1,1], 1.0-scale, 1.0+scale)
    x = (x + noise) * gain
    if drop > 0:
        mask = tf.cast(tf.random.uniform(tf.shape(x)) > drop, x.dtype)
        x = x * mask
    return x

# ================== 安全加载模型（.h5 或 SavedModel 目录） ==================
def try_load_keras_model(path):
    if not path: return None
    if not os.path.exists(path): return None
    try:
        mdl = models.load_model(path, compile=False)
        print(f"[模型] 成功加载：{path}")
        return mdl
    except Exception as e:
        print(f"[模型] 加载失败（{e}）：{path}")
        return None

# ================== 频域特征（无 FE 时的无监督备选） ==================
def spectral_feature_vec(x, fs):
    nperseg = min(len(x), 4096); nperseg = nperseg if nperseg>=8 else len(x)
    f, Pxx = welch(x, fs=fs if fs>0 else 1.0, nperseg=nperseg)
    P = Pxx / (np.sum(Pxx) + 1e-12)
    sc = np.sum(f * P)  # 质心
    bw = np.sqrt(np.sum(((f - sc) ** 2) * P))  # 带宽
    cdf = np.cumsum(P); roll95 = f[np.searchsorted(cdf, 0.95)]
    pfreq = f[np.argmax(Pxx)]
    sent = sp_entropy(P + 1e-12, base=2)
    nyq = fs/2 if fs>0 else 0.5
    edges = list(range(0, int(nyq)+1000, 1000)) or [0, int(nyq)]
    bands = []
    for lo,hi in zip(edges[:-1], edges[1:]):
        mask = (f>=lo) & (f<hi)
        e = np.trapz(Pxx[mask], f[mask]) / (np.trapz(Pxx, f) + 1e-12) if np.any(mask) else 0.0
        bands.append(e)
    return np.r_[sc, bw, roll95, pfreq, sent, bands].astype(np.float32)

def build_unsup_features(X, fs):
    Z = np.zeros((len(X), 5 + max(1, int((fs/2)//1000))), dtype=np.float32)
    for i in range(len(X)):
        Z[i] = spectral_feature_vec(X[i], fs)
    return Z

# ================== 自动种子：优先模型，其次 KMeans ==================
def autoseed_from_model_or_kmeans(Xu, fs, input_len, align_mode,
                                  fe_path, seed_model, seed_thr, seed_per_class, seed_total,
                                  k_classes=4):
    # 1) 先尝试 seed_model 伪标注
    mdl = try_load_keras_model(seed_model) if seed_model else None
    if mdl is not None:
        try:
            outs = mdl.output if isinstance(mdl.output, (list, tuple)) else [mdl.output]
            names = [getattr(o, "name", f"out{i}") for i,o in enumerate(outs)]
            if any("cls_head" in n for n in names):
                cls_out = [o for o,n in zip(outs, names) if "cls_head" in n][0]
                mdl = models.Model(mdl.input, cls_out)
        except Exception:
            pass
        X_in = Xu if Xu.shape[1]==input_len else align_segments(Xu, input_len, mode=align_mode)
        probs=[]
        for i in range(0, len(X_in), 512):
            p = mdl.predict(X_in[i:i+512,:,None], verbose=0)
            probs.append(p)
        P = np.vstack(probs); pred = P.argmax(1); conf = P.max(1)
        idx_thr = np.where(conf >= float(seed_thr))[0]
        if idx_thr.size > 0:
            X_seed = X_in[idx_thr]; y_seed = pred[idx_thr]
            Xs, ys = [], []
            for c in np.unique(y_seed):
                ids = np.where(y_seed==c)[0][:seed_per_class]
                Xs.append(X_seed[ids]); ys.append(np.full(len(ids), c, dtype=np.int64))
            Xs = np.vstack(Xs); ys = np.concatenate(ys)
            if seed_total is not None and Xs.shape[0] > seed_total:
                sel = np.random.RandomState(42).choice(Xs.shape[0], seed_total, replace=False)
                Xs, ys = Xs[sel], ys[sel]
            cls2id = {c:i for i,c in enumerate(sorted(np.unique(ys).tolist()))}
            ys = np.array([cls2id[c] for c in ys], dtype=np.int64)
            print(f"[种子-模型] 采用 {Xs.shape[0]} 条（阈值 {seed_thr}，每类≤{seed_per_class}）")
            return Xs, ys, cls2id
        print("[种子-模型] 高置信度不足，转用 KMeans。")

    # 2) KMeans（优先 FE 特征，没有 FE 则频域特征）
    fe = try_load_keras_model(fe_path) if fe_path else None
    if fe is not None and hasattr(fe, "input_shape"):
        L = fe.input_shape[1]
        X_in = Xu if Xu.shape[1]==L else align_segments(Xu, L, mode=align_mode)
        feat_model = models.Model(fe.input, fe.layers[-1].output)
        Z=[]
        for i in range(0, len(X_in), 512):
            Z.append(feat_model.predict(X_in[i:i+512,:,None], verbose=0))
        Z = np.vstack(Z)
        print(f"[KMeans] 使用 FE 特征：{Z.shape}")
    else:
        Z = build_unsup_features(Xu, fs)
        print(f"[KMeans] 使用频域特征：{Z.shape}")

    k = int(k_classes)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)
    d = np.linalg.norm(Z - km.cluster_centers_[labels], axis=1)
    Xs, ys = [], []
    for c in range(k):
        ids = np.where(labels==c)[0]
        if ids.size == 0: continue
        order = ids[np.argsort(d[ids])]
        take = order[:min(seed_per_class, order.size)]
        Xs.append(Xu[take]); ys.append(np.full(len(take), c, dtype=np.int64))
    if not Xs:
        return None
    Xs = np.vstack(Xs); ys = np.concatenate(ys)
    if seed_total is not None and Xs.shape[0] > seed_total:
        sel = np.random.RandomState(42).choice(Xs.shape[0], seed_total, replace=False)
        Xs, ys = Xs[sel], ys[sel]
    cls2id = {c:i for i,c in enumerate(sorted(np.unique(ys).tolist()))}
    print(f"[种子-KMeans] 采用 {Xs.shape[0]} 条（簇数={k}，每簇≤{seed_per_class}）")
    return Xs, ys, cls2id

# ================== 训练（FixMatch） ==================
def train_fixmatch(args):
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    figs = os.path.join(args.out_dir, "figs"); os.makedirs(figs, exist_ok=True)

    Xu, f_u = build_target_unlabeled(args.target_unlabeled_root, fs=args.fs_target,
                                     win_sec=args.win_sec, overlap=args.overlap, zscore=bool(args.zscore))
    Xl, yl, cls2id = build_target_labeled_folder(args.target_labeled_root, fs=args.fs_target,
                                                 win_sec=args.win_sec, overlap=args.overlap, zscore=bool(args.zscore))

    base_len = (Xl.shape[1] if Xl is not None else int(args.win_sec*args.fs_target))
    if Xl is None:
        print("[提示] 未检测到 '目标域有标注' 文件夹，进入无标注自举模式（autoseed）。")
        X_seed, y_seed, cls2id = autoseed_from_model_or_kmeans(
            Xu=Xu, fs=args.fs_target, input_len=base_len, align_mode=args.align_mode,
            fe_path=args.fe_path, seed_model=args.seed_model, seed_thr=args.seed_thr,
            seed_per_class=args.seed_per_class, seed_total=args.seed_total,
            k_classes=args.k_classes
        )
        if X_seed is None:
            raise FileNotFoundError("自举失败：没有可用的种子样本。请检查 --seed_model 路径或增大 --seed_per_class / 降低 --seed_thr。")
        Xl, yl = X_seed, y_seed

    id2cls = {i:c for c,i in cls2id.items()}

    if Xu.shape[1] != base_len:
        print(f"[对齐] 无标注切片 {Xu.shape[1]} → {base_len}（{args.align_mode}）")
        Xu = align_segments(Xu, base_len, mode=args.align_mode)
    if Xl.shape[1] != base_len:
        Xl = align_segments(Xl, base_len, mode=args.align_mode)

    fe = try_load_keras_model(args.fe_path)
    if fe is None or (hasattr(fe, "input_shape") and fe.input_shape[1]!=base_len):
        fe = resnet1d_backbone(base_len)
        print("[模型] 使用随机初始化的特征提取器（或长度不匹配已重建）")

    clf = build_classifier(fe, n_classes=len(cls2id))

    # 冻结前 1/3 轮次
    for layer in fe.layers:
        layer.trainable = False

    bs = args.batch_size
    ds_l = tf.data.Dataset.from_tensor_slices((Xl[:,:,None], yl)).shuffle(len(Xl)).batch(bs).prefetch(tf.data.AUTOTUNE)
    ds_u = tf.data.Dataset.from_tensor_slices((Xu[:,:,None],)).shuffle(len(Xu)).batch(bs).prefetch(tf.data.AUTOTUNE)

    # ========== 关键修复点 2：legacy.Adam + None 梯度过滤 ==========
    opt = AdamLegacy(learning_rate=args.lr)
    ce  = losses.SparseCategoricalCrossentropy()

    hist = dict(sup=[], cons=[], total=[], acc=[])

    for ep in range(1, args.epochs+1):
        if ep == int(args.epochs/3)+1:
            for layer in fe.layers:
                layer.trainable = True
            print("[训练] 解冻特征提取器，开始联合微调。")

        it_l = iter(ds_l); it_u = iter(ds_u)
        steps = min(len(ds_l), len(ds_u))
        sup_loss = cons_loss = total_loss = 0.0; n = 0

        for _ in range(steps):
            xb_l, yb_l = next(it_l)
            (xb_u,)    = next(it_u)

            xw = aug_weak(xb_u)
            jitter = getattr(args, "strong_jitter", 0.03)
            scale  = getattr(args, "strong_scale",  0.10)
            drop   = getattr(args, "strong_drop",   0.10)
            xs = aug_strong(xb_u, jitter=jitter, scale=scale, drop=drop)

            with tf.GradientTape() as tape:
                logits_l = clf(xb_l, training=True)
                L_sup = ce(yb_l, logits_l)

                pw = tf.stop_gradient(tf.nn.softmax(clf(xw, training=True), axis=1))
                conf = tf.reduce_max(pw, axis=1)
                yhat = tf.argmax(pw, axis=1, output_type=tf.int32)
                mask = tf.cast(conf >= args.tau, tf.float32)

                logits_s = clf(xs, training=True)
                L_cons_all = tf.keras.losses.sparse_categorical_crossentropy(yhat, logits_s, from_logits=False)
                L_cons = tf.reduce_mean(mask * L_cons_all)

                L = L_sup + args.lambda_u * L_cons

            grads = tape.gradient(L, clf.trainable_variables)
            # 过滤 None 梯度，避免个别层因 mask 导致无梯度时报错
            gv = [(g, v) for g, v in zip(grads, clf.trainable_variables) if g is not None]
            if gv:
                opt.apply_gradients(gv)

            sup_loss += float(L_sup.numpy()) * xb_l.shape[0]
            cons_loss += float(L_cons.numpy()) * xb_l.shape[0]
            total_loss += float(L.numpy()) * xb_l.shape[0]
            n += xb_l.shape[0]

        accs=[]
        for xb, yb in tf.data.Dataset.from_tensor_slices((Xl[:,:,None], yl)).batch(bs):
            p = clf(xb, training=False)
            accs.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,1), yb), tf.float32)).numpy())
        acc = float(np.mean(accs))

        hist["sup"].append(sup_loss/max(1,n))
        hist["cons"].append(cons_loss/max(1,n))
        hist["total"].append(total_loss/max(1,n))
        hist["acc"].append(acc)
        print(f"[Epoch {ep:02d}] 监督={hist['sup'][-1]:.4f}  一致性={hist['cons'][-1]:.4f}  总损失={hist['total'][-1]:.4f}  准确率={acc:.4f}")

    clf.save(os.path.join(args.out_dir, "q3_target_fixmatch_autoseed_v2.h5"))
    print(f"[保存] 模型：{os.path.join(args.out_dir, 'q3_target_fixmatch_autoseed_v2.h5')}")

    plt.figure(); plt.plot(hist["sup"]); plt.plot(hist["cons"]); plt.plot(hist["total"])
    plt.xlabel("轮次"); plt.ylabel("损失"); plt.legend(["监督","一致性","总损失"]); plt.title("训练损失（FixMatch）")
    plt.tight_layout(); plt.savefig(os.path.join(figs,"loss_curves.png"), dpi=150); plt.show()

    plt.figure(); plt.plot(hist["acc"]); plt.xlabel("轮次"); plt.ylabel("准确率"); plt.title("标注/种子集准确率")
    plt.tight_layout(); plt.savefig(os.path.join(figs,"acc_seed.png"), dpi=150); plt.show()

    y_pred=[]
    for xb, yb in tf.data.Dataset.from_tensor_slices((Xl[:,:,None], yl)).batch(bs):
        y_pred.append(tf.argmax(clf(xb, training=False), axis=1).numpy())
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    cm = confusion_matrix(yl, y_pred, labels=list(range(len(id2cls))))
    plt.figure(); plt.imshow(cm, aspect='auto')
    plt.title("标注/种子集混淆矩阵"); plt.xlabel("预测类别"); plt.ylabel("真实类别")
    ticks=[id2cls[i] for i in range(len(id2cls))]
    plt.xticks(range(len(ticks)), ticks, rotation=45); plt.yticks(range(len(ticks)), ticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(os.path.join(figs,"cm_seed.png"), dpi=150); plt.show()

    conf_all=[]
    for i in range(0, len(Xu), 512):
        p = clf.predict(Xu[i:i+512,:,None], verbose=0)
        conf_all.append(p.max(1))
    conf_all = np.concatenate(conf_all) if conf_all else np.array([])
    plt.figure(); plt.hist(conf_all, bins=50)
    plt.xlabel("伪标签置信度"); plt.ylabel("计数"); plt.title("无标注伪标签置信度分布（训练后）")
    plt.tight_layout(); plt.savefig(os.path.join(figs,"pseudo_conf_hist.png"), dpi=150); plt.show()

# ================== CLI ==================
def build_args():
    p = argparse.ArgumentParser("第三问：半监督目标域（自动种子 + FixMatch，模型/无监督双备份）", allow_abbrev=False)
    p.add_argument("--target_labeled_root",   type=str, default="./目标域有标注",
                   help="若存在 B/IR/OR/N 子文件夹则使用，否则自动种子")
    p.add_argument("--target_unlabeled_root", type=str, default="./目标域数据集",
                   help="A.mat~P.mat 等未标注 .mat 所在文件夹")
    p.add_argument("--out_dir",     type=str, default="./q3_out")
    # 对齐与采样
    p.add_argument("--fs_target",   type=int, default=32000)
    p.add_argument("--win_sec",     type=float, default=1.0)
    p.add_argument("--overlap",     type=float, default=0.5)
    p.add_argument("--zscore",      action="store_true")
    p.add_argument("--align_mode",  type=str, default="resample", choices=["resample","crop","pad"])
    # 特征提取器与种子模型
    p.add_argument("--fe_path",     type=str, default="./q2_out/fe_extractor.h5",
                   help="第二问训练的特征提取器（可选；若没有也可仅用频域特征做无监督）")
    p.add_argument("--seed_model",  type=str, default="",
                   help="用于自动种子的分类模型：如 q2 的 dann_model.h5 或 q1 的 best_model.h5；为空则跳过")
    p.add_argument("--seed_thr",    type=float, default=0.95,
                   help="自动种子（模型法）：高置信度阈值")
    p.add_argument("--seed_per_class", type=int, default=400,
                   help="自动种子：每类/每簇最多抽取数量")
    p.add_argument("--seed_total",  type=int, default=None,
                   help="自动种子：总上限（可选）")
    p.add_argument("--k_classes",   type=int, default=4,
                   help="KMeans 簇数（纯无标注时常取 4 对应 B/IR/OR/N）")
    # 训练
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--tau",         type=float, default=0.9, help="FixMatch 伪标签阈值")
    p.add_argument("--lambda_u",    type=float, default=1.0, help="一致性权重")
    # 强增强超参
    p.add_argument("--strong_jitter", type=float, default=0.03, help="强增强：高斯抖动幅度（Std.）")
    p.add_argument("--strong_scale",  type=float, default=0.10, help="强增强：整体幅值随机缩放范围")
    p.add_argument("--strong_drop",   type=float, default=0.10, help="强增强：随机丢点比例 [0,1)")
    # 可视化
    p.add_argument("--tsne_n",      type=int, default=2000)
    return p

if __name__ == "__main__":
    sanitize_argv()
    args = build_args().parse_args()
    train_fixmatch(args)
