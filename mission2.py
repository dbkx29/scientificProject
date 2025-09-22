import os, re, glob, math, random, argparse, sys
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def sanitize_argv():
    """吞掉 Notebook 注入的 -f <kernel.json> 以及未知参数，避免 argparse 误判。"""
    keep = ("--data_root","--out_dir","--fs","--win_sec","--overlap",
            "--batch_size","--epochs","--lr","--val_ratio","--zscore","--help")
    argv = sys.argv[:]; out=[argv[0]]; i=1; skip=False
    while i < len(argv):
        a=argv[i]
        if skip: skip=False; i+=1; continue
        if a == "-f": skip=True; i+=1; continue
        if any(a.startswith(k) for k in keep):
            out.append(a)
            if ("=" not in a) and i+1<len(argv) and not argv[i+1].startswith("-"):
                out.append(argv[i+1]); i+=1
        i+=1
    sys.argv = out

# ================== 读取 .mat ==================
def load_mat(file_path: str):
    md = sio.loadmat(file_path)
    return {k:v for k,v in md.items() if not k.startswith("__") and isinstance(v, np.ndarray)}

def squeeze1d(x: np.ndarray):
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

# ================== 滑窗 & 标准化 ==================
def sliding_windows(x, win, hop):
    if x.size < win: return np.empty((0,win))
    n = 1 + (x.size - win)//hop
    out = np.zeros((n,win), dtype=np.float32)
    for i in range(n): out[i,:] = x[i*hop:i*hop+win]
    return out

def zscore_norm(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-12
    return (X-m)/s

# ================== 类别解析 ==================
FN_PAT = re.compile(r'^(?P<cls>IR|OR|B|N)', re.IGNORECASE)
def parse_cls_from_name(fname: str):
    m = FN_PAT.match(os.path.basename(fname))
    if m: return m.group("cls").upper()
    if "normal" in fname.lower(): return "N"
    return "UNK"

# ================== 数据构建 ==================
def build_dataset(data_root, fs_default=12000, win_sec=1.0, overlap=0.5, zscore=True, val_ratio=0.2):
    mat_files = sorted(glob.glob(os.path.join(data_root, "**", "*.mat"), recursive=True))
    if not mat_files:
        raise FileNotFoundError(f"未在 {os.path.abspath(data_root)} 找到 .mat 文件，请确认 data_root 路径。")
    Xs, ys = [], []
    classes = set()
    for fp in mat_files:
        md = load_mat(fp)
        _, sig = choose_signal(md)
        fs = fs_default
        win = int(win_sec*fs); hop = max(1, int(win*(1-overlap)))
        segs = sliding_windows(sig, win, hop)
        if segs.shape[0]==0: continue
        if zscore: segs = zscore_norm(segs)
        cls = parse_cls_from_name(fp)
        if cls=="UNK": continue
        Xs.append(segs)
        ys.append(np.full(segs.shape[0], cls))
        classes.add(cls)
    classes = sorted(classes)
    cls2id = {c:i for i,c in enumerate(classes)}
    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys)
    yid = np.array([cls2id[c] for c in y], dtype=np.int64)

    idx = np.arange(len(X)); np.random.shuffle(idx)
    n_val = max(1, int(len(X)*val_ratio))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    return (X[trn_idx], yid[trn_idx]), (X[val_idx], yid[val_idx]), cls2id

# ================== ResNet1D (Keras) ==================
def resnet1d_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv1D(filters, 7, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(filters, 7, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride!=1 or shortcut.shape[-1]!=filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut]); x = layers.ReLU()(x)
    return x

def build_resnet1d(input_len, n_classes):
    inp = layers.Input(shape=(input_len,1))
    x = layers.Conv1D(32,7,strides=2,padding="same",use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool1D(3,strides=2,padding="same")(x)
    x = resnet1d_block(x,64,1)
    x = resnet1d_block(x,128,2)
    x = resnet1d_block(x,256,2)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out)

# ================== 训练/评估主流程 ==================
def run_q1(data_root="./源域数据集", out_dir="./q1_tf_out",
           fs=12000, win_sec=1.0, overlap=0.5, batch_size=128,
           epochs=20, lr=1e-3, val_ratio=0.2, zscore=True):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)

    (Xtr,ytr),(Xva,yva),cls2id = build_dataset(
        data_root=data_root, fs_default=fs, win_sec=win_sec, overlap=overlap,
        zscore=zscore, val_ratio=val_ratio
    )
    id2cls = {i:c for c,i in cls2id.items()}
    print(f"[数据] 训练: {Xtr.shape}, 验证: {Xva.shape}, 类别映射: {cls2id}")

    def to_ds(X,y,bs,shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X[:,:,None], y))
        if shuffle: ds = ds.shuffle(buffer_size=len(X))
        return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

    tr_ds = to_ds(Xtr,ytr,batch_size,True)
    va_ds = to_ds(Xva,yva,batch_size,False)

    model = build_resnet1d(Xtr.shape[1], len(cls2id))
    model.compile(optimizer=optimizers.Adam(lr), loss=losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    hist = model.fit(tr_ds, validation_data=va_ds, epochs=epochs)

    # 曲线（每图单独，未指定颜色）
    plt.figure(); plt.plot(hist.history["loss"]); plt.plot(hist.history["val_loss"])
    plt.xlabel("轮次"); plt.ylabel("Loss"); plt.title("训练/验证 Loss"); plt.tight_layout(); plt.show()
    plt.figure(); plt.plot(hist.history["accuracy"]); plt.plot(hist.history["val_accuracy"])
    plt.xlabel("轮次"); plt.ylabel("Acc");  plt.title("训练/验证 Acc");  plt.tight_layout(); plt.show()

    # 验证集评估
    y_pred = model.predict(Xva[:,:,None]).argmax(1)
    print("\n分类报告：")
    print(classification_report(yva, y_pred, target_names=[id2cls[i] for i in range(len(id2cls))]))
    cm = confusion_matrix(yva, y_pred)
    plt.figure(); plt.imshow(cm, aspect='auto'); plt.title("验证集混淆矩阵")
    plt.xlabel("预测类别"); plt.ylabel("真实类别")
    plt.xticks(range(len(id2cls)), [id2cls[i] for i in range(len(id2cls))], rotation=45)
    plt.yticks(range(len(id2cls)), [id2cls[i] for i in range(len(id2cls))])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.show()

    model.save(os.path.join(out_dir,"best_model.h5"))
    print(f"[输出] 模型已保存：{os.path.join(out_dir,'best_model.h5')}")
    return {"model":model, "cls2id":cls2id, "id2cls":id2cls}

# ================== CLI（可选） ==================
def build_args():
    p = argparse.ArgumentParser("第一问：源域故障智能识别（TensorFlow ResNet1D）", allow_abbrev=False)
    p.add_argument("--data_root", type=str, default="./源域数据集", help="源域数据根目录（默认 ./源域数据集）")
    p.add_argument("--out_dir",  type=str, default="./q1_tf_out")
    p.add_argument("--fs",       type=int, default=12000)
    p.add_argument("--win_sec",  type=float, default=1.0)
    p.add_argument("--overlap",  type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--val_ratio",type=float, default=0.2)
    p.add_argument("--zscore",   action="store_true")
    return p

if __name__ == "__main__":
    sanitize_argv()
    parser = build_args()
    # Notebook 环境下可能还有未知参数，这里再保险 parse_known_args
    args, _ = parser.parse_known_args()
    run_q1(data_root=args.data_root, out_dir=args.out_dir, fs=args.fs,
           win_sec=args.win_sec, overlap=args.overlap, batch_size=args.batch_size,
           epochs=args.epochs, lr=args.lr, val_ratio=args.val_ratio, zscore=bool(args.zscore))
