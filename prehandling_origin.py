import os, sys, re, glob, math, json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import matplotlib
import matplotlib.pyplot as plt

# ============= 中文字体（自动） =============
def setup_chinese_font():
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
setup_chinese_font()

# ============= 基础工具 =============
def safe_mkdir(p): os.makedirs(p, exist_ok=True)

def sanitize_argv():
    keep = ("--data_root","--out_dir","--plot_seconds","--max_points_plot",
            "--zscore_norm","--save_psd_npz","--limit_files","--help")
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

def find_numeric_vars(md: Dict) -> List[str]:
    return [k for k,v in md.items() if not k.startswith("__") and isinstance(v,np.ndarray) and np.issubdtype(v.dtype, np.number)]

def squeeze1d(x: np.ndarray) -> np.ndarray:
    x = np.array(x); 
    if np.iscomplexobj(x): x = np.real(x)
    x = np.squeeze(x)
    if x.ndim==1: return x
    if x.ndim==2:
        if 1 in x.shape and max(x.shape)>1: return x.reshape(-1)
        return (x[:,0] if x.shape[0]>=x.shape[1] else x[0,:]).reshape(-1)
    return x.reshape(-1)

def load_mat(file_path: str) -> Dict[str,np.ndarray]:
    try:
        md = sio.loadmat(file_path)
        return {k:v for k,v in md.items() if not k.startswith("__") and isinstance(v,np.ndarray) and np.issubdtype(v.dtype,np.number)}
    except Exception:
        try:
            import h5py
            out={}
            with h5py.File(file_path,"r") as f:
                def collect(name,obj):
                    if isinstance(obj,h5py.Dataset):
                        arr=np.array(obj)
                        if np.issubdtype(arr.dtype,np.number):
                            out[name.split("/")[-1]] = arr
                f.visititems(collect)
            return out
        except Exception as e:
            raise RuntimeError(f"读取MAT失败：{e}")

def compute_stats(sig: np.ndarray)->Dict[str,float]:
    if sig.size==0:
        return dict(均值=np.nan,标准差=np.nan,RMS=np.nan,峭度=np.nan,偏度=np.nan,峰值=np.nan,峰值因子=np.nan)
    mean=float(np.mean(sig)); std=float(np.std(sig)); rms=float(np.sqrt(np.mean(sig**2)))
    krt=float(kurtosis(sig,fisher=True,bias=False)); skw=float(skew(sig,bias=False))
    peak=float(np.max(np.abs(sig))); crest=float(peak/rms) if rms>0 else np.nan
    return dict(均值=mean,标准差=std,RMS=rms,峭度=krt,偏度=skw,峰值=peak,峰值因子=crest)

def maybe_downsample(sig: np.ndarray, fs: Optional[float], max_pts: Optional[int]):
    if not max_pts or sig.size<=max_pts: return sig, fs
    step=int(math.ceil(sig.size/max_pts))
    return sig[::step], (fs/step if fs else fs)

# ============= 源域命名解析 =============
# 支持：IR014_3.mat、B007_0.mat、B028_0_(1797rpm).mat、OR014_1.mat 等
FN_PAT = re.compile(
    r'^(?P<cls>IR|OR|B|N)[^\d_]*'
    r'(?P<size>\d{3})?'
    r'(?:_(?P<load>[0-3]))?'
    r'(?:.*?\((?P<rpm>\d+)\s*rpm\))?',
    re.IGNORECASE
)

def parse_filename(file_name: str) -> Dict[str, Optional[str]]:
    base = os.path.splitext(os.path.basename(file_name))[0]
    m = FN_PAT.match(base)
    out = dict(故障类别=None, 尺寸inch=None, 载荷=None, RPM=None)
    if m:
        cls = m.group("cls").upper()
        out["故障类别"] = {"IR":"内圈","OR":"外圈","B":"滚动体","N":"正常"}.get(cls, cls)
        size = m.group("size")
        out["尺寸inch"] = (float(size)/1000.0 if size else None)
        load = m.group("load"); out["载荷"] = (int(load) if load is not None else None)
        rpm = m.group("rpm");  out["RPM"] = (int(rpm) if rpm else None)
    return out

def parse_path_labels(path_parts: List[str]) -> Dict[str, Optional[str]]:
    # 从路径中识别采样率、位置、OR方位
    fs = None; pos=None; or_pos=None
    for p in path_parts:
        if "12khz" in p.lower(): fs=12000.0
        if "48khz" in p.lower(): fs=48000.0
        if re.search(r'\bDE\b', p, re.IGNORECASE): pos="驱动端(DE)"
        if re.search(r'\bFE\b', p, re.IGNORECASE): pos="风扇端(FE)"
        if re.search(r'\bBA\b', p, re.IGNORECASE): pos="基座(BA)"
        if re.search(r'normal', p, re.IGNORECASE): pos="正常位"
        if p.lower() in ("centered","opposite","orthogonal"):
            or_pos = {"centered":"6点位(Centered)","opposite":"12点位(Opposite)","orthogonal":"3点位(Orthogonal)"}[p.lower()]
    return dict(采样率=fs, 采样点位=pos, 外圈方位=or_pos)

# ============= 作图（中文 & 保存） =============
def save_time(sig, fs, title, path):
    plt.figure()
    if fs and fs>0: t=np.arange(len(sig))/fs; plt.plot(t,sig); plt.xlabel("时间 (秒)")
    else: plt.plot(sig); plt.xlabel("样本索引")
    plt.ylabel("幅值"); plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def save_psd(sig, fs, title, path, return_arrays=False):
    from scipy.signal import welch
    plt.figure()
    nper = min(len(sig),4096); nper = nper if nper>=8 else len(sig)
    if fs and fs>0:
        f,P = welch(sig, fs=fs, nperseg=nper); plt.xlabel("频率 (Hz)")
    else:
        f,P = welch(sig, fs=1.0, nperseg=nper); plt.xlabel("归一化频率")
    plt.semilogy(f,P); plt.ylabel("PSD"); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    return (f,P) if return_arrays else None

def save_spec(sig, fs, title, path):
    plt.figure()
    fs_use = fs if (fs and fs>0) else 1.0
    if len(sig)>=4096: N=1024
    elif len(sig)>=2048: N=512
    else: N=256
    plt.specgram(sig, NFFT=N, Fs=fs_use, noverlap=N//2)
    plt.xlabel("时间 (秒)" if (fs and fs>0) else "时间 (样本)")
    plt.ylabel("频率 (Hz)" if (fs and fs>0) else "频率 (cycles/sample)")
    plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

# ============= 综合性图（保存+显示） =============
def show_save_features_scatter(df, out_prefix):
    plt.figure()
    ok=df[['峭度','峰值因子']].replace([np.inf,-np.inf],np.nan).dropna()
    plt.scatter(ok['峭度'],ok['峰值因子']); plt.xlabel("峭度"); plt.ylabel("峰值因子"); plt.title("综合：峭度 vs 峰值因子")
    plt.tight_layout(); plt.savefig(out_prefix+"_kurtosis_crest.png",dpi=150); plt.show()

    plt.figure()
    ok=df[['RMS','峰值']].replace([np.inf,-np.inf],np.nan).dropna()
    plt.scatter(ok['RMS'],ok['峰值']); plt.xlabel("RMS"); plt.ylabel("峰值"); plt.title("综合：RMS vs 峰值")
    plt.tight_layout(); plt.savefig(out_prefix+"_rms_peak.png",dpi=150); plt.show()

def show_save_psd_overlay(psd_bank, out_png, max_curves=15):
    if not psd_bank: return
    plt.figure()
    for i,(label,f,P) in enumerate(psd_bank[:max_curves]):
        plt.semilogy(f,P,label=label)
    plt.xlabel("频率 (Hz)"); plt.ylabel("PSD"); plt.title("综合：多文件PSD叠加（前若干）")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.show()

def show_save_psd_heatmap(psd_bank, out_png, freq_max=None, n_bins=256):
    if not psd_bank: return
    f0=psd_bank[0][1]
    if freq_max is not None:
        mask=f0<=freq_max; f0=f0[mask]
    X=[]
    for _,f,P in psd_bank:
        if freq_max is not None:
            P=P[f<=freq_max]; f=f[f<=freq_max]
        if len(f)!=len(f0) or np.any(f!=f0):
            Pi=np.interp(f0,f,P)
        else:
            Pi=P
        X.append(np.log10(Pi+1e-12))
    X=np.vstack(X)
    if X.shape[1]>n_bins:
        idx=np.linspace(0,X.shape[1]-1,n_bins).astype(int)
        X=X[:,idx]; f_disp=f0[idx]
    else:
        f_disp=f0
    plt.figure()
    plt.imshow(X,aspect='auto',origin='lower',extent=[f_disp[0],f_disp[-1],0,X.shape[0]])
    plt.xlabel("频率 (Hz)"); plt.ylabel("文件索引"); plt.title("综合：PSD热力图（文件×频率）")
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.show()

def show_save_pca(df, out_png):
    cols=["均值","标准差","RMS","峭度","偏度","峰值","峰值因子"]
    X=df[cols].values.astype(float)
    mask=np.all(np.isfinite(X),axis=1); X=X[mask]
    if X.shape[0]<2: return
    X=(X-X.mean(0,keepdims=True))/(X.std(0,keepdims=True)+1e-12)
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    Z=X@Vt[:2].T
    plt.figure()
    plt.scatter(Z[:,0],Z[:,1]); plt.xlabel("主成分1"); plt.ylabel("主成分2"); plt.title("综合：统计特征PCA二维散点")
    plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.show()

# ============= 主流程 =============
def main_cli():
    sanitize_argv()
    # 只接受 --key=value
    args = {
        "data_root":"./源域数据集",
        "out_dir":"./源域输出",
        "plot_seconds":"3",
        "max_points_plot":"200000",
        "zscore_norm":"false",
        "save_psd_npz":"true",
        "limit_files":"0"
    }
    for a in sys.argv[1:]:
        if not a.startswith("--"): continue
        if "=" in a:
            k,v=a[2:].split("=",1); args[k]=v
        else:
            args[a[2:]]="true"

    def to_bool(s): return str(s).strip().lower() in ("1","true","yes","y","on")

    data_root=args.get("data_root","./源域数据集")
    out_dir  =args.get("out_dir","./源域输出")
    plot_seconds = float(args.get("plot_seconds","3")); plot_seconds = None if plot_seconds<=0 else plot_seconds
    max_points_plot = int(args.get("max_points_plot","200000")); max_points_plot = None if max_points_plot<=0 else max_points_plot
    zscore_norm = to_bool(args.get("zscore_norm","false"))
    save_psd_npz = to_bool(args.get("save_psd_npz","true"))
    limit_files = int(args.get("limit_files","0"))  # 调试时可限制最多处理N个文件

    # 目录
    plots_dir=os.path.join(out_dir,"plots"); safe_mkdir(plots_dir)
    meta_dir =os.path.join(out_dir,"meta");  safe_mkdir(meta_dir)
    npz_dir  =os.path.join(out_dir,"npz");   safe_mkdir(npz_dir)
    aggr_dir =os.path.join(out_dir,"aggregates"); safe_mkdir(aggr_dir)

    # 递归搜集所有 .mat
    mat_files = [p for p in glob.glob(os.path.join(data_root,"**","*.mat"), recursive=True)]
    if limit_files>0: mat_files = mat_files[:limit_files]
    if not mat_files:
        print(f"[WARN] 未在 {os.path.abspath(data_root)} 找到 .mat"); return

    print(f"[INFO] 共发现 {len(mat_files)} 个 .mat（递归）。")
    rows=[]; psd_bank=[]

    for idx,fp in enumerate(sorted(mat_files)):
        rel = os.path.relpath(fp, data_root)
        parts = [p for p in rel.replace("\\","/").split("/")[:-1] if p]
        labels_path = parse_path_labels(parts)
        labels_name = parse_filename(os.path.basename(fp))

        # 解析采样率优先级：路径>文件名括号RPM换算(无法)>数据内RPM(无法求fs)>未知
        fs = labels_path["采样率"]

        try:
            md = load_mat(fp)
            # 源域常见键：*DE_time/*FE_time/*BA_time/RPM
            sigs={}
            for k,v in md.items():
                kl=k.lower()
                if 'de' in kl and 'time' in kl: sigs['DE']=squeeze1d(v)
                elif 'fe' in kl and 'time' in kl: sigs['FE']=squeeze1d(v)
                elif 'ba' in kl and 'time' in kl: sigs['BA']=squeeze1d(v)
                elif 'rpm' in kl: rpm_arr=squeeze1d(v); labels_name["RPM"]=int(np.median(rpm_arr)) if rpm_arr.size else labels_name.get("RPM")
            if not sigs:  # 兜底：把所有数值变量都当作通道
                for k in find_numeric_vars(md): sigs[k]=squeeze1d(md[k])

            for ch_name, sig_raw in sigs.items():
                sig = sig_raw.astype(float)

                # 作图片段长度：按 plot_seconds 和 fs（若fs未知就全长）
                fs_use = fs
                if plot_seconds and (fs_use and fs_use>0):
                    sig = sig[: int(min(len(sig), plot_seconds*fs_use))]

                # 统计（原始片段）；绘图可选 z-score
                stats_base = sig.copy()
                if zscore_norm and sig.size>1:
                    m,s = float(np.mean(sig)), float(np.std(sig))
                    if s>0: sig=(sig-m)/s

                # 绘图降采样
                sig_plot, fs_plot = maybe_downsample(sig, fs_use, max_points_plot)

                # 保存单文件三图
                title_prefix = f"{labels_name.get('故障类别') or labels_path.get('采样点位') or ch_name}"
                pos = labels_path.get("采样点位") or ch_name
                orpos = labels_path.get("外圈方位")
                tag = "_".join([x for x in [
                    os.path.splitext(os.path.basename(fp))[0],
                    pos,
                    orpos if orpos else None
                ] if x])
                save_time(sig_plot, fs_plot, f"{title_prefix} 时域波形", os.path.join(plots_dir, f"{tag}_时域.png"))
                fP = save_psd(sig_plot, fs_plot, f"{title_prefix} PSD 功率谱密度", os.path.join(plots_dir, f"{tag}_PSD.png"), return_arrays=True)
                save_spec(sig_plot, fs_plot, f"{title_prefix} 频谱图（Spectrogram）", os.path.join(plots_dir, f"{tag}_谱图.png"))

                # 行汇总 & 单条JSON
                stats = compute_stats(stats_base)
                duration = (len(sig_plot)/fs_plot if (fs_plot and fs_plot>0) else np.nan)
                row = {
                    "相对路径": rel, "文件": os.path.basename(fp), "通道": ch_name,
                    "采样率": fs_plot if (fs_plot and fs_plot>0) else fs,
                    "时长(秒)": duration if (duration==duration) else np.nan,
                    "故障类别": labels_name.get("故障类别"),
                    "外圈方位": labels_path.get("外圈方位"),
                    "采样点位": labels_path.get("采样点位"),
                    "尺寸(inch)": labels_name.get("尺寸inch"),
                    "载荷": labels_name.get("载荷"),
                    "RPM": labels_name.get("RPM"),
                    **stats
                }
                rows.append(row)

                meta_path = os.path.join(meta_dir, f"{tag}.json")
                with open(meta_path,"w",encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False, indent=2)

                if fP is not None and fs_plot and fs_plot>0:
                    label = f"{os.path.basename(fp)}-{ch_name}"
                    psd_bank.append((label, fP[0], fP[1]))
                    # 可选保存PSD数值
                    if save_psd_npz:
                        np.savez(os.path.join(npz_dir, f"{tag}_psd.npz"), f=fP[0], Pxx=fP[1])

        except Exception as e:
            rows.append({"相对路径": rel, "文件": os.path.basename(fp), "通道": "",
                         "采样率": np.nan, "时长(秒)": np.nan, "故障类别": None,
                         "外圈方位": None, "采样点位": None, "尺寸(inch)": None,
                         "载荷": None, "RPM": None,
                         "均值": np.nan, "标准差": np.nan, "RMS": np.nan, "峭度": np.nan,
                         "偏度": np.nan, "峰值": np.nan, "峰值因子": np.nan,
                         "备注": f"ERROR: {e}"})
            print(f"[ERROR] {rel}: {e}")

        if (idx+1)%50==0:
            print(f"[INFO] 进度 {idx+1}/{len(mat_files)}")

    # 汇总表
    df=pd.DataFrame(rows)
    safe_mkdir(out_dir)
    df.to_csv(os.path.join(out_dir,"summary.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[OK] 汇总完成：{os.path.join(out_dir,'summary.csv')}")
    print(f"[OK] 单文件图片：{os.path.join(out_dir,'plots')}")
    print(f"[OK] 元数据JSON：{os.path.join(out_dir,'meta')}")
    print(f"[OK] PSD npz：   {os.path.join(out_dir,'npz')}")

    # 综合性图（保存+显示）
    aggr = os.path.join(out_dir,"aggregates"); safe_mkdir(aggr)
    show_save_features_scatter(df, os.path.join(aggr,"features_scatter"))
    show_save_psd_overlay(psd_bank, os.path.join(aggr,"psd_overlay.png"), max_curves=15)
    show_save_psd_heatmap(psd_bank, os.path.join(aggr,"psd_heatmap.png"), freq_max=None, n_bins=256)
    show_save_pca(df, os.path.join(aggr,"pca_scatter.png"))
    print(f"[OK] 综合性图：{aggr}（同时已显示）")

if __name__ == "__main__":
    main_cli()
