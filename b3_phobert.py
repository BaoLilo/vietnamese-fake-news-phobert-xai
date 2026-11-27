import os
import ssl
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import matplotlib.pyplot as plt
    co_plt = True
except Exception:
    co_plt = False

df = pd.read_csv("data/tin_tuc_sach.csv")
df["van_ban"] = df["van_ban"].fillna("").astype(str)
df["van_ban_goc"] = df["van_ban_goc"].fillna("").astype(str)

cot_hanh_vi = ["do_dai", "so_cham_than", "so_dau_hoi", "ty_le_hoa"]
for c in cot_hanh_vi:
    if c not in df.columns:
        df[c] = 0.0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

y = df["nhan"].astype(int).values
idx_tr, idx_te = train_test_split(
    np.arange(len(df)), test_size=0.25, random_state=17, stratify=y
)

vb_tr = df["van_ban"].iloc[idx_tr].tolist()
vb_te = df["van_ban"].iloc[idx_te].tolist()
vb_tr_arr = np.array(vb_tr)
hv_tr = df[cot_hanh_vi].iloc[idx_tr].values
hv_te = df[cot_hanh_vi].iloc[idx_te].values
y_tr = y[idx_tr]
y_te = y[idx_te]

with open("data/mo_hinh_co_so.pkl", "rb") as f:
    du_lieu_cs = pickle.load(f)

tfidf = du_lieu_cs["tfidf"]
X_tfidf_tr = tfidf.transform(vb_tr)
X_tfidf_te = tfidf.transform(vb_te)

scaler_hv = MaxAbsScaler()
X_hv_tr = csr_matrix(scaler_hv.fit_transform(hv_tr))
X_hv_te = csr_matrix(scaler_hv.transform(hv_te))

dau_du_lieu = hashlib.md5("\n".join(vb_tr + vb_te).encode("utf-8")).hexdigest()
duong_dan_vec = "data/phobert_vectors.npz"
co_phobert = False
da_nap_cache = False

if os.path.exists(duong_dan_vec):
    data_vec = np.load(duong_dan_vec, allow_pickle=False)
    if "dau_du_lieu" in data_vec and str(data_vec["dau_du_lieu"].item()) == dau_du_lieu:
        X_emb_tr = data_vec["X_tr"]
        X_emb_te = data_vec["X_te"]
        if X_emb_tr.shape[0] == len(vb_tr) and X_emb_te.shape[0] == len(vb_te):
            co_phobert = True
            da_nap_cache = True
            print("Da nap vector PhoBERT khop du lieu tu data/phobert_vectors.npz")

if not da_nap_cache:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        thiet_bi = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print("Dang tai PhoBERT (vinai/phobert-base-v2) tren thiet bi:", thiet_bi)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(thiet_bi)
        phobert.eval()

        def trich_xuat_batch(danh_sach_vb, co_lo=64):
            ket_qua_vec = []
            tong_lo = (len(danh_sach_vb) + co_lo - 1) // co_lo
            for i in range(0, len(danh_sach_vb), co_lo):
                lo_vb = danh_sach_vb[i:i + co_lo]
                ma_hoa = tokenizer(
                    lo_vb, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(thiet_bi)
                with torch.no_grad():
                    dau_ra = phobert(**ma_hoa)
                    mask = ma_hoa["attention_mask"].unsqueeze(-1).expand(dau_ra.last_hidden_state.size()).float()
                    tong_vec = torch.sum(dau_ra.last_hidden_state * mask, 1)
                    tong_mask = torch.clamp(mask.sum(1), min=1e-9)
                    vec_tb = (tong_vec / tong_mask).cpu().numpy()
                    ket_qua_vec.append(vec_tb)
                so_lo = (i // co_lo) + 1
                if so_lo % 15 == 0 or so_lo == tong_lo:
                    print(f"  Da xu ly {so_lo}/{tong_lo} batch...")
            return np.vstack(ket_qua_vec)

        print("Trich xuat vector tap hoc:")
        X_emb_tr = trich_xuat_batch(vb_tr)
        print("Trich xuat vector tap kiem tra:")
        X_emb_te = trich_xuat_batch(vb_te)
        np.savez_compressed(duong_dan_vec, X_tr=X_emb_tr, X_te=X_emb_te, dau_du_lieu=np.array(dau_du_lieu))
        co_phobert = True
    except Exception as e:
        print("Chuyen sang che do SVD du phong do:", str(e))
        svd = TruncatedSVD(n_components=256, random_state=17)
        X_emb_tr = svd.fit_transform(X_tfidf_tr)
        X_emb_te = svd.transform(X_tfidf_te)

scaler_emb = StandardScaler()
X_emb_tr_s = scaler_emb.fit_transform(X_emb_tr)
X_emb_te_s = scaler_emb.transform(X_emb_te)

X_ket_hop_tr = hstack([X_tfidf_tr, X_hv_tr, csr_matrix(X_emb_tr_s * 0.2)])
X_ket_hop_te = hstack([X_tfidf_te, X_hv_te, csr_matrix(X_emb_te_s * 0.2)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
du_lieu_fold = []
for f_tr, f_val in skf.split(X_emb_tr, y_tr):
    sc_emb_f = StandardScaler()
    X_emb_f_tr = sc_emb_f.fit_transform(X_emb_tr[f_tr])
    X_emb_f_val = sc_emb_f.transform(X_emb_tr[f_val])

    tfidf_f = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_tf_f_tr = tfidf_f.fit_transform(vb_tr_arr[f_tr])
    X_tf_f_val = tfidf_f.transform(vb_tr_arr[f_val])

    sc_hv_f = MaxAbsScaler()
    X_hv_f_tr = csr_matrix(sc_hv_f.fit_transform(hv_tr[f_tr]))
    X_hv_f_val = csr_matrix(sc_hv_f.transform(hv_tr[f_val]))

    X_kh_f_tr = hstack([X_tf_f_tr, X_hv_f_tr, csr_matrix(X_emb_f_tr * 0.2)])
    X_kh_f_val = hstack([X_tf_f_val, X_hv_f_val, csr_matrix(X_emb_f_val * 0.2)])
    du_lieu_fold.append((X_emb_f_tr, X_emb_f_val, X_kh_f_tr, X_kh_f_val, y_tr[f_tr], y_tr[f_val]))

ten_goc = "PhoBERT" if co_phobert else "SVD-256"
print()
print("So chieu vector ngu nghia:", X_emb_tr.shape[1])
print()

mo_hinh_sota = {
    f"{ten_goc} + Hoi quy Logistic": (LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=17), False),
    f"{ten_goc} + SVM kernel RBF": (SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=17), False),
    f"{ten_goc} + MLP (Deep Learning)": (MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=17), False),
    f"{ten_goc} + TF-IDF + Hanh vi (Lai ghep)": (LinearSVC(C=0.3, class_weight="balanced", max_iter=5000, random_state=17), True)
}

print(f"{'Mo hinh SOTA':40s} {'CV_F1':>10s} {'CV_Std':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
ket_qua = []
tot_nhat_cv_f1 = -1.0
mh_tot_nhat = None
du_doan_tot_nhat = None
ten_tot_nhat = ""
la_lai_ghep_tot_nhat = False

for ten, (mh_goc, la_lai_ghep) in mo_hinh_sota.items():
    ds_f1_cv = []
    for X_emb_f_tr, X_emb_f_val, X_kh_f_tr, X_kh_f_val, y_f_tr, y_f_val in du_lieu_fold:
        mh_cv = clone(mh_goc)
        X_f_tr = X_kh_f_tr if la_lai_ghep else X_emb_f_tr
        X_f_val = X_kh_f_val if la_lai_ghep else X_emb_f_val
        mh_cv.fit(X_f_tr, y_f_tr)
        du_doan_f = mh_cv.predict(X_f_val)
        ds_f1_cv.append(f1_score(y_f_val, du_doan_f, zero_division=0))

    cv_f1_mean = float(np.mean(ds_f1_cv))
    cv_f1_std = float(np.std(ds_f1_cv))

    mh = clone(mh_goc)
    X_h = X_ket_hop_tr if la_lai_ghep else X_emb_tr_s
    X_k = X_ket_hop_te if la_lai_ghep else X_emb_te_s
    mh.fit(X_h, y_tr)
    du_doan = mh.predict(X_k)
    acc = accuracy_score(y_te, du_doan)
    pre = precision_score(y_te, du_doan, zero_division=0)
    rec = recall_score(y_te, du_doan, zero_division=0)
    f1 = f1_score(y_te, du_doan, zero_division=0)
    print(f"{ten:40s} {cv_f1_mean:10.4f} {cv_f1_std:8.4f} {acc:10.4f} {pre:10.4f} {rec:10.4f} {f1:10.4f}")
    ket_qua.append({
        "mo_hinh": ten,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1
    })
    if cv_f1_mean > tot_nhat_cv_f1:
        tot_nhat_cv_f1 = cv_f1_mean
        mh_tot_nhat = mh
        du_doan_tot_nhat = du_doan
        ten_tot_nhat = ten
        la_lai_ghep_tot_nhat = la_lai_ghep

print()
print("Mo hinh SOTA tot nhat theo 5-fold CV F1:", ten_tot_nhat)
cm = confusion_matrix(y_te, du_doan_tot_nhat)
print("Bang nham lan (Hang: That, Cot: Doan):")
print(cm)

df_te = df.iloc[idx_te].copy().reset_index(drop=True)
df_te["du_doan"] = du_doan_tot_nhat
df_te["so_tu_goc"] = df_te["van_ban_goc"].str.split().str.len()
df_te["tren_128_tu"] = (df_te["so_tu_goc"] > 128).astype(int)

nhom_kq = []
for nh_thuc, nh_doan, ten_nhom in [
    (0, 0, "TN (Tin that dung)"),
    (0, 1, "FP (Tin that bao nham)"),
    (1, 0, "FN (Tin gia bo sot)"),
    (1, 1, "TP (Tin gia dung)")
]:
    con = df_te[(df_te["nhan"] == nh_thuc) & (df_te["du_doan"] == nh_doan)]
    if len(con) > 0:
        nhom_kq.append({
            "nhom": ten_nhom,
            "so_mau": int(len(con)),
            "so_tu_tb": round(float(con["so_tu_goc"].mean()), 2),
            "so_tu_tv": round(float(con["so_tu_goc"].median()), 2),
            "ty_le_tren_128_tu_%": round(float(con["tren_128_tu"].mean() * 100), 2),
            "cham_than_tb": round(float(con["so_cham_than"].mean()), 4),
            "dau_hoi_tb": round(float(con["so_dau_hoi"].mean()), 4),
            "ty_le_hoa_tb": round(float(con["ty_le_hoa"].mean()), 4)
        })

df_loi = pd.DataFrame(nhom_kq)
df_loi.to_csv("data/phan_tich_loi_sota.csv", index=False)
print()
print("Thong ke phan tich loi tren tap kiem tra:")
print(df_loi.to_string(index=False))

pd.DataFrame(ket_qua).to_csv("data/ket_qua_sota.csv", index=False)

with open("data/mo_hinh_sota.pkl", "wb") as f:
    pickle.dump({
        "ten": ten_tot_nhat,
        "la_lai_ghep": la_lai_ghep_tot_nhat,
        "model": mh_tot_nhat,
        "scaler_emb": scaler_emb,
        "scaler_hv": scaler_hv,
        "co_phobert": co_phobert
    }, f)

if co_plt:
    plt.figure(figsize=(5.5, 4.2), dpi=200)
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {ten_tot_nhat}", fontsize=10)
    plt.xlabel("Du doan")
    plt.ylabel("Thuc te")
    plt.xticks([0, 1], ["Tin that (0)", "Tin gia (1)"])
    plt.yticks([0, 1], ["Tin that (0)", "Tin gia (1)"])
    nguong = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            mau_chu = "white" if cm[i, j] > nguong else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=mau_chu, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_sota.png")