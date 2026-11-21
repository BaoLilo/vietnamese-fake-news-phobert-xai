import os
import ssl
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
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

duong_dan_vec = "data/phobert_vectors.npz"
co_phobert = False

if os.path.exists(duong_dan_vec):
    data_vec = np.load(duong_dan_vec)
    X_emb_tr = data_vec["X_tr"]
    X_emb_te = data_vec["X_te"]
    co_phobert = True
    print("Da nap vector PhoBERT co san tu data/phobert_vectors.npz")
else:
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
        np.savez_compressed(duong_dan_vec, X_tr=X_emb_tr, X_te=X_emb_te)
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

ten_goc = "PhoBERT" if co_phobert else "SVD-256"
print()
print("So chieu vector ngữ nghĩa:", X_emb_tr.shape[1])
print()

mo_hinh_sota = {
    f"{ten_goc} + Hoi quy Logistic": (LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=17), X_emb_tr_s, X_emb_te_s),
    f"{ten_goc} + SVM kernel RBF": (SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=17), X_emb_tr_s, X_emb_te_s),
    f"{ten_goc} + MLP (Deep Learning)": (MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=17), X_emb_tr_s, X_emb_te_s),
    f"{ten_goc} + TF-IDF + Hanh vi (Lai ghep)": (LinearSVC(C=0.3, class_weight="balanced", max_iter=5000, random_state=17), X_ket_hop_tr, X_ket_hop_te)
}

print(f"{'Mo hinh SOTA':40s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
ket_qua = []
tot_nhat_f1 = -1.0
mh_tot_nhat = None
du_doan_tot_nhat = None
ten_tot_nhat = ""

for ten, (mh, X_h, X_k) in mo_hinh_sota.items():
    mh.fit(X_h, y_tr)
    du_doan = mh.predict(X_k)
    acc = accuracy_score(y_te, du_doan)
    pre = precision_score(y_te, du_doan, zero_division=0)
    rec = recall_score(y_te, du_doan, zero_division=0)
    f1 = f1_score(y_te, du_doan, zero_division=0)
    print(f"{ten:40s} {acc:10.4f} {pre:10.4f} {rec:10.4f} {f1:10.4f}")
    ket_qua.append({"mo_hinh": ten, "accuracy": acc, "precision": pre, "recall": rec, "f1": f1})
    if f1 > tot_nhat_f1:
        tot_nhat_f1 = f1
        mh_tot_nhat = mh
        du_doan_tot_nhat = du_doan
        ten_tot_nhat = ten

print()
print("Mo hinh SOTA tot nhat theo F1:", ten_tot_nhat)
cm = confusion_matrix(y_te, du_doan_tot_nhat)
print("Bang nham lan (Hang: That, Cot: Doan):")
print(cm)

pd.DataFrame(ket_qua).to_csv("data/ket_qua_sota.csv", index=False)

with open("data/mo_hinh_sota.pkl", "wb") as f:
    pickle.dump({
        "ten": ten_tot_nhat,
        "model": mh_tot_nhat,
        "scaler_emb": scaler_emb,
        "scaler_hv": scaler_hv,
        "co_phobert": co_phobert
    }, f)

if co_plt:
    plt.figure(figsize=(5, 4), dpi=200)
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {ten_tot_nhat}")
    plt.xlabel("Du doan")
    plt.ylabel("Thuc te")
    plt.xticks([0, 1], ["Tin that (0)", "Tin gia (1)"])
    plt.yticks([0, 1], ["Tin that (0)", "Tin gia (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_sota.png")