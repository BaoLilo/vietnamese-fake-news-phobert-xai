import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from lightgbm import LGBMClassifier
    co_lgbm = True
except Exception:
    co_lgbm = False

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

vb_tr = df["van_ban"].iloc[idx_tr].values
vb_te = df["van_ban"].iloc[idx_te].values
hv_tr = df[cot_hanh_vi].iloc[idx_tr].values
hv_te = df[cot_hanh_vi].iloc[idx_te].values
y_tr = y[idx_tr]
y_te = y[idx_te]

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_tfidf_tr = tfidf.fit_transform(vb_tr)
X_tfidf_te = tfidf.transform(vb_te)

scaler = MaxAbsScaler()
X_hv_tr = csr_matrix(scaler.fit_transform(hv_tr))
X_hv_te = csr_matrix(scaler.transform(hv_te))

X_full_tr = hstack([X_tfidf_tr, X_hv_tr])
X_full_te = hstack([X_tfidf_te, X_hv_te])

print("Tap hoc:", len(y_tr), "mau | Tap kiem tra:", len(y_te), "mau")
print("So chieu TF-IDF:", X_tfidf_tr.shape[1], "| So chieu tong:", X_full_tr.shape[1])
print()

mo_hinh = {
    "Naive Bayes (TF-IDF)": (ComplementNB(), X_tfidf_tr, X_tfidf_te),
    "Hoi quy Logistic": (LogisticRegression(max_iter=2000, random_state=17), X_full_tr, X_full_te),
    "Hoi quy Logistic (Can bang lop)": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17), X_full_tr, X_full_te),
    "Random Forest": (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=17, n_jobs=-1), X_full_tr, X_full_te),
    "SVM tuyen tinh (Chi TF-IDF)": (LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=17), X_tfidf_tr, X_tfidf_te),
    "SVM tuyen tinh (TF-IDF + Hanh vi)": (LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=17), X_full_tr, X_full_te),
    "SVM kernel RBF": (SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=17), X_full_tr, X_full_te)
}

if co_lgbm:
    mo_hinh["LightGBM"] = (LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=17, verbose=-1), X_full_tr, X_full_te)

print(f"{'Mo hinh':35s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
ket_qua = []
tot_nhat_f1 = -1.0
mh_tot_nhat = None
du_doan_tot_nhat = None
ten_tot_nhat = ""

for ten, (mh, X_h, X_k) in mo_hinh.items():
    mh.fit(X_h, y_tr)
    du_doan = mh.predict(X_k)
    acc = accuracy_score(y_te, du_doan)
    pre = precision_score(y_te, du_doan, zero_division=0)
    rec = recall_score(y_te, du_doan, zero_division=0)
    f1 = f1_score(y_te, du_doan, zero_division=0)
    print(f"{ten:35s} {acc:10.4f} {pre:10.4f} {rec:10.4f} {f1:10.4f}")
    ket_qua.append({"mo_hinh": ten, "accuracy": acc, "precision": pre, "recall": rec, "f1": f1})
    if f1 > tot_nhat_f1:
        tot_nhat_f1 = f1
        mh_tot_nhat = mh
        du_doan_tot_nhat = du_doan
        ten_tot_nhat = ten

print()
print("Mo hinh co so tot nhat theo F1:", ten_tot_nhat)
cm = confusion_matrix(y_te, du_doan_tot_nhat)
print("Bang nham lan (Hang: That, Cot: Doan):")
print(cm)

pd.DataFrame(ket_qua).to_csv("data/ket_qua_co_so.csv", index=False)

svm_xai = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
svm_xai.fit(X_tfidf_tr, y_tr)
with open("data/mo_hinh_co_so.pkl", "wb") as f:
    pickle.dump({"tfidf": tfidf, "scaler": scaler, "model": mh_tot_nhat, "model_xai": svm_xai}, f)

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
    plt.savefig("data/confusion_matrix_co_so.png")