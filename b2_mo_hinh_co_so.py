import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
du_lieu_fold = []
for f_tr, f_val in skf.split(vb_tr, y_tr):
    tfidf_f = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_tf_f_tr = tfidf_f.fit_transform(vb_tr[f_tr])
    X_tf_f_val = tfidf_f.transform(vb_tr[f_val])

    scaler_f = MaxAbsScaler()
    X_hv_f_tr = csr_matrix(scaler_f.fit_transform(hv_tr[f_tr]))
    X_hv_f_val = csr_matrix(scaler_f.transform(hv_tr[f_val]))

    X_fl_f_tr = hstack([X_tf_f_tr, X_hv_f_tr])
    X_fl_f_val = hstack([X_tf_f_val, X_hv_f_val])
    du_lieu_fold.append((X_tf_f_tr, X_tf_f_val, X_fl_f_tr, X_fl_f_val, y_tr[f_tr], y_tr[f_val]))

print("Tap hoc:", len(y_tr), "mau (Tin that:", int((y_tr == 0).sum()), "| Tin gia:", int((y_tr == 1).sum()), ")")
print("Tap kiem tra:", len(y_te), "mau (Tin that:", int((y_te == 0).sum()), "| Tin gia:", int((y_te == 1).sum()), ")")
print("So chieu TF-IDF:", X_tfidf_tr.shape[1], "| So chieu tong:", X_full_tr.shape[1])
print()

mo_hinh = {
    "SVM kernel RBF (Mac dinh)": (SVC(kernel="rbf", C=1.0, random_state=17), True),
    "Hoi quy Logistic (Mac dinh)": (LogisticRegression(max_iter=2000, random_state=17), True),
    "Naive Bayes (TF-IDF)": (ComplementNB(), False),
    "Random Forest (Can bang lop)": (RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=17, n_jobs=-1), True),
    "Hoi quy Logistic (Can bang lop)": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17), True),
    "SVM kernel RBF (Can bang lop)": (SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=17), True),
    "SVM tuyen tinh (Chi TF-IDF)": (LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=17), False),
    "SVM tuyen tinh (TF-IDF + Hanh vi)": (LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=17), True)
}

if co_lgbm:
    mo_hinh["LightGBM (Can bang lop)"] = (LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=17, verbose=-1), True)

print(f"{'Mo hinh':35s} {'CV_F1':>10s} {'CV_Std':>8s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
ket_qua = []
tot_nhat_cv_f1 = -1.0
mh_tot_nhat = None
du_doan_tot_nhat = None
ten_tot_nhat = ""
dung_hv_tot_nhat = True

for ten, (mh_goc, dung_hv) in mo_hinh.items():
    ds_f1_cv = []
    for X_tf_f_tr, X_tf_f_val, X_fl_f_tr, X_fl_f_val, y_f_tr, y_f_val in du_lieu_fold:
        mh_cv = clone(mh_goc)
        X_f_tr = X_fl_f_tr if dung_hv else X_tf_f_tr
        X_f_val = X_fl_f_val if dung_hv else X_tf_f_val
        mh_cv.fit(X_f_tr, y_f_tr)
        du_doan_f = mh_cv.predict(X_f_val)
        ds_f1_cv.append(f1_score(y_f_val, du_doan_f, zero_division=0))

    cv_f1_mean = float(np.mean(ds_f1_cv))
    cv_f1_std = float(np.std(ds_f1_cv))

    mh = clone(mh_goc)
    X_h = X_full_tr if dung_hv else X_tfidf_tr
    X_k = X_full_te if dung_hv else X_tfidf_te
    mh.fit(X_h, y_tr)
    du_doan = mh.predict(X_k)
    acc = accuracy_score(y_te, du_doan)
    pre = precision_score(y_te, du_doan, zero_division=0)
    rec = recall_score(y_te, du_doan, zero_division=0)
    f1 = f1_score(y_te, du_doan, zero_division=0)
    print(f"{ten:35s} {cv_f1_mean:10.4f} {cv_f1_std:8.4f} {acc:10.4f} {pre:10.4f} {rec:10.4f} {f1:10.4f}")
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
        dung_hv_tot_nhat = dung_hv

print()
print("Mo hinh co so tot nhat theo 5-fold CV F1:", ten_tot_nhat)
cm = confusion_matrix(y_te, du_doan_tot_nhat)
print("Bang nham lan (Hang: That, Cot: Doan):")
print(cm)

pd.DataFrame(ket_qua).to_csv("data/ket_qua_co_so.csv", index=False)

lr_xai = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
lr_xai.fit(X_tfidf_tr, y_tr)
with open("data/mo_hinh_co_so.pkl", "wb") as f:
    pickle.dump({
        "ten": ten_tot_nhat,
        "dung_hanh_vi": dung_hv_tot_nhat,
        "tfidf": tfidf,
        "scaler": scaler,
        "model": mh_tot_nhat,
        "model_xai": lr_xai
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
    plt.savefig("data/confusion_matrix_co_so.png")