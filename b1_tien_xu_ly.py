import os
import re
import ssl
import unicodedata
import urllib.request
import numpy as np
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

try:
    from pyvi import ViTokenizer
    co_pyvi = True
except Exception:
    co_pyvi = False

os.makedirs("data", exist_ok=True)
duong_dan_goc = "data/tin_tuc_goc.csv"
duong_dan_sach = "data/tin_tuc_sach.csv"

danh_sach_link = [
    "https://raw.githubusercontent.com/PoLsss/Read_time_Vietnamese_FakeNews_Detection/main/Data/train.csv",
    "https://raw.githubusercontent.com/PoLsss/Read_time_Vietnamese_FakeNews_Detection/master/Data/train.csv",
    "https://raw.githubusercontent.com/thanhhocse96/vfnd-vietnamese-fake-news-datasets/master/CSV/vn_news_223_tdlfr.csv"
]

if not os.path.exists(duong_dan_goc):
    da_tai = False
    for link in danh_sach_link:
        try:
            urllib.request.urlretrieve(link, duong_dan_goc)
            da_tai = True
            break
        except Exception:
            continue
    if not da_tai:
        url_du_phong = "https://huggingface.co/datasets/tranthaihoa/vifactcheck/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
        df_dp = pd.read_parquet(url_du_phong)
        cot_vb_dp = [c for c in df_dp.columns if c.lower() in ["statement", "context", "text", "post_message"]][0]
        cot_nh_dp = [c for c in df_dp.columns if "label" in c.lower()][0]
        df_dp = df_dp[df_dp[cot_nh_dp].isin([0, 1])][[cot_vb_dp, cot_nh_dp]]
        df_dp.columns = ["post_message", "label"]
        df_dp.to_csv(duong_dan_goc, index=False)

df = pd.read_csv(duong_dan_goc, on_bad_lines="skip")

cac_cot_vb = [c for c in df.columns if c.lower() in ["post_message", "text", "statement", "content"]]
cot_van_ban = cac_cot_vb[0] if len(cac_cot_vb) > 0 else df.columns[0]

cac_cot_nh = [c for c in df.columns if "label" in c.lower()]
cot_nhan = cac_cot_nh[0] if len(cac_cot_nh) > 0 else df.columns[-1]

df = df[[cot_van_ban, cot_nhan]].dropna()
df.columns = ["van_ban_goc", "nhan"]
df["van_ban_goc"] = df["van_ban_goc"].astype(str)
df["nhan"] = pd.to_numeric(df["nhan"], errors="coerce")
df = df[df["nhan"].isin([0, 1])].copy()
df["nhan"] = df["nhan"].astype(int)
df = df[df["van_ban_goc"].str.strip().str.len() > 10].reset_index(drop=True)

def tinh_ty_le_hoa(s):
    chu_cai = [c for c in s if c.isalpha()]
    if len(chu_cai) == 0:
        return 0.0
    return sum(1 for c in chu_cai if c.isupper()) / len(chu_cai)

def lam_sach(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    if co_pyvi and len(s) > 0:
        s = ViTokenizer.tokenize(s)
    return s

df["do_dai"] = df["van_ban_goc"].str.len()
df["so_cham_than"] = df["van_ban_goc"].str.count("!")
df["so_dau_hoi"] = df["van_ban_goc"].str.count(r"\?")
df["ty_le_hoa"] = df["van_ban_goc"].apply(tinh_ty_le_hoa)
df["van_ban"] = df["van_ban_goc"].apply(lam_sach)
df = df[df["van_ban"].str.len() > 5].reset_index(drop=True)

df.to_csv(duong_dan_sach, index=False)

print("Tong so mau sau khi lam sach:", len(df))
print("Phan bo nhan (0: Tin that, 1: Tin gia):")
print(df["nhan"].value_counts().to_string())
print("Che do tach tu pyvi:", co_pyvi)
print("5 dong dau tien:")
print(df[["van_ban", "do_dai", "so_cham_than", "ty_le_hoa", "nhan"]].head().to_string())