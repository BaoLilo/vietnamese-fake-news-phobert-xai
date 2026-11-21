import os
import re
import ssl
import json
import pickle
import unicodedata
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from pyvi import ViTokenizer
    co_pyvi = True
except Exception:
    co_pyvi = False

with open("data/mo_hinh_co_so.pkl", "rb") as f:
    du_lieu_cs = pickle.load(f)

with open("data/mo_hinh_sota.pkl", "rb") as f:
    du_lieu_sota = pickle.load(f)

tfidf = du_lieu_cs["tfidf"]
scaler_cs = du_lieu_cs["scaler"]
mh_cs = du_lieu_cs["model"]
mh_xai = du_lieu_cs["model_xai"]

mh_sota = du_lieu_sota["model"]
scaler_emb = du_lieu_sota["scaler_emb"]

thiet_bi = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2").to(thiet_bi)
phobert.eval()

vec_khong = np.zeros((1, 768))
DIEM_CHAN_RBF = float(mh_sota.decision_function(vec_khong)[0])

df_cs = pd.read_csv("data/ket_qua_co_so.csv")
df_st = pd.read_csv("data/ket_qua_sota.csv")
df_tong = pd.concat([df_cs, df_st], ignore_index=True)
df_tong.to_csv("data/bang_so_sanh_tong_hop.csv", index=False)

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

def ma_hoa_phobert(danh_sach_vb):
    ma_hoa = tokenizer(
        danh_sach_vb, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(thiet_bi)
    with torch.no_grad():
        dau_ra = phobert(**ma_hoa)
        mask = ma_hoa["attention_mask"].unsqueeze(-1).expand(dau_ra.last_hidden_state.size()).float()
        tong_vec = torch.sum(dau_ra.last_hidden_state * mask, 1)
        tong_mask = torch.clamp(mask.sum(1), min=1e-9)
        vec_tb = (tong_vec / tong_mask).cpu().numpy()
    return scaler_emb.transform(vec_tb)

def phan_tich_tin(van_ban_goc):
    vb_sach = lam_sach(van_ban_goc)
    cac_tu = vb_sach.split()
    if len(cac_tu) == 0:
        return {"nhan": 0, "canh_bao_nguy_trang": False, "xac_suat_gia": 0.0, "diem_svm": 0.0, "tu_khoa": [], "hanh_vi": {}}

    do_dai = len(van_ban_goc)
    so_cham_than = van_ban_goc.count("!")
    so_dau_hoi = van_ban_goc.count("?")
    ty_le_hoa = tinh_ty_le_hoa(van_ban_goc)

    vec_hv = scaler_cs.transform([[do_dai, so_cham_than, so_dau_hoi, ty_le_hoa]])
    vec_tfidf = tfidf.transform([vb_sach])
    vec_cs_full = hstack([vec_tfidf, csr_matrix(vec_hv)])
    diem_baseline = float(mh_cs.decision_function(vec_cs_full)[0])

    vec_goc = ma_hoa_phobert([vb_sach])
    diem_sota_tho = float(mh_sota.decision_function(vec_goc)[0])
    ty_le_do_dai = min(1.0, len(cac_tu) / 30.0)
    diem_sota_hieu_chinh = diem_sota_tho - DIEM_CHAN_RBF * (1.0 - 0.7 * ty_le_do_dai)

    gioi_han_tu = min(len(cac_tu), 40)
    danh_sach_che = []
    for i in range(gioi_han_tu):
        tu_con_lai = cac_tu[:i] + cac_tu[i+1:]
        danh_sach_che.append(" ".join(tu_con_lai) if len(tu_con_lai) > 0 else " ")

    vec_che = ma_hoa_phobert(danh_sach_che)
    diem_che = mh_sota.decision_function(vec_che)
    do_lech = diem_sota_tho - diem_che

    tu_vung = tfidf.vocabulary_
    he_so_lr = mh_xai.coef_[0]
    diem_tu = []
    tong_lech_duong = 0.0
    tong_lech_am = 0.0

    for i, tu in enumerate(cac_tu):
        d = float(do_lech[i]) if i < gioi_han_tu else 0.0
        if tu in tu_vung:
            d = d * 0.7 + float(he_so_lr[tu_vung[tu]]) * 0.3
        if d > 0.02:
            tong_lech_duong += d
        elif d < -0.02:
            tong_lech_am += abs(d)
        diem_tu.append((tu.replace("_", " "), round(d, 4)))

    diem_ket_hop = 0.65 * diem_sota_hieu_chinh + 0.35 * diem_baseline + 0.5 * (tong_lech_duong - tong_lech_am)
    xac_suat_gia = float(1.0 / (1.0 + np.exp(-diem_ket_hop)))
    nhan_du_doan = 1 if diem_ket_hop > 0 else 0
    canh_bao_nguy_trang = (nhan_du_doan == 0) and (tong_lech_duong > tong_lech_am) and (tong_lech_duong > 0.08)

    hanh_vi = {
        "do_dai": do_dai,
        "so_cham_than": so_cham_than,
        "so_dau_hoi": so_dau_hoi,
        "ty_le_hoa": round(ty_le_hoa * 100, 2)
    }

    return {
        "nhan": nhan_du_doan,
        "canh_bao_nguy_trang": canh_bao_nguy_trang,
        "xac_suat_gia": round(xac_suat_gia * 100, 2),
        "diem_svm": round(diem_ket_hop, 4),
        "diem_sota_tho": round(diem_sota_tho, 4),
        "tu_khoa": diem_tu,
        "hanh_vi": hanh_vi
    }

GIAO_DIEN_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phát Hiện Tin Giả Tiếng Việt (PhoBERT + XAI)</title>
<style>
:root { --nen: #e8ecf1; --giay: #ffffff; --muc: #0f1b2d; --muc-phu: #4b5668; --vien: #c4cdd9; --nhat: #f3f5f8; --do: #b91c1c; --xanh: #15803d; --nut: #12294a; --tieu-diem: #1d4ed8; }
* { box-sizing: border-box; }
body { margin: 0; height: 100vh; display: flex; flex-direction: column; background: var(--nen); color: var(--muc); font-family: "Segoe UI", "SF Pro Text", Roboto, "Helvetica Neue", Arial, sans-serif; }
header { display: flex; align-items: center; justify-content: space-between; height: 62px; padding: 0 28px; background: var(--giay); border-bottom: 2px solid var(--muc); flex-shrink: 0; }
h1 { margin: 0; font-size: 21px; font-weight: 800; letter-spacing: -0.01em; }
.nhan-so { display: flex; gap: 8px; }
.nhan-so span { padding: 4px 10px; border: 1px solid var(--vien); border-radius: 4px; background: var(--nhat); font-size: 13px; font-weight: 600; }
main { flex: 1; min-height: 0; display: grid; grid-template-columns: 5fr 7fr; gap: 20px; padding: 20px 28px 24px; }
.cot { display: flex; flex-direction: column; gap: 16px; min-height: 0; }
.khung { background: var(--giay); border: 1px solid var(--vien); border-radius: 6px; padding: 16px 18px; }
.khung h2 { margin: 0 0 10px; font-size: 15px; font-weight: 700; }
#khung-nhap { flex: 1; display: flex; flex-direction: column; min-height: 0; }
textarea { flex: 1; width: 100%; min-height: 150px; resize: none; padding: 12px 14px; border: 1px solid var(--vien); border-radius: 4px; font-size: 16px; line-height: 1.55; font-family: inherit; color: var(--muc); background: var(--giay); }
textarea:focus { outline: none; border-color: var(--muc); box-shadow: 0 0 0 2px var(--vien); }
.hang-mau { display: flex; gap: 10px; margin: 12px 0; }
button { font-family: inherit; cursor: pointer; }
button:focus-visible { outline: 3px solid var(--tieu-diem); outline-offset: 2px; }
.nut-mau { flex: 1; padding: 9px 12px; border: 1px solid var(--muc); border-radius: 4px; background: var(--giay); color: var(--muc); font-size: 14px; font-weight: 600; }
.nut-mau:hover { background: var(--nhat); }
#nut-chay { width: 100%; height: 52px; border: none; border-radius: 4px; background: var(--nut); color: #fff; font-size: 17px; font-weight: 700; display: flex; align-items: center; justify-content: center; gap: 10px; }
#nut-chay:hover { background: #0b1d37; }
#nut-chay:disabled { cursor: wait; background: #46566f; }
.quay { display: none; width: 16px; height: 16px; border: 3px solid rgba(255,255,255,0.35); border-top-color: #fff; border-radius: 50%; animation: quay 0.7s linear infinite; }
#nut-chay.dang .quay { display: inline-block; }
@keyframes quay { to { transform: rotate(360deg); } }
.dai-the { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.the { padding: 10px 12px; border: 1px solid var(--vien); border-radius: 4px; background: var(--nhat); }
.the b { display: block; font-size: 26px; font-weight: 800; line-height: 1.15; }
.the span { font-size: 12.5px; color: var(--muc-phu); }
#phai { transition: opacity 0.2s; }
#phai.cho { opacity: 0.4; }
.ket-luan { display: grid; grid-template-columns: auto 1fr; gap: 28px; align-items: center; padding: 18px 24px; border-radius: 10px; border: 1px solid var(--vien); background: var(--giay); color: var(--muc); }
.ket-luan.gia { background: var(--do); border-color: var(--do); color: #fff; }
.ket-luan.that { background: var(--xanh); border-color: var(--xanh); color: #fff; }
.kl-nho { font-size: 13px; font-weight: 600; opacity: 0.85; }
.kl-chinh { font-size: 40px; font-weight: 800; line-height: 1.1; letter-spacing: -0.01em; }
.kl-dong { display: flex; justify-content: space-between; align-items: baseline; font-size: 15px; font-weight: 600; }
.kl-dong b { font-size: 26px; font-weight: 800; }
.nen-thanh { height: 12px; margin: 6px 0 8px; border-radius: 6px; background: rgba(255,255,255,0.32); overflow: hidden; }
.day-thanh { height: 100%; background: #fff; border-radius: 6px; }
.kl-fx { font-size: 14px; }
.ket-luan .trong-kl { grid-column: 1 / -1; font-size: 15px; color: var(--muc-phu); }
#luu-y { display: none; padding: 10px 14px; border: 1px solid #b45309; border-radius: 6px; background: #fef3c7; color: #5c3203; font-size: 13.5px; line-height: 1.45; }
#khung-xai { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.dau-khung { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }
.dau-khung h2 { margin: 0; }
.chu-thich { display: flex; gap: 14px; font-size: 13px; font-weight: 600; }
.chu-thich i { display: inline-block; width: 14px; height: 14px; margin-right: 6px; border-radius: 3px; vertical-align: -2px; }
#xai { flex: 1; min-height: 0; overflow: auto; padding: 10px 14px; border: 1px solid var(--vien); border-radius: 4px; background: var(--nhat); font-size: 17px; line-height: 2.15; }
.tu { padding: 2px 4px; margin: 0 1px; border-radius: 3px; }
.trong { color: var(--muc-phu); font-size: 14px; }
.hang { display: grid; grid-template-columns: 120px 1fr; align-items: center; height: 24px; }
.nhan-tu { font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 8px; }
.duong { position: relative; height: 16px; }
.duong::before { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: #8794a8; }
.thanh { position: absolute; top: 1px; height: 14px; border-radius: 2px; }
.so { position: absolute; top: 0; font-size: 13px; font-weight: 700; line-height: 16px; white-space: nowrap; }
.truc { display: flex; justify-content: space-between; margin: 0 0 6px 120px; font-size: 13px; font-weight: 700; }
#goi-y { display: none; position: fixed; z-index: 10; transform: translate(-50%, -100%); padding: 4px 9px; border-radius: 4px; background: var(--muc); color: #fff; font-size: 13px; font-weight: 600; pointer-events: none; white-space: nowrap; }
@media (max-width: 980px) {
  body { height: auto; }
  main { grid-template-columns: 1fr; }
  .ket-luan { grid-template-columns: 1fr; gap: 10px; }
  header { height: auto; flex-direction: column; align-items: flex-start; gap: 8px; padding: 12px 20px; }
  #xai { max-height: 320px; }
}
</style>
</head>
<body>
<header>
  <h1>Phát Hiện Tin Giả Tiếng Việt (PhoBERT + XAI)</h1>
  <div class="nhan-so"><span>Accuracy 91,06%</span><span>F1 0,7585</span><span>Cổng 8080</span></div>
</header>
<main>
  <section class="cot">
    <div class="khung" id="khung-nhap">
      <h2>Bài đăng mạng xã hội</h2>
      <textarea id="vb" placeholder="Nhập hoặc dán nội dung bài đăng cần kiểm tra..."></textarea>
      <div class="hang-mau">
        <button class="nut-mau" onclick="chonMau(0)">Mẫu 1: Tin giả</button>
        <button class="nut-mau" onclick="chonMau(1)">Mẫu 2: Tin thật</button>
      </div>
      <button id="nut-chay" onclick="kiemTra()"><span class="quay"></span><span id="chu-nut">Phân tích bài đăng</span></button>
    </div>
    <div class="khung">
      <h2>Đặc trưng hành vi của bài đăng</h2>
      <div class="dai-the">
        <div class="the"><b id="hv-dai">-</b><span>Độ dài (ký tự)</span></div>
        <div class="the"><b id="hv-cham">-</b><span>Dấu chấm than (!)</span></div>
        <div class="the"><b id="hv-hoi">-</b><span>Dấu hỏi (?)</span></div>
        <div class="the"><b id="hv-hoa">-</b><span>Tỷ lệ chữ hoa</span></div>
      </div>
    </div>
  </section>
  <section class="cot" id="phai">
    <div class="ket-luan" id="kl"><div class="trong-kl">Chưa có kết quả. Nhập bài đăng rồi bấm Phân tích bài đăng.</div></div>
    <div id="luu-y">Lưu ý XAI: Văn phong câu giống tin chính thống nhưng xuất hiện cụm từ khóa mang trọng số nghi vấn ĐỎ cao hơn XANH. Cần kiểm chứng nguồn thực tế (Fact-check).</div>
    <div class="khung" id="khung-xai">
      <div class="dau-khung">
        <h2>Văn bản phân tích từ khóa (XAI)</h2>
        <div class="chu-thich">
          <span><i style="background:rgba(220,38,38,0.7)"></i>&gt; +0,025: Tin giả</span>
          <span><i style="background:rgba(22,163,74,0.7)"></i>&lt; -0,025: Tin thật</span>
        </div>
      </div>
      <div id="xai"><span class="trong">Các từ khóa quan trọng sẽ được tô màu tại đây.</span></div>
    </div>
    <div class="khung">
      <h2>Top từ khóa ảnh hưởng mạnh nhất</h2>
      <div class="truc"><span style="color:var(--xanh)">Củng cố Tin thật</span><span style="color:var(--do)">Đẩy về Tin giả</span></div>
      <div id="thanh"><div class="trong">Chưa có dữ liệu.</div></div>
    </div>
  </section>
</main>
<div id="goi-y"></div>
<script>
const mau = [
  "KHẨN CẤP!!! Chia sẻ ngay kẻo xóa! Phát hiện loại nước uống đóng chai có chứa chất kịch độc gây tử vong ngay lập tức tại TP.HCM, mọi người tuyệt đối đừng uống!!",
  "Sáng nay Bộ Y tế công bố lịch tiêm chủng mở rộng định kỳ cho trẻ em tại các trạm y tế phường xã trên toàn quốc theo kế hoạch năm."
];
function chonMau(i) { document.getElementById("vb").value = mau[i]; kiemTra(); }
function dinhDang(n, k, dau) {
  if (typeof n !== "number") return "-";
  const s = Math.abs(n).toFixed(k).replace(".", ",");
  if (n < 0) return "-" + s;
  return (dau ? "+" : "") + s;
}
function veThanh(ds) {
  const gop = Object.create(null);
  ds.forEach(([tu, d]) => {
    if (Math.abs(d) > 0.025 && (!(tu in gop) || Math.abs(d) > Math.abs(gop[tu]))) gop[tu] = d;
  });
  const top = Object.entries(gop).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8);
  if (top.length === 0) return `<div class="trong">Không có từ khóa vượt ngưỡng ±0,025.</div>`;
  const max = Math.abs(top[0][1]);
  return top.map(([tu, d]) => {
    const w = Math.abs(d) / max * 36;
    const duong = d > 0;
    const thanh = duong ? `left:50%;width:${w}%;background:var(--do)` : `right:50%;width:${w}%;background:var(--xanh)`;
    const so = duong ? `left:calc(50% + ${w}% + 6px);color:var(--do)` : `right:calc(50% + ${w}% + 6px);color:var(--xanh)`;
    return `<div class="hang"><div class="nhan-tu" title="${tu}">${tu}</div><div class="duong"><div class="thanh" style="${thanh}"></div><div class="so" style="${so}">${dinhDang(d, 4, true)}</div></div></div>`;
  }).join("");
}
async function kiemTra() {
  const vb = document.getElementById("vb").value.trim();
  if (!vb) return;
  const nut = document.getElementById("nut-chay");
  const phai = document.getElementById("phai");
  nut.disabled = true;
  nut.classList.add("dang");
  document.getElementById("chu-nut").textContent = "Đang phân tích...";
  phai.classList.add("cho");
  const kl = document.getElementById("kl");
  try {
    const res = await fetch("/predict?text=" + encodeURIComponent(vb));
    const data = await res.json();
    const hv = data.hanh_vi || {};
    document.getElementById("hv-dai").textContent = hv.do_dai !== undefined ? hv.do_dai : "-";
    document.getElementById("hv-cham").textContent = hv.so_cham_than !== undefined ? hv.so_cham_than : "-";
    document.getElementById("hv-hoi").textContent = hv.so_dau_hoi !== undefined ? hv.so_dau_hoi : "-";
    document.getElementById("hv-hoa").textContent = hv.ty_le_hoa !== undefined ? dinhDang(hv.ty_le_hoa, 2) + "%" : "-";
    const gia = data.nhan === 1;
    const xs = gia ? data.xac_suat_gia : 100 - data.xac_suat_gia;
    kl.className = "ket-luan " + (gia ? "gia" : "that");
    kl.innerHTML = `<div><div class="kl-nho">Kết luận</div><div class="kl-chinh">${gia ? "TIN GIẢ" : "TIN THẬT"}</div></div>
      <div><div class="kl-dong"><span>Xác suất ${gia ? "Tin giả" : "Tin thật"} (mức độ tin cậy)</span><b>${dinhDang(xs, 2)}%</b></div>
      <div class="nen-thanh"><div class="day-thanh" style="width:${xs}%"></div></div>
      <div class="kl-fx">Điểm hàm quyết định f(x) = <b>${dinhDang(data.diem_svm, 4, true)}</b> &nbsp;|&nbsp; RBF gốc = ${dinhDang(data.diem_sota_tho, 4, true)}</div></div>`;
    document.getElementById("luu-y").style.display = data.canh_bao_nguy_trang ? "block" : "none";
    let tuHtml = "";
    data.tu_khoa.forEach(([tu, diem]) => {
      let bg = "transparent";
      const dam = Math.max(0.18, Math.min(Math.abs(diem) * 2.5, 0.78));
      if (diem > 0.025) bg = `rgba(220, 38, 38, ${dam})`;
      else if (diem < -0.025) bg = `rgba(22, 163, 74, ${dam})`;
      tuHtml += `<span class="tu" style="background:${bg}" data-tu="${tu}" data-diem="${dinhDang(diem, 4, true)}">${tu}</span> `;
    });
    document.getElementById("xai").innerHTML = tuHtml || `<span class="trong">Không có từ khóa nào để phân tích.</span>`;
    document.getElementById("thanh").innerHTML = veThanh(data.tu_khoa);
  } catch (e) {
    kl.className = "ket-luan";
    kl.innerHTML = `<div class="trong-kl">Không nhận được kết quả từ máy chủ cổng 8080. Kiểm tra cửa sổ chạy Python rồi bấm Phân tích bài đăng lại.</div>`;
  } finally {
    nut.disabled = false;
    nut.classList.remove("dang");
    document.getElementById("chu-nut").textContent = "Phân tích bài đăng";
    phai.classList.remove("cho");
  }
}
const goiY = document.getElementById("goi-y");
const khungXai = document.getElementById("xai");
khungXai.addEventListener("mouseover", function(e) {
  const t = e.target.closest(".tu");
  if (!t) { goiY.style.display = "none"; return; }
  goiY.textContent = t.dataset.tu + ": " + t.dataset.diem;
  const r = t.getBoundingClientRect();
  goiY.style.display = "block";
  goiY.style.left = (r.left + r.width / 2) + "px";
  goiY.style.top = (r.top - 6) + "px";
});
khungXai.addEventListener("mouseleave", function() { goiY.style.display = "none"; });
khungXai.addEventListener("scroll", function() { goiY.style.display = "none"; });
</script>
</body>
</html>"""

class MayChuDemo(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/predict":
            qs = urllib.parse.parse_qs(parsed.query)
            vb = qs.get("text", [""])[0]
            kq = phan_tich_tin(vb)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(kq, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(GIAO_DIEN_HTML.encode("utf-8"))

if __name__ == "__main__":
    cong = 8080
    print(f"Mo trinh duyet tai dia chi: http://localhost:{cong}")
    print("Nhan Ctrl + C de dung may chu.")
    HTTPServer(("127.0.0.1", cong), MayChuDemo).serve_forever()
