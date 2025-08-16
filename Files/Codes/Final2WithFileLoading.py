import matplotlib
matplotlib.use('TkAgg')  # Spyder’da ayrı pencerede grafik için

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score
)
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import re

# ─── AYARLAR ───────────────────────────────────────────────────────────────────
MIN_CONSEC_MONTHS = 3
PROB_THRESHOLD    = 0.50

W_ORT  = 1.00
W_FULL = 0.00

TYPE_MAP = {"Mesken":0, "Ticarethane":1, "Sanayi":2, "Diğer":3}

ENERJI_YELLOW   = "#FFC700"
GOLD_ORANGE     = "#FFA500"
BLACK_TEXT      = "#000000"
WHITE_TEXT      = "#FFFFFF"
ENERJISA_BLUE   = "#004B9B"

BG_COLOR      = WHITE_TEXT
TEXT_COLOR    = BLACK_TEXT
BUTTON_COLOR  = ENERJISA_BLUE
BUTTON_HOVER  = "#003366"
ENTRY_BG      = "#f2f2f2"

BUTTON_WIDTH   = 17
BUTTON_HEIGHT  = 1
BUTTON_PADX    = 3
BUTTON_PADY    = 3

FONT_TITLE    = ("Segoe UI", 25, "bold")
FONT_NORMAL   = ("Segoe UI", 18)
NAME_FONT     = ("Segoe UI", 16, "bold")


# ─── 1) Veri Hazırlama & Özellik Çıkarımı ───────────────────────────────────────
df = pd.read_csv("son_veri_ml_ready.txt", sep=",")
ort_cols    = [f"ort_tuketim_{i}" for i in range(1,27)]
demand_cols = [f"demand_{i}"      for i in range(1,27)]
df[ort_cols + demand_cols] = df[ort_cols + demand_cols].apply(pd.to_numeric, errors="coerce")

def max_percent_drop(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) < 2:
        return 0.0
    prev = arr[:-1]
    nxt  = arr[1:]
    denom = np.where(prev == 0, np.nan, prev)
    diffs = (prev - nxt) / denom
    return np.nanmax(diffs)

def sustained_drop(arr):
    arr = np.array(arr, dtype=float)
    best = 0.0
    n = len(arr)
    for i in range(n):
        for j in range(i + MIN_CONSEC_MONTHS - 1, n):
            seg = arr[i:j+1]
            if len(seg) >= MIN_CONSEC_MONTHS and np.all(np.diff(seg) < 0):
                drop = (seg[0] - seg[-1]) / seg[0] if seg[0] else 0
                best = max(best, drop)
    return best

def abrupt_zero(arr):
    arr = np.array(arr, dtype=float)
    return int(any(arr[i] > 0 and arr[i+1] == 0 for i in range(len(arr)-1)))

def avg_drop(arr):
    arr = np.array(arr, dtype=float)
    return (arr[0] - arr[-1]) / arr[0] if len(arr) >= 2 and arr[0] != 0 else 0

def extract_all_features(o, d, atype):
    feats = {
        "inst_drop_ort": max_percent_drop(o),
        "sust_drop_ort": sustained_drop(o),
        "zero_ort":      abrupt_zero(o),
        "avg_drop_ort":  avg_drop(o),
        "inst_drop_dem": max_percent_drop(d),
        "sust_drop_dem": sustained_drop(d),
        "zero_dem":      abrupt_zero(d),
        "avg_drop_dem":  avg_drop(d),
    }
    code = TYPE_MAP.get(atype, 3)
    for _, t in TYPE_MAP.items():
        feats[f"type_{t}"] = 1 if t == code else 0
    return feats

def extract_ort_features(o, d, atype):
    feats = {
        "inst_drop_ort": max_percent_drop(o),
        "sust_drop_ort": sustained_drop(o),
        "zero_ort":      abrupt_zero(o),
        "avg_drop_ort":  avg_drop(o),
    }
    code = TYPE_MAP.get(atype, 3)
    for _, t in TYPE_MAP.items():
        feats[f"type_{t}"] = 1 if t == code else 0
    return feats

feat_all = df.apply(lambda r: pd.Series(extract_all_features(
    r[ort_cols].values, r[demand_cols].values, r["type"])), axis=1)
feat_ort = df.apply(lambda r: pd.Series(extract_ort_features(
    r[ort_cols].values, r[demand_cols].values, r["type"])), axis=1)

X_full = feat_all
X_ort  = feat_ort
y      = df["kacak"]

# ─── 2) Model Eğitimi ───────────────────────────────────────────────────────────
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_full, y, test_size=0.3, stratify=y, random_state=42)
Xo_train, Xo_test, yo_train, yo_test = train_test_split(
    X_ort, y, test_size=0.3, stratify=y, random_state=42)

clf_full = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf_ort  = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

clf_full.fit(Xf_train, yf_train)
clf_ort.fit(Xo_train, yo_train)

normal_indices = [i for i, v in zip(Xf_test.index, yf_test) if v == 0]
fraud_indices  = [i for i, v in zip(Xf_test.index, yf_test) if v == 1]


# ─── 3) Tahmin ve Grafik Fonksiyonları ─────────────────────────────────────────
def combined_probability(o, d, atype):
    row_all = pd.DataFrame([extract_all_features(o, d, atype)], columns=X_full.columns).fillna(0)
    row_ort = pd.DataFrame([extract_ort_features(o, d, atype)], columns=X_ort.columns).fillna(0)
    p_full = clf_full.predict_proba(row_all)[0, 1]
    p_ort  = clf_ort.predict_proba(row_ort)[0, 1]
    return W_FULL * p_full + W_ORT * p_ort

def risk_category(p):
    if   p >= 0.90: return "Yüksek Risk", "#D32F2F"
    elif p >= 0.75: return "Risk",       "#F57C00"
    elif p >= 0.60: return "Takip Şart", "#FFA000"
    elif p >= 0.50: return "Takip Şart", "#FFB300"
    elif p >= 0.30: return "Normal",     "#388E3C"
    else:           return "Güvenli",    "#2E7D32"


def show_input_plots():
    try:
        o = [float(x) for x in ort_entry.get("1.0", tk.END).split(",") if x.strip()]
        d = [float(x) for x in dem_entry.get("1.0", tk.END).split(",") if x.strip()]
        if len(o) < 2 or len(d) < 2:
            raise ValueError("En az 2 değer girilmeli.")
    except Exception as e:
        return messagebox.showerror("Hata", str(e))

    plt.figure(); plt.plot(range(1, len(o)+1), o, marker='o')
    plt.title("Ortalama Tüketim (kWh)"); plt.xlabel("Ölçüm"); plt.ylabel("Tüketim (kWh)"); plt.tight_layout(); plt.show()

    plt.figure(); plt.plot(range(1, len(d)+1), d, marker='o')
    plt.title("Demand (kWh)"); plt.xlabel("Ölçüm"); plt.ylabel("Demand (kWh)"); plt.tight_layout(); plt.show()


def show_main_plots():
    y_probs = []
    for idx in Xf_test.index:
        o = df.loc[idx, ort_cols].astype(float).values
        d = df.loc[idx, demand_cols].astype(float).values
        at = df.loc[idx, "type"]
        y_probs.append(combined_probability(o, d, at))
    y_probs = np.array(y_probs)
    y_true  = yf_test.values
    y_pred  = (y_probs >= PROB_THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(); plt.imshow(cm, cmap=plt.cm.Blues); plt.title("Genel CM")
    plt.colorbar(); plt.xlabel("Pred"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout(); plt.show()

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], '--')
    plt.title(f"ROC (AUC={roc_auc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.show()

    precisions, recalls, thr = precision_recall_curve(y_true, y_probs)
    accuracies = [accuracy_score(y_true, (y_probs >= t).astype(int)) for t in thr]
    plt.figure()
    plt.plot(thr, accuracies, label="Accuracy")
    plt.plot(thr, precisions[:-1], label="Precision")
    plt.plot(thr, recalls[:-1], label="Recall")
    plt.title("Performance vs Threshold"); plt.xlabel("Threshold")
    plt.legend(); plt.tight_layout(); plt.show()


def prompt_and_show_user_plot():
    win = tk.Toplevel(root); win.title("Grafik Seç"); win.geometry("300x230"); win.configure(bg=BG_COLOR)
    tk.Label(win, text="Normal Örnek No:", font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(10,0))
    ns = tk.Spinbox(win, from_=1, to=len(normal_indices), font=FONT_NORMAL); ns.pack()
    tk.Label(win, text="Kaçak Örnek No:", font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(10,0))
    fs = tk.Spinbox(win, from_=1, to=len(fraud_indices), font=FONT_NORMAL); fs.pack()

    def plot_sel():
        n = int(ns.get())-1; f = int(fs.get())-1
        idx_n = normal_indices[n]; idx_f = fraud_indices[f]
        o_n = df.loc[idx_n, ort_cols].astype(float).values
        o_f = df.loc[idx_f, ort_cols].astype(float).values

        plt.figure(); plt.plot(range(1, 27), o_n, label="Normal")
        plt.plot(range(1, 27), o_f, label="Kaçak")
        plt.title("Ortalama Tüketim (kWh)"); plt.xlabel("Ay"); plt.ylabel("Tüketim (kWh)")
        plt.legend(); plt.tight_layout(); plt.show()

    btn = tk.Button(win, text="Göster", command=plot_sel, font=FONT_NORMAL,
                    bg=BUTTON_COLOR, fg=BLACK_TEXT, activebackground=BUTTON_HOVER,
                    bd=0, relief="flat", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                    padx=BUTTON_PADX, pady=BUTTON_PADY)
    btn.pack(pady=20); btn.bind("<Enter>", lambda e: btn.config(bg=BUTTON_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=BUTTON_COLOR))


def predict():
    try:
        o = [float(x) for x in ort_entry.get("1.0", tk.END).split(",") if x.strip()]
        d = [float(x) for x in dem_entry.get("1.0", tk.END).split(",") if x.strip()]
        at = tip_menu.get()
        if len(o) < 2 or len(d) < 2:
            raise ValueError("En az 2 değer girilmeli.")
        p = combined_probability(o, d, at)
        show_result_window(p)
    except Exception as e:
        messagebox.showerror("Hata", str(e))


# ─── YARDIMCI: prefix'e göre satırdan dizi çek (26 şartı yok) ───────────────────
def extract_sequence_from_row(row, prefix):
    # prefix: "ort_tuketim_" veya "demand_"
    cols = [c for c in row.index if c.startswith(prefix)]
    def idx_of(c):
        m = re.match(rf'^{re.escape(prefix)}(\d+)$', c)
        if m:
            return int(m.group(1))
        # fallback: kolon adındaki tüm rakamları al
        digits = re.findall(r'\d+', c)
        return int(digits[0]) if digits else 10**9
    cols = sorted(cols, key=idx_of)
    vals = pd.to_numeric(row[cols], errors="coerce").values if cols else np.array([], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)
    return vals

# ─── YENİ: Etiketli dosya ile test (değiştirildi) ──────────────────────────────
def test_with_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
    if not path:
        return
    df_file = pd.read_csv(path, sep=",")
    need_cols = {"type", "kacak"}
    if not need_cols.issubset(set(df_file.columns)):
        return messagebox.showerror("Hata", "Dosyada 'type' ve 'kacak' kolonları bulunmalı.")

    y_true = pd.to_numeric(df_file["kacak"], errors="coerce").fillna(0).astype(int).values
    y_probs = []
    for _, row in df_file.iterrows():
        o = extract_sequence_from_row(row, "ort_tuketim_")
        d = extract_sequence_from_row(row, "demand_")
        at = row["type"]
        if len(o) < 2 or len(d) < 2:
            # Yetersiz veri varsa güvenli tarafta kal: 0 risk ver.
            y_probs.append(0.0)
            continue
        y_probs.append(combined_probability(o, d, at))

    y_probs = np.array(y_probs)
    y_pred  = (y_probs >= PROB_THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(); plt.imshow(cm, cmap=plt.cm.Blues); plt.title("Dosya ile CM")
    plt.colorbar(); plt.xlabel("Pred"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout(); plt.show()

    acc = accuracy_score(y_true, y_pred)
    messagebox.showinfo("Sonuç", f"Dosya testi tamamlandı.\nDoğruluk: %{acc*100:.2f}")

# ─── YENİ: Etiketsiz dosya ile tahmin (değiştirildi) ────────────────────────────
def predict_with_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
    if not path:
        return
    df_file = pd.read_csv(path, sep=",")
    if "type" not in df_file.columns:
        return messagebox.showerror("Hata", "Dosyada 'type' kolonu bulunmalı.")

    win = tk.Toplevel(root); win.title("Dosya Tahmin Sonuçları"); win.geometry("400x400")
    txt = tk.Text(win, font=FONT_NORMAL)
    txt.pack(fill="both", expand=True)

    for idx, row in df_file.iterrows():
        o  = extract_sequence_from_row(row, "ort_tuketim_")
        d  = extract_sequence_from_row(row, "demand_")
        at = row["type"]
        if len(o) < 2 or len(d) < 2:
            txt.insert(tk.END, f"Satır {idx+1}: Veri yetersiz (en az 2 değer gerekli)\n")
            continue
        p = combined_probability(o, d, at)
        cat, _ = risk_category(p)
        txt.insert(tk.END, f"Satır {idx+1}: %{p*100:.1f} - {cat}\n")

    txt.config(state="disabled")


def show_result_window(p):
    category, clr = risk_category(p)
    win = tk.Toplevel(root); win.title("Sonuç"); win.geometry("400x250")
    win.configure(bg=BG_COLOR); win.resizable(False, False)
    tk.Label(win, text=category, font=("Segoe UI", 20, "bold"), fg=clr, bg=BG_COLOR).pack(pady=(30,10))
    tk.Label(win, text=f"Olasılık: {p*100:.1f}%", font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR).pack()
    tk.Label(win, text=f"Eşik: %{PROB_THRESHOLD*100:.0f}", font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(0,20))
    btn = tk.Button(win, text="Kapat", command=win.destroy, font=FONT_NORMAL,
                    bg=BUTTON_COLOR, fg=BLACK_TEXT, activebackground=BUTTON_HOVER,
                    bd=0, relief="flat", width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                    padx=BUTTON_PADX, pady=BUTTON_PADY)
    btn.pack(pady=10); btn.bind("<Enter>", lambda e: btn.config(bg=BUTTON_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=BUTTON_COLOR))


# ─── 4) GUI BAŞLATMA ────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("⚡ Power Check")
root.geometry("700x760")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

def on_enter(e): e.widget['background'] = BUTTON_HOVER
def on_leave(e): e.widget['background'] = BUTTON_COLOR

# Logolar, isimler, başlık
logo_frame = tk.Frame(root, bg=BG_COLOR); logo_frame.pack(pady=5)
for p, w, h in [("toroslarLogo.png", 140, 75), ("secondAppLogo.png", 220, 200), ("spark.png", 140, 140)]:
    try:
        img = Image.open(p).resize((w, h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(logo_frame, image=photo, bg=BG_COLOR); lbl.image = photo; lbl.pack(side=tk.LEFT, padx=10)
    except:
        pass

name_frame = tk.Frame(root, bg=BG_COLOR); name_frame.pack()
for n in ["Mehmet Fatih Göğüş","Burak Eren Özdemir","Tuçe Kar","Berkay Dalkılıç"]:
    tk.Label(name_frame, text=n, font=NAME_FONT, bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=5)
tk.Label(root, text="ve", font=NAME_FONT, bg=BG_COLOR, fg=TEXT_COLOR).pack()
tk.Label(root, text="Mesut Bilgiç", font=NAME_FONT, bg=BG_COLOR, fg=TEXT_COLOR).pack()
tk.Label(root, text="Kaçak Tespit Uygulaması", font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=10)

# Girdi alanları
frame = tk.Frame(root, bg=BG_COLOR); frame.pack(padx=10, fill="x")
def add_labeled(label):
    tk.Label(frame, text=label, font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR, anchor="w").pack(fill="x", pady=(5,0))
    txt = tk.Text(frame, height=2, font=FONT_NORMAL, bg=ENTRY_BG, fg=TEXT_COLOR, insertbackground="black")
    txt.pack(fill="x", pady=(0,5)); return txt

ort_entry = add_labeled("Ortalama Tüketim (kWh) (virgülle):")
dem_entry = add_labeled("Demand (kWh) (virgülle):")
tk.Label(frame, text="Abone Tipi:", font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR, anchor="w").pack(fill="x", pady=(5,0))
tip_menu = ttk.Combobox(frame, values=list(TYPE_MAP.keys()), state="readonly", font=FONT_NORMAL)
tip_menu.current(0); tip_menu.pack(fill="x", pady=(0,15))

# Butonlar: Grafik ve Tahmin
btn_row = tk.Frame(root, bg=BG_COLOR); btn_row.pack(pady=5)
for txt, cmd, col in [
    ("📈 Girdi Grafikleri",        show_input_plots,      ENERJI_YELLOW),
    ("📋 Kullanıcı Grafiği",       prompt_and_show_user_plot, GOLD_ORANGE),
    ("📊 Genel Metrik Grafikleri", show_main_plots,       ENERJISA_BLUE)
]:
    b = tk.Button(btn_row, text=txt, command=cmd, font=FONT_NORMAL, bg=col, fg=BLACK_TEXT,
                  activebackground=BUTTON_HOVER, bd=0, relief="flat",
                  width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                  padx=BUTTON_PADX, pady=BUTTON_PADY)
    b.pack(side=tk.LEFT, padx=5); b.bind("<Enter>", on_enter); b.bind("<Leave>", on_leave)

action_row = tk.Frame(root, bg=BG_COLOR); action_row.pack(pady=10)
btn_predict = tk.Button(action_row, text="🔍 Tahmin Et", command=predict,
                        font=FONT_NORMAL, bg=BUTTON_COLOR, fg=BLACK_TEXT,
                        activebackground=BUTTON_HOVER, bd=0, relief="flat",
                        width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                        padx=BUTTON_PADX, pady=BUTTON_PADY)
btn_predict.pack(side=tk.LEFT, padx=5); btn_predict.bind("<Enter>", on_enter); btn_predict.bind("<Leave>", on_leave)

btn_clear = tk.Button(action_row, text="❌ Temizle",
                      command=lambda: (ort_entry.delete("1.0", tk.END), dem_entry.delete("1.0", tk.END)),
                      font=FONT_NORMAL, bg=BUTTON_COLOR, fg=BLACK_TEXT,
                      activebackground=BUTTON_HOVER, bd=0, relief="flat",
                      width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                      padx=BUTTON_PADX, pady=BUTTON_PADY)
btn_clear.pack(side=tk.LEFT, padx=5); btn_clear.bind("<Enter>", on_enter); btn_clear.bind("<Leave>", on_leave)

# Dosya ile Test ve Tahmin Butonları
file_row = tk.Frame(root, bg=BG_COLOR); file_row.pack(pady=10)
btn_file_test = tk.Button(file_row, text="📁 Test", command=test_with_file,
                          font=FONT_NORMAL, bg=ENERJI_YELLOW, fg=BLACK_TEXT,
                          activebackground=BUTTON_HOVER, bd=0, relief="flat",
                          width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                          padx=BUTTON_PADX, pady=BUTTON_PADY)
btn_file_test.pack(side=tk.LEFT, padx=5); btn_file_test.bind("<Enter>", on_enter); btn_file_test.bind("<Leave>", on_leave)

btn_file_pred = tk.Button(file_row, text="📂 Tahmin", command=predict_with_file,
                          font=FONT_NORMAL, bg=GOLD_ORANGE, fg=BLACK_TEXT,
                          activebackground=BUTTON_HOVER, bd=0, relief="flat",
                          width=BUTTON_WIDTH, height=BUTTON_HEIGHT,
                          padx=BUTTON_PADX, pady=BUTTON_PADY)
btn_file_pred.pack(side=tk.LEFT, padx=5); btn_file_pred.bind("<Enter>", on_enter); btn_file_pred.bind("<Leave>", on_leave)

root.mainloop()
