"""
Acoustic leak detection for the SIH wearable (idea-stage prototype).

Classes : no_leak | leak | background_noise
Input   : 1-second audio windows (16 kHz) from a contact mic / MEMS mic
Models  : (1) Random Forest on MFCC stats  -> baseline
          (2) Small CNN on log-mel spectrograms -> main model
Outputs : metrics + confusion matrices (PNG) for your PPT, saved model.

Run:
    pip install numpy scipy scikit-learn librosa matplotlib torch
    python leak_detection_ai.py                      # uses SYNTHETIC data
    python leak_detection_ai.py --data_dir dataset   # uses REAL .wav files
        dataset/no_leak/*.wav
        dataset/leak/*.wav
        dataset/background_noise/*.wav
"""
import argparse, glob, os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

SR = 16000
WIN = SR                     # 1 second windows
CLASSES = ["no_leak", "leak", "background_noise"]
N_MELS, HOP = 64, 512        # -> spectrogram of shape (64, 32)
SEED = 42


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def bandpass(x, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def synth_window(label, rng):
    """Rough physical proxy: leaks = broadband hiss (~0.4-6 kHz) from turbulent
    flow through a small orifice. Replace with real recordings when you have them."""
    n = WIN
    t = np.arange(n) / SR
    floor = rng.normal(0, rng.uniform(0.01, 0.08), n)

    if label == 0:    # quiet pipe: mains hum + faint low-frequency flow
        x = (floor + 0.05 * np.sin(2 * np.pi * rng.choice([50, 100]) * t)
             + 0.03 * bandpass(rng.normal(size=n), 50, 300))
    elif label == 1:  # leak: band-limited broadband hiss
        hiss = bandpass(rng.normal(size=n), rng.uniform(400, 1000), rng.uniform(3000, 6000))
        x = floor + rng.uniform(0.003, 0.05) * hiss / (hiss.std() + 1e-9)
    else:             # background: motor harmonics + speech/tap-like bursts
        f0 = rng.uniform(80, 250)
        x = floor + sum(0.1 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6))
        bursts = (np.sin(2 * np.pi * rng.uniform(2, 6) * t) > 0).astype(float)
        x += 0.1 * bursts * bandpass(rng.normal(size=n), 200, rng.uniform(2500, 6000))   # tap/water-flow can look hiss-like

    # shared ambient (hum + rumble) in EVERY class so classes overlap like real recordings
    x = x + rng.uniform(0, 0.08) * np.sin(2 * np.pi * rng.choice([50, 100, 150]) * t)
    x = x + rng.uniform(0, 0.08) * bandpass(rng.normal(size=n), 50, 400)
    return (x * rng.uniform(0.5, 1.5)).astype(np.float32)   # random gain


def make_synthetic(per_class=600):
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for label in range(len(CLASSES)):
        for _ in range(per_class):
            X.append(synth_window(label, rng))
            y.append(label)
    return np.array(X), np.array(y)


def load_real(data_dir):
    X, y = [], []
    for label, name in enumerate(CLASSES):
        for f in glob.glob(os.path.join(data_dir, name, "*.wav")):
            audio, _ = librosa.load(f, sr=SR, mono=True)
            for i in range(0, len(audio) - WIN + 1, WIN):      # 1 s chunks
                X.append(audio[i:i + WIN])
                y.append(label)
    if not X:
        raise SystemExit(f"No .wav files found under {data_dir}/<class>/")
    return np.array(X, dtype=np.float32), np.array(y)


# ----------------------------------------------------------------------------
# 2. FEATURES
# ----------------------------------------------------------------------------
def logmel(x):
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(m, ref=1.0, top_db=80.0)[:, :32]


def mfcc_stats(x):
    m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=20, n_fft=1024, hop_length=HOP)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


# ----------------------------------------------------------------------------
# 3. FEEDBACK MAPPING (what the trainee actually sees/feels)
# ----------------------------------------------------------------------------
def feedback(p_leak):
    if p_leak > 0.8:
        return "RED   | fast beep + strong vibration  (leak likely here)"
    if p_leak > 0.4:
        return "AMBER | slow beep + light vibration   (getting closer)"
    return "GREEN | no alert"


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
def main(a):
    X, y = load_real(a.data_dir) if a.data_dir else make_synthetic()
    print(f"Data: {len(X)} windows, class counts = {np.bincount(y)}  "
          f"({'REAL' if a.data_dir else 'SYNTHETIC'})")

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25,
                                      stratify=y, random_state=SEED)

    # ---- Baseline: Random Forest on MFCC stats ----
    F = np.array([mfcc_stats(x) for x in X])
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(F[idx_tr], y[idx_tr])
    rf_pred = rf.predict(F[idx_te])
    print("\n=== Random Forest (MFCC) ===")
    print("accuracy:", round(accuracy_score(y[idx_te], rf_pred), 4))
    print(classification_report(y[idx_te], rf_pred, target_names=CLASSES, digits=3))

    fig, ax = plt.subplots(1, 2 if not a.skip_cnn else 1, figsize=(11, 4.5))
    ax = np.atleast_1d(ax)
    ConfusionMatrixDisplay.from_predictions(y[idx_te], rf_pred, display_labels=CLASSES,
                                            ax=ax[0], colorbar=False, cmap="Blues")
    ax[0].set_title("Random Forest (MFCC)")

    # ---- Main model: small CNN on log-mel ----
    if not a.skip_cnn:
        import torch, torch.nn as nn
        torch.manual_seed(SEED)
        S = np.array([logmel(x) for x in X], dtype=np.float32)
        mu, sd = S[idx_tr].mean(), S[idx_tr].std() + 1e-6
        S = ((S - mu) / sd)[:, None]                       # (N,1,64,32)
        Xtr, Xte = torch.tensor(S[idx_tr]), torch.tensor(S[idx_te])
        ytr, yte = torch.tensor(y[idx_tr]), torch.tensor(y[idx_te])

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                            nn.Dropout(0.3), nn.Linear(64, len(CLASSES)))
        opt = torch.optim.Adam(cnn.parameters(), lr=2e-3)
        loss_fn = nn.CrossEntropyLoss()
        hist = []
        for ep in range(a.epochs):
            cnn.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss_fn(cnn(Xtr[b]), ytr[b]).backward()
                opt.step()
            cnn.eval()
            with torch.no_grad():
                acc = (cnn(Xte).argmax(1) == yte).float().mean().item()
            hist.append(acc)
            print(f"epoch {ep + 1:02d}  test acc {acc:.3f}")

        with torch.no_grad():
            probs = torch.softmax(cnn(Xte), 1).numpy()
        cnn_pred = probs.argmax(1)
        print("\n=== CNN (log-mel) ===")
        print(classification_report(y[idx_te], cnn_pred, target_names=CLASSES, digits=3))
        ConfusionMatrixDisplay.from_predictions(y[idx_te], cnn_pred, display_labels=CLASSES,
                                                ax=ax[1], colorbar=False, cmap="Greens")
        ax[1].set_title("CNN (log-mel)")
        torch.save({"model": cnn.state_dict(), "mu": mu, "sd": sd}, "leak_cnn.pt")

        print("\nDemo of trainee feedback on 5 test windows:")
        for p, true in list(zip(probs[:, 1], y[idx_te]))[:5]:
            print(f"  true={CLASSES[true]:<17} P(leak)={p:.2f} -> {feedback(p)}")

        plt.figure(figsize=(5, 3.5))
        plt.plot(range(1, len(hist) + 1), hist, marker="o")
        plt.xlabel("epoch"); plt.ylabel("test accuracy"); plt.grid(alpha=.3)
        plt.title("CNN training curve"); plt.tight_layout()
        plt.savefig("training_curve.png", dpi=200)

    fig.tight_layout()
    fig.savefig("confusion_matrices.png", dpi=200)

    # example spectrogram for slides: leak vs no-leak
    fig2, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for k, lab in enumerate([0, 1]):
        ex = X[np.where(y == lab)[0][0]]
        axs[k].imshow(logmel(ex), origin="lower", aspect="auto", cmap="magma")
        axs[k].set_title(CLASSES[lab]); axs[k].set_xlabel("time"); axs[k].set_ylabel("mel band")
    fig2.tight_layout()
    fig2.savefig("spectrogram_examples.png", dpi=200)
    print("\nSaved: confusion_matrices.png, spectrogram_examples.png"
          + ("" if a.skip_cnn else ", training_curve.png, leak_cnn.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None, help="folder with class subfolders of .wav files")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--skip_cnn", action="store_true", help="only run the RF baseline")
    main(p.parse_args())
