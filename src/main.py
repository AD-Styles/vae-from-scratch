"""
VAE from scratch — Variational Autoencoder
FashionMNIST 데이터셋으로 28×28 이미지 생성을 학습합니다.

Reference:
    - Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)
    - https://arxiv.org/abs/1312.6114
"""

import math
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# matplotlib 한글 폰트 적용
_korean_fonts = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "AppleGothic", "Noto Sans CJK KR", "Gulim"]
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _korean_fonts:
    if _font in _available:
        mpl.rcParams["font.family"] = _font
        break
mpl.rcParams["axes.unicode_minus"] = False


# ───────────────────────────────────────────────────────────
# 1. 경로 / 시드 / 디바이스
# ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ───────────────────────────────────────────────────────────
# 2. 하이퍼파라미터
# ───────────────────────────────────────────────────────────
LATENT_DIM = 16            # 잠재 공간 차원
HIDDEN_CHANNELS = (32, 64) # Conv encoder/decoder hidden channels
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
BETA = 1.0                 # β-VAE coefficient (1.0 = 표준 VAE)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ───────────────────────────────────────────────────────────
# 3. 데이터 (FashionMNIST)
# ───────────────────────────────────────────────────────────
def get_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])  # [0, 1]로 정규화
    train_set = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    val_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_set, val_set


# ───────────────────────────────────────────────────────────
# 4. 모델 컴포넌트
# ───────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Encoder: 28×28 이미지 → latent 분포의 (μ, log σ²) 출력

    구조:
        Conv(1→32, stride=2) → ReLU      # 28×28 → 14×14
        Conv(32→64, stride=2) → ReLU     # 14×14 → 7×7
        Flatten → Linear → (μ, log_var)
    """
    def __init__(self, latent_dim, hidden_channels=(32, 64)):
        super().__init__()
        c1, c2 = hidden_channels
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.flatten_dim = c2 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: latent z → 28×28 이미지 복원

    구조:
        Linear → Reshape (64, 7, 7)
        ConvT(64→32, stride=2) → ReLU   # 7×7 → 14×14
        ConvT(32→1, stride=2) → Sigmoid # 14×14 → 28×28
    """
    def __init__(self, latent_dim, hidden_channels=(32, 64)):
        super().__init__()
        c1, c2 = hidden_channels
        self.fc = nn.Linear(latent_dim, c2 * 7 * 7)
        self.c2 = c2
        self.deconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, self.c2, 7, 7)
        h = F.relu(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))


class VAE(nn.Module):
    """
    Variational Autoencoder.

    핵심 구성:
        - Encoder: x → (μ, log σ²)
        - Reparameterization: z = μ + σ ⊙ ε,  ε ~ N(0, I)
        - Decoder: z → x_recon
    """
    def __init__(self, latent_dim=LATENT_DIM, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hidden_channels)
        self.decoder = Decoder(latent_dim, hidden_channels)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick — gradient가 μ, σ를 통해 흐를 수 있게 함."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, n_samples, device=DEVICE):
        """Prior N(0, I)에서 샘플링해서 새 이미지 생성."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ───────────────────────────────────────────────────────────
# 5. Loss — ELBO (Reconstruction + KL Divergence)
# ───────────────────────────────────────────────────────────
def vae_loss(x_recon, x, mu, logvar, beta=BETA):
    """
    ELBO = -log p(x|z) + β * D_KL(q(z|x) || p(z))

    구성:
        - Reconstruction Loss: BCE (Bernoulli 가정), per sample 평균
        - KL Divergence: -0.5 * Σ(1 + log σ² - μ² - σ²), per sample 평균
    """
    # BCE의 'sum' reduction → 픽셀 합 → 배치 평균
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon + beta * kl
    return total, recon, kl


# ───────────────────────────────────────────────────────────
# 6. 학습 / 평가
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mu, logvar)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon.item() * bs
        total_kl += kl.item() * bs
        n += bs
    model.train()
    return total_loss / n, total_recon / n, total_kl / n


def train_loop(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []

    print("[train] start", flush=True)
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_total = epoch_recon = epoch_kl = 0.0
        n_seen = 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            x_recon, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_recon, x, mu, logvar)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            epoch_total += loss.item() * bs
            epoch_recon += recon.item() * bs
            epoch_kl += kl.item() * bs
            n_seen += bs

        train_total = epoch_total / n_seen
        train_recon = epoch_recon / n_seen
        train_kl = epoch_kl / n_seen
        val_total, val_recon, val_kl = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch:3d}] "
            f"train: total={train_total:.2f} recon={train_recon:.2f} kl={train_kl:.3f} | "
            f"val: total={val_total:.2f} recon={val_recon:.2f} kl={val_kl:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_total": train_total, "train_recon": train_recon, "train_kl": train_kl,
            "val_total": val_total,     "val_recon": val_recon,     "val_kl": val_kl,
        })

    print(f"[train] total time: {(time.time() - t0) / 60:.1f} min")
    return history


# ───────────────────────────────────────────────────────────
# 7. 시각화
# ───────────────────────────────────────────────────────────
def plot_dataset_overview(train_set):
    """01: FashionMNIST 샘플 + 클래스 분포 + 통계."""
    fig = plt.figure(figsize=(14, 8))

    # (a) 클래스별 샘플 이미지 1장씩
    ax_list = []
    sample_per_class = {}
    for img, lbl in train_set:
        if lbl not in sample_per_class:
            sample_per_class[lbl] = img
        if len(sample_per_class) == 10:
            break

    for i in range(10):
        ax = plt.subplot2grid((3, 5), (i // 5, i % 5))
        ax.imshow(sample_per_class[i].squeeze().numpy(), cmap="gray")
        ax.set_title(f"{i}: {CLASS_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.axis("off")
        ax_list.append(ax)

    # (b) 클래스 분포 + 통계 텍스트
    ax_dist = plt.subplot2grid((3, 5), (2, 0), colspan=3)
    targets = train_set.targets.numpy() if hasattr(train_set.targets, "numpy") else np.array(train_set.targets)
    counts = np.bincount(targets, minlength=10)
    bars = ax_dist.bar(range(10), counts, color="steelblue")
    ax_dist.set_xticks(range(10))
    ax_dist.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax_dist.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax_dist.set_title("Train Set Class Distribution", fontsize=13, fontweight="bold")
    ax_dist.grid(True, alpha=0.3, axis="y")

    ax_stats = plt.subplot2grid((3, 5), (2, 3), colspan=2)
    ax_stats.axis("off")
    stats_lines = [
        f"Image size      : 28 × 28 (grayscale)",
        f"Train samples   : {len(train_set):,}",
        f"Number of classes: 10",
        f"Pixel range     : [0, 1]",
        f"Latent dim      : {LATENT_DIM}",
    ]
    ax_stats.text(0.0, 0.85, "\n".join(stats_lines),
                  fontsize=12, family="monospace", fontweight="bold", va="top")
    ax_stats.set_title("Dataset Statistics", loc="left", fontsize=13, fontweight="bold")

    plt.suptitle("FashionMNIST Dataset Overview", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_dataset_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 01_dataset_overview.png")


def plot_training_curve(history):
    """02: ELBO 분해 — Total / Recon / KL Loss 곡선 + Best Epoch 마커."""
    epochs = [h["epoch"] for h in history]
    train_total = [h["train_total"] for h in history]
    train_recon = [h["train_recon"] for h in history]
    train_kl    = [h["train_kl"]    for h in history]
    val_total   = [h["val_total"]   for h in history]
    val_recon   = [h["val_recon"]   for h in history]
    val_kl      = [h["val_kl"]      for h in history]

    # Best Epoch = Val Total Loss(ELBO)가 가장 낮은 시점
    best_idx = int(np.argmin(val_total))
    best_epoch = epochs[best_idx]
    best_val = val_total[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, (ax, (t_train, t_val, title, color_train, color_val)) in enumerate(zip(
        axes,
        [
            (train_total, val_total, "Total Loss (ELBO)", "steelblue", "crimson"),
            (train_recon, val_recon, "Reconstruction Loss (BCE)", "darkorange", "darkred"),
            (train_kl, val_kl, "KL Divergence", "darkgreen", "purple"),
        ],
    )):
        ax.plot(epochs, t_train, label="Train", color=color_train, marker="o", linewidth=2.2, markersize=6)
        ax.plot(epochs, t_val, label="Val", color=color_val, marker="s", linewidth=2.2, markersize=6)
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=10)

        # 첫 번째 패널(Total Loss)에만 Best Epoch 마커 추가
        if i == 0:
            ax.axvline(best_epoch, color="forestgreen", linestyle="--", linewidth=1.6, alpha=0.7)
            ax.scatter([best_epoch], [best_val], s=180, color="gold",
                       edgecolor="darkgreen", linewidth=2.2, zorder=5,
                       label=f"Best Val ELBO = {best_val:.2f} @ epoch {best_epoch}")
            # 값 annotation
            ax.annotate(f"{best_val:.2f}",
                        xy=(best_epoch, best_val),
                        xytext=(best_epoch + 1.5, best_val + 4),
                        fontsize=11, fontweight="bold", color="darkgreen",
                        arrowprops=dict(arrowstyle="->", color="darkgreen", alpha=0.7))
            ax.legend(fontsize=10, loc="upper right")
        else:
            ax.legend(fontsize=11)

    plt.suptitle("VAE Training Curves  —  ELBO = Reconstruction + β·KL", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_training_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 02_training_curve.png")


def plot_reconstruction(model, val_set, n_samples=10):
    """03: Original vs Reconstructed 비교."""
    model.eval()
    # 클래스별로 한 장씩 가져오기
    samples = {}
    for img, lbl in val_set:
        lbl = int(lbl)
        if lbl not in samples:
            samples[lbl] = img
        if len(samples) == n_samples:
            break
    originals = torch.stack([samples[i] for i in range(n_samples)]).to(DEVICE)

    with torch.no_grad():
        recons, _, _ = model(originals)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.6, 4.0))
    for i in range(n_samples):
        axes[0, i].imshow(originals[i].cpu().squeeze(), cmap="gray")
        axes[0, i].set_title(CLASS_NAMES[i], fontsize=10, fontweight="bold")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12, fontweight="bold", rotation=90, labelpad=15)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12, fontweight="bold", rotation=90, labelpad=15)

    # ylabel은 axis off 시 사라지므로 figure text로 표시
    fig.text(0.02, 0.72, "Original",       fontsize=13, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.27, "Reconstructed",  fontsize=13, fontweight="bold", rotation=90, va="center")

    plt.suptitle("Reconstruction Quality  —  Original (top) vs Reconstructed (bottom)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_reconstruction.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 03_reconstruction.png")


def plot_latent_space(model, val_loader, n_points=3000):
    """04: 학습된 latent space — t-SNE 2D 투영, 클래스별 색상."""
    model.eval()
    all_mu = []
    all_labels = []
    seen = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            mu, _ = model.encoder(x)
            all_mu.append(mu.cpu().numpy())
            all_labels.append(y.numpy())
            seen += x.size(0)
            if seen >= n_points:
                break
    mus = np.concatenate(all_mu)[:n_points]
    labels = np.concatenate(all_labels)[:n_points]

    print(f"[viz] running t-SNE on {len(mus)} points (latent_dim={mus.shape[1]})...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca")
    mus_2d = tsne.fit_transform(mus)

    fig, ax = plt.subplots(figsize=(13, 10))
    cmap = plt.get_cmap("tab10")
    for cls in range(10):
        mask = labels == cls
        ax.scatter(mus_2d[mask, 0], mus_2d[mask, 1],
                   c=[cmap(cls)], label=CLASS_NAMES[cls], s=22, alpha=0.75,
                   edgecolors="none")
    ax.set_xlabel("t-SNE dim 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("t-SNE dim 2", fontsize=14, fontweight="bold")
    ax.set_title(f"Latent Space (t-SNE of μ)  —  {len(mus)} points, latent_dim={LATENT_DIM}",
                 fontsize=16, fontweight="bold", pad=12)
    # legend를 plot 바깥(오른쪽)으로 빼서 데이터 가리지 않게
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=12, markerscale=2.5, framealpha=0.95,
              title="Class", title_fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_latent_space.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 04_latent_space.png")


def plot_generation_and_interpolation(model, val_set):
    """05: Random Sampling + 4쌍 Latent Interpolation."""
    model.eval()

    # (a) Prior N(0, I)에서 16개 랜덤 샘플 (4×4 grid)
    with torch.no_grad():
        samples = model.sample(16, device=DEVICE)

    # (b) 4쌍 클래스 간 보간 — 의미적 카테고리를 다양하게 커버
    interp_pairs = [
        (0, 9),  # T-shirt/top  →  Ankle boot   (극단적: 상의 → 신발)
        (0, 2),  # T-shirt/top  →  Pullover    (비슷한 상의끼리)
        (1, 3),  # Trouser      →  Dress       (하의 → 원피스)
        (7, 9),  # Sneaker      →  Ankle boot  (신발끼리)
    ]
    n_steps = 10
    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1)

    all_interp_imgs = []
    for src_class, dst_class in interp_pairs:
        src_img = next(img for img, lbl in val_set if lbl == src_class)
        dst_img = next(img for img, lbl in val_set if lbl == dst_class)
        with torch.no_grad():
            mu_src, _ = model.encoder(src_img.unsqueeze(0).to(DEVICE))
            mu_dst, _ = model.encoder(dst_img.unsqueeze(0).to(DEVICE))
            z_interp = (1 - alphas) * mu_src + alphas * mu_dst
            interp_imgs = model.decoder(z_interp)
        all_interp_imgs.append(interp_imgs)

    # 레이아웃: 위 = 4×4 random samples, 아래 = 4 pairs × 10 steps interpolation
    fig = plt.figure(figsize=(15, 14))

    # (a) Random samples 4×4 grid (위쪽)
    gs_top = fig.add_gridspec(4, 4, left=0.30, right=0.70, top=0.95, bottom=0.55, hspace=0.08, wspace=0.06)
    for i in range(16):
        ax = fig.add_subplot(gs_top[i // 4, i % 4])
        ax.imshow(samples[i].cpu().squeeze(), cmap="gray")
        ax.axis("off")
    fig.text(0.5, 0.97, "(a) Random Samples from N(0, I)",
             fontsize=16, fontweight="bold", ha="center")

    # (b) 4쌍 Latent interpolation (아래쪽)
    gs_bottom = fig.add_gridspec(4, n_steps, left=0.18, right=0.96, top=0.48, bottom=0.07,
                                  hspace=0.15, wspace=0.05)
    for row, ((src_class, dst_class), interp_imgs) in enumerate(zip(interp_pairs, all_interp_imgs)):
        for col in range(n_steps):
            ax = fig.add_subplot(gs_bottom[row, col])
            ax.imshow(interp_imgs[col].cpu().squeeze(), cmap="gray")
            ax.axis("off")
        # 각 행 왼쪽에 클래스 쌍 레이블
        # gridspec의 row top/bottom 위치 계산 (top=0.48, bottom=0.07, 4 rows)
        row_h = (0.48 - 0.07) / 4
        row_y = 0.48 - row_h * (row + 0.5) - 0.005  # 행 중앙
        fig.text(0.17, row_y,
                 f"{CLASS_NAMES[src_class]}\n→\n{CLASS_NAMES[dst_class]}",
                 fontsize=10.5, fontweight="bold", ha="right", va="center",
                 color="#222222")

    fig.text(0.5, 0.51, "(b) Latent Space Interpolation  —  4 class pairs",
             fontsize=16, fontweight="bold", ha="center")
    # α 값 표시 (전체 그리드 양 끝)
    fig.text(0.18, 0.045, "α = 0.0", fontsize=11, ha="left", color="#666666", fontweight="bold")
    fig.text(0.96, 0.045, "α = 1.0", fontsize=11, ha="right", color="#666666", fontweight="bold")

    plt.suptitle("Generation & Interpolation in Latent Space", fontsize=18, fontweight="bold", y=1.00)
    plt.savefig(RESULTS_DIR / "05_generation_interpolation.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 05_generation_interpolation.png")


def plot_latent_traversal(model, val_loader, n_top_dims=8, n_steps=9):
    """06: Latent Traversal — 활성 상위 차원이 무엇을 capture하는지 시각화.

    각 latent dimension이 데이터에 대해 갖는 평균 KL contribution을 계산해서,
    가장 활성화된 상위 n_top_dims 개의 차원만 골라 표시.
    base z를 0 vector로 두고 한 번에 한 차원만 [-3, +3] 범위로 변화시키며 디코딩.
    """
    model.eval()
    all_mu, all_logvar = [], []
    seen = 0
    n_target = 3000
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(DEVICE)
            mu, logvar = model.encoder(x)
            all_mu.append(mu)
            all_logvar.append(logvar)
            seen += x.size(0)
            if seen >= n_target:
                break
    mus = torch.cat(all_mu)[:n_target]
    logvars = torch.cat(all_logvar)[:n_target]

    # KL per dimension (배치 평균): -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp()).mean(dim=0)
    kl_per_dim_np = kl_per_dim.cpu().numpy()

    # 활성도 상위 n_top_dims개 (KL이 큰 = 정보를 많이 담은 차원)
    top_dims = torch.topk(kl_per_dim, n_top_dims).indices.cpu().numpy()
    print(f"[viz] latent traversal - top {n_top_dims} active dims by KL: "
          + ", ".join([f"dim{d}={kl_per_dim_np[d]:.2f}" for d in top_dims]))

    # base = 0 vector (prior 평균), 한 dim씩 [-3, +3]로 변화
    base = torch.zeros(1, LATENT_DIM, device=DEVICE)
    alphas = torch.linspace(-3, 3, n_steps, device=DEVICE)

    fig, axes = plt.subplots(n_top_dims, n_steps,
                             figsize=(n_steps * 1.3 + 1.2, n_top_dims * 1.4))

    with torch.no_grad():
        for i, dim_i in enumerate(top_dims):
            for j, alpha in enumerate(alphas):
                z = base.clone()
                z[0, int(dim_i)] = alpha.item()
                x_recon = model.decoder(z).cpu().squeeze().numpy()
                axes[i, j].imshow(x_recon, cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

            # 행 레이블: dim 번호 + KL 값
            axes[i, 0].set_ylabel(f"dim {int(dim_i):2d}\nKL={kl_per_dim_np[int(dim_i)]:.2f}",
                                  fontsize=11, fontweight="bold",
                                  rotation=0, ha="right", va="center", labelpad=18)

    # 열 레이블: α 값 (첫 행 위쪽)
    for j, alpha in enumerate(alphas):
        axes[0, j].set_title(f"α = {alpha.item():+.1f}",
                             fontsize=11, fontweight="bold", pad=8)

    fig.suptitle(f"Latent Traversal  —  Top {n_top_dims} Active Dimensions (sorted by per-dim KL)",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_latent_traversal.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[viz] saved 06_latent_traversal.png")


# ───────────────────────────────────────────────────────────
# 8. 메인 파이프라인 (체크포인트 save/load 지원)
# ───────────────────────────────────────────────────────────
CHECKPOINT_PATH = DATA_DIR / "checkpoint.pt"


def build_model():
    return VAE(latent_dim=LATENT_DIM, hidden_channels=HIDDEN_CHANNELS).to(DEVICE)


def main():
    """
    실행 옵션:
        python src/main.py             # checkpoint 있으면 load, 없으면 학습 후 저장
        python src/main.py --retrain   # 강제 재학습
        python src/main.py --viz-only  # checkpoint만 load, 학습 스킵
    """
    force_retrain = "--retrain" in sys.argv
    viz_only = "--viz-only" in sys.argv

    print(f"[info] device      : {DEVICE}")
    print(f"[info] latent dim  : {LATENT_DIM}")
    print(f"[info] batch size  : {BATCH_SIZE}")
    print(f"[info] epochs      : {EPOCHS}", flush=True)

    train_loader, val_loader, train_set, val_set = get_dataloaders()
    print(f"[info] train size  : {len(train_set):,}")
    print(f"[info] val   size  : {len(val_set):,}", flush=True)

    # 시각화 01: 데이터셋 개요 (학습 무관)
    plot_dataset_overview(train_set)

    model = build_model()
    print(f"[info] model params: {model.num_params() / 1e3:.1f} K", flush=True)

    if CHECKPOINT_PATH.exists() and not force_retrain:
        print(f"[load] checkpoint found at {CHECKPOINT_PATH.name}, skipping training", flush=True)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        history = ckpt["history"]
    else:
        if viz_only:
            raise FileNotFoundError(f"--viz-only 모드인데 checkpoint가 없습니다: {CHECKPOINT_PATH}")
        history = train_loop(model, train_loader, val_loader)
        torch.save(
            {"model": model.state_dict(), "history": history},
            CHECKPOINT_PATH,
        )
        print(f"[save] checkpoint saved to {CHECKPOINT_PATH.name}", flush=True)

    # 시각화 02~06
    plot_training_curve(history)
    plot_reconstruction(model, val_set)
    plot_latent_space(model, val_loader)
    plot_generation_and_interpolation(model, val_set)
    plot_latent_traversal(model, val_loader)

    print("[done] all visualizations saved to results/")


if __name__ == "__main__":
    main()
