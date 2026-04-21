"""
siamese.py — обучение и оценка сиамской нейросети для сравнения фавиконок.

Структура данных:
    data/
        anchor/         — эталонные иконки брендов (favicon1.png, favicon2.png, ...)
        pos/            — позитивные примеры (те же бренды, другие варианты)
        neg_all/        — все негативные примеры (перемешиваются и делятся на сплиты)

Запуск:
    python siamese.py                        # обучение с дефолтными параметрами
    python siamese.py --epochs 50 --lr 1e-4  # кастомные параметры
    python siamese.py --eval-only best_model.pth  # только оценка
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

# Рандом
SEED = 808
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SiameseNetwork(nn.Module):
    """
    Сиамская сеть для метрического обучения на фавиконках.

    4 блока свёрток (Conv→BN→ReLU×2 → Pool) + AdaptiveAvgPool → MLP → L2-норм.
    Вход: 48×48 RGB. Выход:  вектор размерности embedding_dim.

    Для обучения используется TripletLoss
    """

    def __init__(self, embedding_dim: int = 64, input_size: int = 48):
        super().__init__()
        self.input_size = input_size

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.AdaptiveAvgPool2d(1))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(3,   32,  pool=True)
        self.conv2 = conv_block(32,  64,  pool=True)
        self.conv3 = conv_block(64,  128, pool=True)
        self.conv4 = conv_block(128, 256, pool=False)

        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход для одного изображения"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, anchor, pos, neg=None):
        anchor_emb = self.forward_one(anchor)
        pos_emb = self.forward_one(pos)
        if neg is not None:
            neg_emb = self.forward_one(neg)
            return anchor_emb, pos_emb, neg_emb
        return anchor_emb, pos_emb


class SimilarityClassifier(nn.Module):
    """
    Лёгкая голова-классификатор поверх пары эмбеддингов.

    Принимает два эмбеддинга e1, e2 и возвращает вероятность того,
    что они похожи.

    Признаки: [|e1 - e2|, e1 * e2] — разность и поэлементное произведение.

    Используется поверх основной сиамской нейросети
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),   # логит; применяй sigmoid снаружи
        )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1, emb2: эмбеддинги, shape (B, embedding_dim)
        """
        diff    = torch.abs(emb1 - emb2)         # |e1 - e2|
        product = emb1 * emb2                     # e1 ⊙ e2
        combined = torch.cat([diff, product], dim=1)
        return self.classifier(combined).squeeze(1)


# ── Функции потерь ────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet loss
    margin - граница (минимум на который максимум дальше от минимума).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        d_pos = F.pairwise_distance(anchor, pos, p=2)
        d_neg = F.pairwise_distance(anchor, neg, p=2)
        loss = F.relu(d_pos - d_neg + self.margin)
        # Доля триплетов где loss > 0 (active triplets) — полезный индикатор
        active = (loss > 0).float().mean().item()
        return loss.mean(), active


# ── Аугментации ───────────────────────────────────────────────────────────────

class ReplaceBackground:
    """
    Заменяет фон иконки на случайный цвет.

    mode='auto': если есть прозрачность — использует альфа-маску,
                 иначе заменяет почти-белые пиксели (>220 по всем каналам).
    mode='alpha': всегда по альфа-каналу.
    mode='white': всегда по порогу белого.
    """

    def __init__(self, p: float = 0.5, mode: str = 'auto'):
        self.p = p
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        bg_color = tuple(np.random.randint(50, 200, 3).tolist())
        img_rgba = img.convert('RGBA')

        mode = self.mode
        if mode == 'auto':
            a_ch = np.array(img_rgba.split()[3])
            mode = 'alpha' if (a_ch < 128).any() else 'white'

        if mode == 'alpha':
            bg = Image.new('RGBA', img.size, bg_color + (255,))
            bg.paste(img_rgba, mask=img_rgba.split()[3])
            return bg.convert('RGB')
        else:
            data = np.array(img_rgba)
            mask = (data[:, :, 0] > 220) & (data[:, :, 1] > 220) & (data[:, :, 2] > 220)
            data[mask] = [*bg_color, 255]
            return Image.fromarray(data).convert('RGB')


def make_aug(replace_bg: bool = False) -> transforms.Compose:
    """Возвращает аугментации для фавиконок"""
    steps = []
    if replace_bg:
        steps.append(ReplaceBackground(p=0.5, mode='auto'))
    steps += [
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.05)
        ], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.RandomApply([
            transforms.ColorJitter(hue=0.05, saturation=0.2, brightness=0.1)
        ], p=0.6),
        transforms.Resize((48, 48)),
    ]
    return transforms.Compose(steps)


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Датасет ───────────────────────────────────────────────────────────────────

class FaviconTripletDataset(Dataset):
    """
    Датасет триплетов для одного бренда.

    Каждый элемент: (anchor_aug, pos_aug, neg_random).
    anchor и pos — аугментации одного и того же оригинального изображения.
    neg — случайный файл из neg_files (список передаётся снаружи для контроля сплита).

    Args:
        anchor_path:  путь к эталонной иконке бренда
        neg_files:    список Path к негативным примерам (уже разбит на сплит снаружи)
        num_samples:  длина датасета (сколько триплетов генерировать за эпоху)
        augmentation: pipeline аугментации для anchor/pos
        transform:    финальный transform (ToTensor + Normalize)
    """

    def __init__(
        self,
        anchor_path: str,
        neg_files: list,
        num_samples: int = 900,
        augmentation: transforms.Compose = None,
        transform: transforms.Compose = None,
    ):
        self.original = Image.open(anchor_path).convert('RGB')
        self.neg_files = neg_files
        self.num_samples = num_samples
        self.augmentation = augmentation or make_aug(replace_bg=False)
        self.transform = transform or DEFAULT_TRANSFORM

        if not self.neg_files:
            raise ValueError(f"Нет негативных примеров для {anchor_path}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        anchor = self.augmentation(self.original.copy())
        pos    = self.augmentation(self.original.copy())

        # Случайный негатив с повтором при битом файле
        neg = None
        while neg is None:
            path = random.choice(self.neg_files)
            try:
                neg = Image.open(path).convert('RGB')
            except Exception:
                pass

        return (
            self.transform(anchor),
            self.transform(pos),
            self.transform(neg),
        )



def split_neg_files(neg_dir: str, val_frac: float = 0.15, test_frac: float = 0.10):
    """
    Делит файлы из neg_dir на train / val / test.

    Разбивка происходит один раз по файлам — одни и те же иконки
    не попадают в разные сплиты, что исключает утечку данных.

    Returns:
        (train_files, val_files, test_files)
    """
    exts = {'*.ico', '*.ICO', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg'}
    files = []
    for ext in exts:
        files.extend(Path(neg_dir).glob(ext))
    files = sorted(set(files))  # убираем дубликаты из-за разных масок

    random.shuffle(files)
    n = len(files)
    n_val  = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))

    val_files   = files[:n_val]
    test_files  = files[n_val:n_val + n_test]
    train_files = files[n_val + n_test:]

    print(f"Негативы: {n} всего | train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    return train_files, val_files, test_files


# ── Метрики ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: SiameseNetwork, loader: DataLoader, criterion: TripletLoss,
             device: str) -> dict:
    """
    Считает метрики на loader.

    Метрики:
        loss          — средний triplet loss
        active_frac   — доля активных триплетов (loss > 0), должна снижаться
        pair_acc      — точность на парах: pos ближе neg в X% триплетов
        mean_pos_sim  — средняя косинусная схожесть pos-пар (хотим → 1)
        mean_neg_sim  — средняя косинусная схожесть neg-пар (хотим → 0)
        roc_auc       — AUC на бинарной задаче (pos=1, neg=0) по косинусу
        threshold     — порог cosine_sim при котором F1 максимален
        f1_at_thresh  — F1 при найденном пороге
    """
    model.eval()
    model.to(device)

    all_losses, all_active = [], []
    pos_sims, neg_sims = [], []
    correct_pairs = 0
    total_pairs = 0

    for anchor, pos, neg in loader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        a_emb, p_emb, n_emb = model(anchor, pos, neg)

        loss, active = criterion(a_emb, p_emb, n_emb)
        all_losses.append(loss.item())
        all_active.append(active)

        # Косинусная схожесть → [0, 1]
        cos_pos = ((F.cosine_similarity(a_emb, p_emb) + 1) / 2).cpu().numpy()
        cos_neg = ((F.cosine_similarity(a_emb, n_emb) + 1) / 2).cpu().numpy()

        pos_sims.extend(cos_pos.tolist())
        neg_sims.extend(cos_neg.tolist())

        # pair accuracy: pos должен быть ближе neg
        d_pos = F.pairwise_distance(a_emb, p_emb)
        d_neg = F.pairwise_distance(a_emb, n_emb)
        correct_pairs += (d_pos < d_neg).sum().item()
        total_pairs += anchor.size(0)

    # ROC-AUC на бинарной задаче (pos=1, neg=0)
    scores = pos_sims + neg_sims
    labels = [1] * len(pos_sims) + [0] * len(neg_sims)
    auc = roc_auc_score(labels, scores)

    # Лучший порог по F1
    fpr, tpr, thresholds = roc_curve(labels, scores)
    precision = tpr / (tpr + fpr + 1e-8)
    recall = tpr
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return {
        'loss':           np.mean(all_losses),
        'active_frac':    np.mean(all_active),
        'pair_acc':       correct_pairs / total_pairs,
        'mean_pos_sim':   np.mean(pos_sims),
        'mean_neg_sim':   np.mean(neg_sims),
        'roc_auc':        auc,
        'threshold':      best_threshold,
        'f1_at_thresh':   best_f1,
    }


def print_metrics(prefix: str, m: dict):
    print(
        f"  {prefix:5s} | loss={m['loss']:.4f} active={m['active_frac']:.2f}"
        f" | pair_acc={m['pair_acc']:.3f} auc={m['roc_auc']:.3f}"
        f" | pos_sim={m['mean_pos_sim']:.3f} neg_sim={m['mean_neg_sim']:.3f}"
        f" | best_thr={m['threshold']:.3f} f1={m['f1_at_thresh']:.3f}"
    )


# ── Обучение ──────────────────────────────────────────────────────────────────

def train(
    anchor_paths: list[str],
    neg_dir: str,
    num_epochs: int = 40,
    lr: float = 3e-5,
    batch_size: int = 48,
    embedding_dim: int = 48,
    margin: float = 0.5,
    device: str = 'mps',
    num_samples_per_logo: int = 900,
    save_path: str = 'best_siamese.pth',
    cross_variant_weight: float = 0.5,
):
    """
    Двухфазное обучение:

    1 - Triplet loss на SiameseNetwork.

    2 - BCE loss на SimilarityClassifier (веса SiameseNetwork заморожены).
        Обучаем лёгкую голову предсказывать похож/не похож по паре эмбеддингов.

    Args:
        anchor_paths:         пути ко всем вариантам иконки одного бренда
        neg_dir:              папка со всеми негативами
        num_epochs:           эпох для фазы 1
        lr:                   learning rate
        batch_size:           размер батча
        embedding_dim:        размерность эмбеддинга
        margin:               отступ в triplet loss
        device:               'mps' / 'cuda' / 'cpu'
        num_samples_per_logo: триплетов на вариант иконки за эпоху
        save_path:            куда сохранять лучшую модель
        cross_variant_weight: доля кросс-вариантных триплетов (0..1)
    """
    train_neg, val_neg, test_neg = split_neg_files(neg_dir)

    aug_plain  = make_aug(replace_bg=False)
    aug_bg     = make_aug(replace_bg=True)

    def make_loaders(neg_files, n_samples):
        """Возвращает датаесеты (опционально если лого похожи можно добавить кроссдатасетов)"""
        datasets = []
        # Обычные триплеты — по одному варианту
        n_plain = int(n_samples * (1 - cross_variant_weight))
        for i, path in enumerate(anchor_paths):
            aug = aug_bg if i % 2 == 1 else aug_plain
            datasets.append(FaviconTripletDataset(
                anchor_path=path,
                neg_files=neg_files,
                num_samples=n_plain,
                augmentation=aug,
            ))
        return ConcatDataset(datasets)

    train_dataset = make_loaders(train_neg, num_samples_per_logo)
    val_dataset   = make_loaders(val_neg,   num_samples_per_logo // 3)
    test_dataset  = make_loaders(test_neg,  num_samples_per_logo // 3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    # обучение SiameseNetwork
    print("=" * 80)
    print("!1: обучение основной части")
    print(f"  Устройство: {device} | Эмбеддинг: {embedding_dim}d | margin: {margin}")
    print(f"  Вариантов иконки: {len(anchor_paths)} | cross_variant_weight: {cross_variant_weight}")
    print(f"  Негативов train/val/test: {len(train_neg)}/{len(val_neg)}/{len(test_neg)}")
    print("=" * 80)

    model     = SiameseNetwork(embedding_dim=embedding_dim).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    best_val_auc      = 0.0
    patience_counter  = 0
    early_stop_patience = 15

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = total_active = 0.0

        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            a_emb, p_emb, n_emb = model(anchor, pos, neg)
            loss, active = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss   += loss.item()
            total_active += active

        avg_loss   = total_loss   / len(train_loader)
        avg_active = total_active / len(train_loader)
        val_m      = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m['loss'])

        print(f"\nЭпоха {epoch:3d}/{num_epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"  train | loss={avg_loss:.4f} active={avg_active:.2f}")
        print_metrics('val', val_m)

        if val_m['roc_auc'] > best_val_auc:
            best_val_auc     = val_m['roc_auc']
            patience_counter = 0
            # Конвертируем numpy floats → Python float чтобы weights_only=True работал
            safe_metrics = {k: float(v) for k, v in val_m.items()}
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': safe_metrics,
                'embedding_dim': embedding_dim,
                'margin': float(margin),
            }, save_path)
            print(f"    Лучшая модель сохранена (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping после {epoch} эпох")
                break

    # обучение SimilarityClassifier
    print("\n" + "=" * 80)
    print("!2 обчение головки классификатора")
    print("=" * 80)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():   # замораживаем backbone
        param.requires_grad = False

    clf       = SimilarityClassifier(embedding_dim=embedding_dim).to(device)
    bce       = nn.BCEWithLogitsLoss()
    clf_opt   = torch.optim.Adam(clf.parameters(), lr=1e-3)
    best_clf_auc = 0.0

    for epoch in range(1, 21):          # 20 эпох достаточно
        clf.train()
        total_loss = 0.0

        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            with torch.no_grad():
                a_emb = model.forward_one(anchor)
                p_emb = model.forward_one(pos)
                n_emb = model.forward_one(neg)

            clf_opt.zero_grad()
            # pos-пары: label=1, neg-пары: label=0
            logits_pos = clf(a_emb, p_emb)
            logits_neg = clf(a_emb, n_emb)
            logits = torch.cat([logits_pos, logits_neg])
            labels = torch.cat([
                torch.ones(logits_pos.size(0), device=device),
                torch.zeros(logits_neg.size(0), device=device),
            ])
            loss = bce(logits, labels)
            loss.backward()
            clf_opt.step()
            total_loss += loss.item()

        # Быстрая валидация классификатора
        clf.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for anchor, pos, neg in val_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                a_emb = model.forward_one(anchor)
                p_emb = model.forward_one(pos)
                n_emb = model.forward_one(neg)
                p_pos = torch.sigmoid(clf(a_emb, p_emb)).cpu().numpy()
                p_neg = torch.sigmoid(clf(a_emb, n_emb)).cpu().numpy()
                val_probs.extend(p_pos.tolist() + p_neg.tolist())
                val_labels.extend([1] * len(p_pos) + [0] * len(p_neg))

        clf_auc = roc_auc_score(val_labels, val_probs)
        print(f"  Эпоха {epoch:2d}/20 | bce={total_loss/len(train_loader):.4f}"
              f" | clf_val_auc={clf_auc:.4f}")

        if clf_auc > best_clf_auc:
            best_clf_auc = clf_auc
            clf_save = save_path.replace('.pth', '_clf.pth')
            torch.save({
                'classifier_state_dict': clf.state_dict(),
                'embedding_dim': embedding_dim,
                'val_auc': float(clf_auc),
            }, clf_save)
            print(f"    Лучший классификатор сохранён (auc={clf_auc:.4f}) -> {clf_save}")

    # файнали
    print("\n" + "=" * 80)
    print("Финальная оценка на тестовой выборке:")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False)['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = True

    test_m = evaluate(model, test_loader, criterion, device)
    print_metrics('test', test_m)
    print(f"\n  Рекомендуемый порог cosine similarity: {test_m['threshold']:.3f}")
    print(f"  При пороге: F1={test_m['f1_at_thresh']:.3f}, AUC={test_m['roc_auc']:.3f}")
    print(f"\n  Классификатор AUC (val): {best_clf_auc:.4f}")
    print(f"  Сохранён в: {clf_save}")

    return model, clf, test_m


# инф

@torch.no_grad()
def compare_favicons(
    model: SiameseNetwork,
    img1_path: str,
    img2_path: str,
    classifier: 'SimilarityClassifier | None' = None,
    threshold: float = 0.7,
    device: str = 'mps',
) -> dict:
    """
    Сравнивает две иконки.

    Возвращает сырые метрики схожести и,
    если передан классификатор, бинарное решение.

    Args:
        model:       обученная SiameseNetwork
        img1_path:   путь к первой иконке (эталон бренда)
        img2_path:   путь ко второй иконке (проверяемая)
        classifier:  SimilarityClassifier (опционально)
        threshold:   порог для cosine similarity если классификатора нет
        device:      устройство

    Returns:
        similarity:    косинусная схожесть ∈ [0, 1], 1 = идентичны
        euc_dist:      евклидово расстояние ∈ [0, 2], 0 = идентичны
        is_same_brand: bool — бинарное решение
        confidence:    вероятность того что is_same_brand=True ∈ [0, 1]
        decision_by:   'classifier' или 'threshold'
    """
    model.eval()
    model.to(device)

    img1 = DEFAULT_TRANSFORM(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
    img2 = DEFAULT_TRANSFORM(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)

    emb1 = model.forward_one(img1)
    emb2 = model.forward_one(img2)

    cos  = F.cosine_similarity(emb1, emb2).item()
    dist = F.pairwise_distance(emb1, emb2).item()
    similarity = (cos + 1) / 2

    if classifier is not None:
        classifier.eval()
        classifier.to(device)
        logit = classifier(emb1, emb2).item()
        confidence = float(torch.sigmoid(torch.tensor(logit)))
        is_same = confidence >= 0.5
        decision_by = 'classifier'
    else:
        confidence = similarity
        is_same = similarity >= threshold
        decision_by = 'threshold'

    return {
        'similarity':    round(similarity, 4),
        'euc_dist':      round(dist, 4),
        'is_same_brand': is_same,
        'confidence':    round(confidence, 4),
        'decision_by':   decision_by,
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение сиамской сети для фавиконок')
    parser.add_argument('--anchors',   nargs='+',
                        default=['data/anchor/favicon1.png',
                                 'data/anchor/favicon2.png',
                                 'data/anchor/favicon3.png'])
    parser.add_argument('--neg-dir',   default='data/neg_all')
    parser.add_argument('--epochs',    type=int,   default=40)
    parser.add_argument('--lr',        type=float, default=3e-5)
    parser.add_argument('--batch',     type=int,   default=48)
    parser.add_argument('--emb-dim',   type=int,   default=48)
    parser.add_argument('--margin',    type=float, default=0.5)
    parser.add_argument('--device',    default='mps')
    parser.add_argument('--save',      default='best_siamese.pth')
    parser.add_argument('--eval-only', default=None,
                        help='Путь к .pth — только оценка, без обучения')
    args = parser.parse_args()

    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=args.device, weights_only=False)
        model = SiameseNetwork(
            embedding_dim=ckpt.get('embedding_dim', args.emb_dim)
        )
        model.load_state_dict(ckpt['model_state_dict'])
        _, val_neg, test_neg = split_neg_files(args.neg_dir)
        test_dataset = ConcatDataset([
            FaviconTripletDataset(p, test_neg, num_samples=300)
            for p in args.anchors
        ])
        test_loader = DataLoader(test_dataset, batch_size=args.batch,
                                 shuffle=False, num_workers=0)
        criterion = TripletLoss(margin=ckpt.get('margin', args.margin))
        metrics = evaluate(model, test_loader, criterion, args.device)
        print_metrics('test', metrics)
    else:
        train(
            anchor_paths=args.anchors,
            neg_dir=args.neg_dir,
            num_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch,
            embedding_dim=args.emb_dim,
            margin=args.margin,
            device=args.device,
            save_path=args.save,
        )
