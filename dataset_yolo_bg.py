import argparse
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

LOGO_SCALE_MIN = 0.04
LOGO_SCALE_MAX = 0.20

# Зоны размещения логотипа
PLACEMENT_ZONES = [
    # (x_min, x_max, y_min, y_max, weight)
    (0.01, 0.35, 0.01, 0.12, 0.50),  # верх-лево (самое частое)
    (0.35, 0.65, 0.01, 0.12, 0.15),  # верх-центр
    (0.65, 0.99, 0.01, 0.12, 0.10),  # верх-право
    (0.01, 0.99, 0.12, 0.25, 0.15),  # верхняя треть
    (0.01, 0.99, 0.25, 0.75, 0.07),  # середина (редко)
    (0.01, 0.99, 0.75, 0.99, 0.03),  # низ (очень редко)
]
ZONE_WEIGHTS = [z[4] for z in PLACEMENT_ZONES]

random.seed(1080)


def pad_to_square(img: Image.Image) -> Image.Image:
    """Паддинг до квадрата без деформации"""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    new_img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))
    return new_img


def load_logo(logo_path: str) -> Image.Image:
    """Загружаем логотип, сохраняем прозрачность (т.k. нам важно накладывать на бэкграунд странички (любой)"""
    img = Image.open(logo_path)
    # Конвертируем в RGBA чтобы поддерживать прозрачность
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def aug_logo(logo: Image.Image) -> Image.Image:
    aug = logo.copy()

    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        aug = aug.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    # Сохраняем альфа-канал ДО аугментации цвета
    r, g, b, alpha = aug.split()
    rgb = Image.merge("RGB", (r, g, b))

    # ColorJitter применяем только к RGB
    augs = transforms.Compose([
        transforms.RandomApply(
            [transforms.ColorJitter(hue=0.05, saturation=0.2, brightness=0.1)],
            p=0.6
        ),
    ])
    rgb = augs(rgb)

    # Возвращаем альфа-канал обратно
    r2, g2, b2 = rgb.split()
    aug = Image.merge("RGBA", (r2, g2, b2, alpha))

    if random.random() < 0.2:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    if random.random() < 0.1:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

    return aug


def place_logo_on_background(logo: Image.Image, background: Image.Image) -> tuple[
    Image.Image, tuple[float, float, float, float]]:
    bg = background.convert("RGBA")
    bw, bh = bg.size

    scale = random.uniform(LOGO_SCALE_MIN, LOGO_SCALE_MAX)
    logo_w = int(bw * scale)
    # pad до квадрата, потом ресайзим
    logo_sq = pad_to_square(logo)
    logo_resized = logo_sq.resize((logo_w, logo_w), Image.LANCZOS)
    logo_aug = aug_logo(logo_resized)
    lw, lh = logo_aug.size

    zone = random.choices(PLACEMENT_ZONES, weights=ZONE_WEIGHTS, k=1)[0]
    x_min_r, x_max_r, y_min_r, y_max_r, _ = zone

    x_min = int(bw * x_min_r)
    x_max = max(x_min + lw, int(bw * x_max_r) - lw)
    y_min = int(bh * y_min_r)
    y_max = max(y_min + lh, int(bh * y_max_r) - lh)

    paste_x = random.randint(x_min, max(x_min, x_max))
    paste_y = random.randint(y_min, max(y_min, y_max))

    paste_x = max(0, min(paste_x, bw - lw))  # если вышел за пределы
    paste_y = max(0, min(paste_y, bh - lh))

    result = bg.copy()
    if logo_aug.mode == "RGBA":
        result.paste(logo_aug, (paste_x, paste_y), mask=logo_aug.split()[3])
    else:
        result.paste(logo_aug.convert("RGBA"), (paste_x, paste_y))

    result = result.convert("RGB")

    # YOLO
    cx = (paste_x + lw / 2) / bw
    cy = (paste_y + lh / 2) / bh
    nw = lw / bw
    nh = lh / bh

    return result, (cx, cy, nw, nh)


def make_logo_dataset(
        logo_path: str,
        bg_path: str,
        output: str,
        neg_pr: float = 0.1,
        val_per: float = 0.15,
        test_per: float = 0.10,
):
    """
    Генерирует датасет для обучения YOLO-детектора логотипа.

    Параметры:
        logo_path  — путь к PNG логотипу с прозрачностью
        bg_path    — папка со скриншотами (фоны)
        output     — куда сохранять датасет
        neg_pr     — доля негативных примеров (без логотипа)
        val_per    — доля валидационной выборки
        test_per   — доля тестовой выборки
    """
    logo = load_logo(logo_path)
    print(f"Логотип загружен: {logo_path} ({logo.size[0]}×{logo.size[1]}, {logo.mode})")

    # Собираем все фоны из папки
    bg_path = Path(bg_path)
    backgrounds = [
        p for p in bg_path.rglob("*")
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]
    print(f"Фонов найдено: {len(backgrounds)}")
    random.shuffle(backgrounds)

    backgrounds = backgrounds[:300]  # CPU!

    # Создаём структуру папок датасета
    output = Path(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Разбиваем фоны на train / val / test
    val_count = max(1, int(len(backgrounds) * val_per))
    test_count = max(1, int(len(backgrounds) * test_per))

    splits = {
        "val": backgrounds[:val_count],
        "test": backgrounds[val_count:val_count + test_count],
        "train": backgrounds[val_count + test_count:],
    }

    for split, bgs in splits.items():
        pos_count = 0
        neg_count = 0

        for idx, bg_file in enumerate(bgs):
            try:
                background = Image.open(bg_file).convert("RGB")
            except Exception as e:
                # print(f"Пропускаем {bg_file}: {e}")
                continue

            img_name = f"{split}_{idx:05d}.jpg"
            img_out = output / "images" / split / img_name
            lbl_out = output / "labels" / split / img_name.replace(".jpg", ".txt")

            is_negative = random.random() < neg_pr

            if is_negative:
                # Негативный пример — фон без логотипа, пустой label-файл
                background.save(img_out, quality=90)
                lbl_out.write_text("")
                neg_count += 1
            else:
                # Позитивный пример — накладываем логотип, пишем YOLO bbox
                result, (cx, cy, nw, nh) = place_logo_on_background(logo, background)
                result.save(img_out, quality=90)
                # Формат YOLO: class_id cx cy w h 
                lbl_out.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                pos_count += 1

        print(f"  [{split:5s}] всего: {len(bgs):4d}  |  позитив: {pos_count:4d}  |  негатив: {neg_count:4d}")

    # Сохраняем конфиг датасета для YOLO
    yaml_content = f"""# Автоматически сгенерировано dataset_gen.py
path: {output.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['logo']
"""
    (output / "data.yaml").write_text(yaml_content)
    print(f"\nДатасет сохранён в: {output}")
    print(f"   Конфиг:             {output / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор датасета для YOLO-детектора логотипа")
    parser.add_argument("--logo", required=True, help="Путь к логотипу (PNG с прозрачностью)")
    parser.add_argument("--backgrounds", required=True, help="Папка со скриншотами-фонами")
    parser.add_argument("--output", default="dataset", help="Куда сохранять датасет")
    parser.add_argument("--neg_pr", type=float, default=0.30, help="Доля негативных примеров (0..1)")
    parser.add_argument("--val_per", type=float, default=0.15, help="Доля валидации (0..1)")
    parser.add_argument("--test_per", type=float, default=0.10, help="Доля теста (0..1)")
    args = parser.parse_args()

    make_logo_dataset(
        logo_path=args.logo,
        bg_path=args.backgrounds,
        output=args.output,
        neg_pr=args.neg_pr,
        val_per=args.val_per,
        test_per=args.test_per,
    )

# gen_dataset_bg.py --logo logo.png --backgrounds screens.new --output dataset_bg
