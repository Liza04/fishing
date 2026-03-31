"""
train_yolo.py — обучение YOLOv13n детектора логотипа(ов).
Поддерживает мультиклассовый датасет (несколько брендов).

Требования:
    pip install git+https://github.com/iMoonLab/yolov13.git pillow albumentations
    Веса yolov13n.pt скачать с https://github.com/iMoonLab/yolov13/releases

Использование:
    # Базовый запуск
    python train_yolo.py

    # Указать датасет и имя эксперимента
    python train_yolo.py --data dataset_multi/data.yaml --name moex_v1

    # Продолжить с чекпоинта
    python train_yolo.py --resume runs/logo/moex_v1/weights/last.pt

"""

import os
import argparse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Обучение YOLOv13n детектора логотипов (M4 / CPU)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--data", default="dataset_multi/data.yaml",
                    help="Путь к data.yaml датасета")
parser.add_argument("--weights", default="yolov13n.pt",
                    help="Стартовые веса (pretrained или чекпоинт)")
parser.add_argument("--name", default="yolov13n_logo",
                    help="Имя эксперимента (папка в runs/logo/)")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640,
                    help="Размер входа)")
parser.add_argument("--batch", type=int, default=8,
                    help="Размер батча")
parser.add_argument("--resume", default=None,
                    help="Продолжить с чекпоинта: путь к last.pt")
args = parser.parse_args()

# ── Устройство ─────────────────────────────────────────────────────────────────
DEVICE = 'cpu'
'''
mps_ok = torch.backends.mps.is_available()
print(f"MPS  доступен: {mps_ok}")
DEVICE = "mps" if mps_ok else "cpu"
print(f"Используем устройство: {DEVICE}")
'''

# ── Модель ─────────────────────────────────────────────────────────────────────

if args.resume:
    # Продолжаем прерванное обучение
    print(f"Продолжаем с чекпоинта: {args.resume}")
    model = YOLO(args.resume)
else:
    # Fine-tuning с pretrained весами COCO — сходится намного быстрее нуля
    print(f"Загружаем веса: {args.weights}")
    model = YOLO(args.weights)

# ── Обучение ───────────────────────────────────────────────────────────────────

results = model.train(
    # Данные
    data=args.data,

    # Базовые параметры
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=DEVICE,

    workers=0,

    amp=False,

    # Логирование
    project="runs/logo",
    name=args.name,

    # Early stopping — остановимся если val-метрика не растёт N эпох
    patience=20,

    # Оптимизатор
    optimizer="AdamW",
    lr0=0.001,  # для мультикласса чуть ниже чем для single_cls
    lrf=0.01,  # финальный lr = lr0 * lrf
    weight_decay=0.0005,
    momentum=0.937,

    # Аугментации — важны для синтетического датасета
    mosaic=1.0,  # мозаика из 4 изображений — сильно помогает
    mixup=0.0,  # mixup для логотипов скорее мешает
    copy_paste=0.1,  # вырезаем логотип и вставляем в другое место
    degrees=5.0,  # поворот ±5°
    translate=0.1,  # сдвиг ±10%
    scale=0.3,  # масштаб ±30%
    fliplr=0.3,  # горизонтальное зеркало
    flipud=0.0,  # вертикальное 
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.2,

    # single_cls=True УБРАНО 

    # Чекпоинты каждые 10 эпох (на случай прерывания)
    save_period=10,

    # Порог для val-детекций (не влияет на веса, только на метрики)
    conf=0.25,
    iou=0.7,

    # Продолжить с чекпоинта если указан --resume
    resume=bool(args.resume),
)



print("\n   Обучение завершено!")
print(f"   Лучшие веса:    {results.save_dir}/weights/best_m.pt")
print(f"   Последние веса: {results.save_dir}/weights/last_m.pt")
print(f"\n   Запуск инференса:")
print(f"   python detect.py --model {results.save_dir}/weights/best.pt --source screenshots/")
