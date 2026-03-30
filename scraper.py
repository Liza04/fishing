"""
scraper.py — массовый асинхронный скриншотер сайтов по списку доменов.

Описание
=================
Скрипт делает скриншоты сайтов по списку доменов, используя Playwright (Chromium).
Для скорости используется двухуровневый параллелизм:
  - несколько процессов (multiprocessing), каждый держит свой браузер
  - внутри каждого процесса — пул вкладок (asyncio), работающих одновременно

Итого вкладок = NUM_PROCESSES × CONCURRENCY (по умолчанию 10 × 5 = 50).

Результаты пишутся в папку с PNG-файлами и сопровождаются отчётом report.jsonl,
где каждая строка — JSON с полями: domain, url, status, screenshot_path, error_msg.

Уже готовые скриншоты не перезаписываются — можно останавливать и продолжать.

Примеры запуска
---------------
# запуск — список доменов из файла, скриншоты в папку screens, количество параллельных процессов, воркеров, размер чанка
    python scraper.py --domains domains.txt --output screens --processes 8 --workers 6 --chunk 100

Формат файла доменов (domains.txt)
-----------------------------------
Один домен на строку.
Поддерживаются форматы:
    example.com
    https://example.com
    http://example.com

Формат отчёта (report.jsonl)
-----------------------------
Каждая строка — отдельный JSON-объект:
    {"domain": "example.com", "url": "https://example.com",
     "status": "ok", "screenshot_path": "screen_out/example.com.png", "error_msg": null}

Статусы: ok | timeout | error | skip
"""

import asyncio
import argparse
import json
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from multiprocessing import Process, Queue
from queue import Empty

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import sys

# ── Константы ─────────────────────────────────────────────────────────────────

VIEW = {'width': 1280, 'height': 1080}  # размер окна браузера
SCREEN_FULL = False  # полная страница или только viewport
TIMEOUT_MS = 7_000  # таймаут загрузки страницы (мс)
WAIT_LOAD_MS = 500  # ожидание после загрузки (мс)
MAX_RETRIES = 1  # попыток на домен
CONCURRENCY = 5  # вкладок на процесс
NUM_PROCESSES = 10  # количество процессов
CHUNK_SIZE = 100  # доменов в одном задании


# ── Датакласс результата ──────────────────────────────────────────────────────

@dataclass
class DomainResult:
    """Результат обработки одного домена."""
    domain: str  # исходный домен как в файле
    url: str  # URL с которым работали
    status: str  # ok | timeout | error | skip
    screenshot_path: Optional[str] = None  # путь к PNG, если успешно
    error_msg: Optional[str] = None  # текст ошибки, если была


# ── Вспомогательные функции ───────────────────────────────────────────────────

def make_url(domain: str) -> str:
    """
    Нормализует домен в полный HTTPS-URL.

    Если домен уже начинается с https:// — возвращает как есть.
    Если начинается с http:// — убирает схему и подставляет https://.
    Иначе просто добавляет https://.

    Args:
        domain: строка вида 'example.com', 'http://example.com' и т.п.

    Returns:
        URL вида 'https://example.com'

    Examples:
        >>> make_url('example.com')
        'https://example.com'
        >>> make_url('http://example.com')
        'https://example.com'
        >>> make_url('https://example.com')
        'https://example.com'
    """
    if domain.startswith('http://'):
        domain = domain[7:]
    elif domain.startswith('https://'):
        return domain
    return f'https://{domain}'


def domain_to_filename(domain: str) -> str:
    """
    Превращает домен в безопасное имя файла (без слешей и схемы).

    Убирает https:// и http://, заменяет / на _, обрезает точки по краям.
    Регистр не меняется — сохраняется как в исходном файле.

    Args:
        domain: домен или URL

    Returns:
        Строка, пригодная как имя файла (без расширения)

    Examples:
        >>> domain_to_filename('example.com')
        'example.com'
        >>> domain_to_filename('https://example.com/path')
        'example.com_path'
    """
    return (
        domain
        .replace('https://', '')
        .replace('http://', '')
        .replace('/', '_')
        .strip('.')
    )


# ── Асинхронный скриншот одного домена ───────────────────────────────────────

async def screenshot_by_domain(
        context_queue: asyncio.Queue,
        domain: str,
        output_dir: Path,
) -> DomainResult:
    """
    Делает скриншот одного домена, используя контекст браузера из пула.

    Алгоритм:
        1. Берёт готовый контекст из context_queue (блокирует если все заняты).
        2. Открывает новую вкладку, блокирует медиафайлы для скорости.
        3. Навигируется на URL, ждёт загрузки.
        4. Делает скриншот и сохраняет в output_dir.
        5. В любом случае закрывает вкладку и возвращает контекст в пул.

    Если скриншот уже существует — сразу возвращает ok без запроса к сайту.
    При ошибке повторяет MAX_RETRIES раз, затем возвращает статус timeout/error.

    Args:
        context_queue: asyncio.Queue с готовыми контекстами Playwright.
        domain:        Домен для скриншота (любой формат, нормализуется через make_url).
        output_dir:    Папка для сохранения PNG-файлов.

    Returns:
        DomainResult со статусом ok / timeout / error.
    """
    url = make_url(domain)
    if not url:
        return DomainResult(domain=domain, url='', status='error', error_msg='empty domain')

    screenshot_path = output_dir / f'{domain_to_filename(domain)}.png'

    # Пропускаем если скриншот уже есть (можно перезапускать)
    if screenshot_path.exists():
        return DomainResult(domain=domain, url=url, status='ok',
                            screenshot_path=str(screenshot_path))

    context = await context_queue.get()
    page = None

    for attempt in range(MAX_RETRIES):
        start_time = time.monotonic()
        try:
            page = await context.new_page()

            # Блокируем тяжёлые медиафайлы — ускоряет загрузку
            await page.route(
                '**/*.{mp4,webm,ogg,mp3,wav,zip}',
                lambda route: route.abort()
            )
            await page.goto(url, timeout=TIMEOUT_MS, wait_until='load')
            await page.wait_for_timeout(WAIT_LOAD_MS)
            await page.screenshot(path=str(screenshot_path), full_page=SCREEN_FULL, type='png')

            duration = time.monotonic() - start_time
            return DomainResult(domain=domain, url=url, status='ok',
                                screenshot_path=str(screenshot_path))

        except PlaywrightTimeout:
            if attempt == MAX_RETRIES - 1:
                return DomainResult(domain=domain, url=url,
                                    status='timeout', error_msg='timeout')

        except Exception as e:
            error = str(e)[:100]
            if attempt == MAX_RETRIES - 1:
                return DomainResult(domain=domain, url=url,
                                    error_msg=error, status='error')

        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            await context_queue.put(context)


# ── Точка входа воркер-процесса ───────────────────────────────────────────────

def worker_process(
        pid: int,
        task_queue: Queue,
        result_queue: Queue,
        output_dir: str,
        workers: int,
):
    """
    Точка входа для дочернего процесса — запускает asyncio event loop.

    Каждый процесс держит свой экземпляр Chromium с пулом из `workers` контекстов.
    Процессы общаются с главным через multiprocessing.Queue:
        task_queue  — получают чанки доменов
        result_queue — отправляют результаты (dict от DomainResult)

    Args:
        pid:          Номер процесса (для логов).
        task_queue:   Очередь входящих заданий (чанков доменов).
        result_queue: Очередь исходящих результатов.
        output_dir:   Папка для скриншотов (строка, т.к. Path не сериализуется между процессами).
        workers:      Количество параллельных вкладок (контекстов) в этом процессе.
    """
    asyncio.run(_worker_loop(pid, task_queue, result_queue, Path(output_dir), workers))


# ── Asyncio цикл внутри воркера ───────────────────────────────────────────────

async def _worker_loop(
        pid: int,
        task_queue: Queue,
        result_queue: Queue,
        output_dir: Path,
        workers: int,
):
    """
    Основной асинхронный цикл воркера: получает чанки, обрабатывает пачками.

    Запускает браузер, создаёт пул из `workers` независимых контекстов
    Контексты хранятся в asyncio.Queue, откуда screenshot_by_domain их берёт и возвращает.

    Чтение из multiprocessing.Queue делается через run_in_executor чтобы не блокировать
    event loop — это позволяет параллельно обрабатывать скриншоты и ждать новых задач.

    Args:
        pid:          Номер процесса (для логов).
        task_queue:   Очередь заданий от главного процесса.
        result_queue: Очередь результатов в главный процесс.
        output_dir:   Папка для скриншотов.
        workers:      Размер пула контекстов (= параллельных вкладок).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
        )

        # Создаём пул контекстов один раз на весь процесс
        contexts = [
            await browser.new_context(
                viewport=VIEW,
                ignore_https_errors=True,
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
            )
            for _ in range(workers)
        ]

        context_queue: asyncio.Queue = asyncio.Queue()
        for ctx in contexts:
            await context_queue.put(ctx)

        loop = asyncio.get_event_loop()

        while True:
            try:
                # Блокирующий get из multiprocessing.Queue — выносим в executor
                chunk: list[str] = await loop.run_in_executor(
                    None, _queue_get, task_queue
                )
            except _Stop:
                break

            if chunk is None:
                break

            # Запускаем все домены чанка параллельно (ограничено размером пула контекстов)
            tasks = [
                screenshot_by_domain(context_queue, domain, output_dir)
                for domain in chunk
            ]
            chunk_res = await asyncio.gather(*tasks)

            for res in chunk_res:
                result_queue.put(asdict(res))

        for ctx in contexts:
            try:
                await ctx.close()
            except Exception:
                pass

        await browser.close()

    logging.info(f'[P{pid}] done')


# ── Вспомогательная функция чтения из multiprocessing.Queue ──────────────────

class _Stop(Exception):
    """Стоп-сигнал (получен None от feeder-потока)"""


def _queue_get(q: Queue):
    """
    Блокирующее чтение из multiprocessing.Queue с поддержкой стоп-сигнала.

    Крутится в цикле с таймаутом 5 сек, чтобы не зависнуть навсегда если
    главный процесс упал. None в очереди интерпретируется как стоп-сигнал
    и бросает _Stop.

    Args:
        q: multiprocessing.Queue для чтения.

    Returns:
        Следующий элемент из очереди (чанк доменов).

    Raises:
        _Stop: если получен None (стоп-сигнал от feeder).
    """
    while True:
        try:
            item = q.get(timeout=5)
            if item is None:
                raise _Stop
            return item
        except Empty:
            continue


# ── Главная функция оркестрации ───────────────────────────────────────────────

def run_multiprocess(
        domains: list[str],
        output_dir: Path,
        num_processes: int,
        workers_per_proc: int,
        chunk_size: int,
):
    """
    Запускает массовое создание скриншотов через пул процессов.

    Архитектура:
        Главный процесс
            ├── feeder-поток: раскладывает домены чанками в task_queue
            ├── процесс 0: Chromium + asyncio (workers_per_proc вкладок)
            ├── процесс 1: Chromium + asyncio (workers_per_proc вкладок)
            └── ...
        Все процессы пишут результаты в result_queue.
        Главный процесс читает result_queue и пишет report.jsonl.

    Идемпотентность: домены с уже существующим скриншотом пропускаются внутри
    screenshot_by_domain — перезапуск после сбоя продолжит с того места.

    Args:
        domains:          Список доменов для обработки.
        output_dir:       Папка для скриншотов и report.jsonl.
        num_processes:    Количество параллельных процессов (браузеров).
        workers_per_proc: Количество вкладок на процесс.
        chunk_size:       Сколько доменов передавать процессу за раз.

    Returns:
        list[dict]: Список результатов (каждый — asdict(DomainResult)).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_queue: Queue = Queue(maxsize=num_processes * 4)
    result_queue: Queue = Queue()

    processes = []
    for pid in range(num_processes):
        p = Process(
            target=worker_process,
            args=(pid, task_queue, result_queue, str(output_dir), workers_per_proc),
            daemon=True,
        )
        p.start()
        processes.append(p)

    total = len(domains)
    logging.info(f'Доменов: {total} | Процессов: {num_processes} | Вкладок на процесс: {workers_per_proc}')

    def feeder():
        """Подаёт чанки в task_queue, потом рассылает стоп-сигналы (None) всем процессам."""
        for i in range(0, total, chunk_size):
            task_queue.put(domains[i:i + chunk_size])
        for _ in processes:
            task_queue.put(None)

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    results = []
    report_path = output_dir / 'report.jsonl'
    processed = 0

    with open(report_path, 'w', encoding='utf-8') as report_f:
        while processed < total:
            try:
                item = result_queue.get(timeout=30)
                results.append(item)
                report_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                processed += 1

                if processed % 500 == 0:
                    ok = sum(1 for r in results if r['status'] == 'ok')
                    logging.info(f'[PROGRESS] {processed}/{total}  OK={ok}')

            except Empty:
                alive = [p for p in processes if p.is_alive()]
                if not alive:
                    logging.warning('Все воркеры завершились раньше времени')
                    break

    feeder_thread.join(timeout=5)
    for p in processes:
        p.join()

    ok = sum(1 for r in results if r['status'] == 'ok')
    timeout = sum(1 for r in results if r['status'] == 'timeout')
    error = sum(1 for r in results if r['status'] == 'error')
    skipped = sum(1 for r in results if r['status'] == 'skip')

    logging.info('=' * 50)
    logging.info(f'Готово! Всего: {total}')
    logging.info(f'  ✓ OK:      {ok}')
    logging.info(f'  ✗ Timeout: {timeout}')
    logging.info(f'  ✗ Error:   {error}')
    logging.info(f'  - Skipped: {skipped}')
    logging.info(f'  Отчёт:     {report_path}')

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """
    Точка входа CLI. Парсит аргументы, читает файл доменов, запускает run_multiprocess.

    Файл доменов — текстовый, один домен на строку.
    Пустые строки и строки начинающиеся с # пропускаются.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description='Screenshot scraper (multiprocess + asyncio)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python scraper.py --domains domains.txt --output screens --processes 20 --workers 10 --chunk 500
        """
    )
    parser.add_argument('--domains', required=True, help='Файл с доменами (по одному на строку)')
    parser.add_argument('--output', default='screen_out', help='Папка для скриншотов (default: screen_out)')
    parser.add_argument('--processes', type=int, default=NUM_PROCESSES,
                        help=f'Процессов (default: {NUM_PROCESSES})')
    parser.add_argument('--workers', type=int, default=CONCURRENCY,
                        help=f'Вкладок на процесс (default: {CONCURRENCY})')
    parser.add_argument('--chunk', type=int, default=CHUNK_SIZE,
                        help=f'Размер чанка (default: {CHUNK_SIZE})')
    args = parser.parse_args()

    domains_file = Path(args.domains)
    if not domains_file.exists():
        print(f'Файл не найден: {domains_file}')
        sys.exit(1)

    with open(domains_file, encoding='utf-8') as f:
        domains = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]

    logging.info(f'Загружено доменов: {len(domains)}')

    run_multiprocess(
        domains=domains,
        output_dir=Path(args.output),
        num_processes=args.processes,
        workers_per_proc=args.workers,
        chunk_size=args.chunk,
    )


if __name__ == '__main__':
    main()
