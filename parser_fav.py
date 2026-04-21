import asyncio
import io
import logging
from urllib.parse import urljoin, urlparse

import aiohttp
import cairosvg
from bs4 import BeautifulSoup
from PIL import Image

import os
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncFaviconExtractor:

    def __init__(
        self,
        img_size=(48, 48),
        concurrency=500,          # кол-во одновременных запросов
        timeout_page=40,          # таймаут на загрузку страницы
        timeout_favicon=40,        # таймаут на скачивание фавиконки
        retries=2,
    ):
        self.img_size = img_size
        self.concurrency = concurrency
        self.timeout_page = aiohttp.ClientTimeout(total=timeout_page)
        self.timeout_favicon = aiohttp.ClientTimeout(total=timeout_favicon)
        self.retries = retries
        self._semaphore: asyncio.Semaphore | None = None
        self._session: aiohttp.ClientSession | None = None


    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            ssl=False,              # пропускаем невалидные серты
            ttl_dns_cache=300,
            force_close=False,
        )
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Cookie": "CONSENT=YES+cb.20230531-04-p0.en+FX+908",
        }
        self._session = aiohttp.ClientSession(connector=connector, headers=headers)
        self._semaphore = asyncio.Semaphore(self.concurrency)
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()


    async def _get(self, url: str, timeout: aiohttp.ClientTimeout, method="GET"):
        """Выполняет GET/HEAD запрос с повторами."""
        for attempt in range(self.retries + 1):
            try:
                fn = self._session.get if method == "GET" else self._session.head
                resp = await fn(url, timeout=timeout, allow_redirects=True)
                return resp
            except Exception as exc:
                if attempt == self.retries:
                    logger.debug("Failed %s after %d retries: %s", url, self.retries, exc)
                    return None
                await asyncio.sleep(0.3 * (attempt + 1))

    async def _find_favicon_url(self, url: str):
        resp = await self._get(url, self.timeout_page)
        if resp is None:
            return None

        try:
            html = await resp.text(errors="replace")
        except Exception:
            html = ""

        soup = BeautifulSoup(html, "html.parser")

        for rel in ("icon", "shortcut icon", "apple-touch-icon"):
            tag = soup.find("link", rel=rel)
            if tag and tag.get("href"):
                return urljoin(url, tag["href"])

        parsed = urlparse(url)
        default = f"{parsed.scheme}://{parsed.netloc}/favicon.ico"
        head = await self._get(default, self.timeout_favicon, method="HEAD")
        if head and head.status == 200:
            return default

        return None

    def _to_image(self, data: bytes, content_type: str, favicon_url: str) -> Image.Image:
        """Декодирует bytes → PIL.Image и ресайзит."""
        if "svg" in content_type or favicon_url.endswith(".svg"):
            png = cairosvg.svg2png(
                bytestring=data,
                output_width=self.img_size[0],
                output_height=self.img_size[1],
            )
            img = Image.open(io.BytesIO(png))
        else:
            img = Image.open(io.BytesIO(data))

        return img.resize(self.img_size, Image.Resampling.LANCZOS)

    async def extract(self, url: str):
        """Извлекает фавиконку для одного URL."""
        async with self._semaphore:
            favicon_url = await self._find_favicon_url(url)
            if not favicon_url:
                logger.debug("No favicon found for %s", url)
                return None

            resp = await self._get(favicon_url, self.timeout_favicon)
            if resp is None or resp.status != 200:
                return None

            try:
                data = await resp.read()
                content_type = resp.headers.get("Content-Type", "")
                return self._to_image(data, content_type, favicon_url)
            except Exception as exc:
                logger.debug("Failed to decode favicon for %s: %s", url, exc)
                return None

    async def extract_many(self, urls: list[str]):
        tasks = {url: asyncio.create_task(self.extract(url)) for url in urls}
        results = {}
        for url, task in tasks.items():
            try:
                results[url] = await task
            except Exception as exc:
                logger.warning("Unhandled error for %s: %s", url, exc)
                results[url] = None
        return results

async def process_domains(
    domains: list[str],
    output_dir: str = "sus_sites",
    concurrency: int = 500,
) -> pd.DataFrame:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "domain": domains,
        "is_exist": 0,
        "rate": 0,
        "is_sus": 0,
    })

    def save_image(img, path: str):
        img.save(path)

    loop = asyncio.get_event_loop()

    async with AsyncFaviconExtractor(concurrency=concurrency) as extractor:

        async def fetch_and_save(idx: int, domain: str):
            url = f"https://{domain}"
            img = await extractor.extract(url)

            if img is None:
                print(f"✗ Не найдено: {url}")
                return idx, False

            save_path = os.path.join(output_dir, f"out_{domain}.png")
            # Сохраняем в threadpool — PIL.save() не async
            await loop.run_in_executor(None, save_image, img, save_path)
            print(f"✓ {url} → {save_path}")
            return idx, True

        tasks = [
            asyncio.create_task(fetch_and_save(i, domain))
            for i, domain in enumerate(domains)
        ]

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Favicons")

    for idx, success in results:
        if success:
            df.loc[idx, "is_exist"] = 1

    return df

def read_csv(file_path: str, has_header: bool = True) -> list[str]:
    df = pd.read_csv(file_path, header=0 if has_header else None)
    return df.iloc[:, 0].tolist()


async def main():
    domains = read_csv("suspicious_sites_2.csv", has_header=True)
    print(f"Загружено доменов: {len(domains)}")

    df = await process_domains(
        domains,
        output_dir="sus_sites",
        concurrency=500,
    )

    # Сохраняем результаты
    df.to_csv("results.csv", index=False)
    print(df.head(10))
    print(f"\nНайдено фавиконок: {df['is_exist'].sum()} / {len(df)}")

#  await main()
