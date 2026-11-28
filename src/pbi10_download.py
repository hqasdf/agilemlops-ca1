"""Download the three inputs as one staged batch, with bounded response sizes."""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def download_batch(urls, destination, max_bytes=250_000_000, timeout=30):
    """Download three named CSV inputs; publish only after all downloads succeed."""
    names = ("customers", "orders", "labels")
    if set(urls) != set(names):
        raise ValueError("Provide customers, orders and labels URLs")
    destination = Path(destination)
    if not destination.is_dir():
        raise ValueError("Destination directory does not exist")
    if max_bytes < 1 or timeout <= 0:
        raise ValueError("Size and timeout must be positive")
    for name in names:
        parsed = urlparse(urls[name])
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"{name}: expected an HTTP(S) URL")
        if (destination / f"{name}.csv").exists():
            raise ValueError(f"{name}: destination already exists")
    with TemporaryDirectory(dir=destination, prefix=".batch-") as stage:
        staged = {}
        for name in names:
            path = Path(stage) / f"{name}.csv"
            try:
                req = Request(urls[name], headers={"User-Agent": "AMLOPS-PBI10/1.0"})
                with urlopen(req, timeout=timeout) as response, path.open("wb") as output:
                    if urlparse(response.geturl()).scheme not in ("http", "https"):
                        raise ValueError(f"{name}: invalid redirect")
                    if response.headers.get_content_type() in ("text/html", "application/xhtml+xml"):
                        raise ValueError(f"{name}: server returned HTML, not CSV")
                    size = 0
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"{name}: download exceeds size limit")
                        output.write(chunk)
                    if size == 0:
                        raise ValueError(f"{name}: download is empty")
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                raise ValueError(f"{name}: download failed ({type(error).__name__})") from error
            staged[name] = path
        result = {}
        for name in names:
            target = destination / f"{name}.csv"
            staged[name].rename(target)
            result[name] = target
        return result

