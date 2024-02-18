import os
import wget
import ssl


def download_checkpoint_if_needed(filename: str, url: str = None, cache_dir: str = "ckpt_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if not os.path.isfile(path) and url is not None:
        print(f"Downloading checkpoint file: {url}")
        ssl._create_default_https_context = ssl._create_unverified_context
        wget.download(url, path)
    return path
