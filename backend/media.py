import re
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class MediaGateway:
    """Thread-safe media proxy & disk/memory caching gateway for movie posters."""
    def __init__(self, max_cache_size=800, timeout=6.0):
        self._cache = {}
        self._lock = threading.Lock()
        self._max_cache_size = max_cache_size
        self._timeout = timeout
        self._session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=20)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def fetch_poster(self, filename: str) -> tuple[bytes | None, str | None]:
        clean_file = (filename or '').lstrip('/')
        if not clean_file or not re.match(r'^[a-zA-Z0-9_\-\.]+\.(?:jpg|jpeg|png|webp)$', clean_file):
            return None, None

        with self._lock:
            if clean_file in self._cache:
                return self._cache[clean_file]

        tmdb_url = f"https://image.tmdb.org/t/p/w500/{clean_file}"
        try:
            resp = self._session.get(tmdb_url, timeout=self._timeout)
            if resp.status_code == 200 and resp.content:
                ctype = resp.headers.get('Content-Type', 'image/jpeg')
                with self._lock:
                    if len(self._cache) >= self._max_cache_size:
                        self._cache.clear()
                    self._cache[clean_file] = (resp.content, ctype)
                return resp.content, ctype
        except Exception:
            pass

        return None, None

media_gateway = MediaGateway()
