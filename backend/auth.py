import time
import hmac
import hashlib
from backend.config import SESSION_SECRET

SESSION_EXPIRY_SECONDS = 30 * 86400  # 30 days

def create_session_token(username: str) -> str:
    """Generates a cryptographically signed HMAC SHA-256 session token."""
    clean_user = (username or '').strip().lstrip('@').lower()
    if not clean_user:
        return ""
    ts = int(time.time())
    payload = f"{clean_user}:{ts}"
    sig = hmac.new(SESSION_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"

def verify_session_token(token: str) -> tuple[bool, str]:
    """Validates an incoming HMAC session token and returns (is_valid, username)."""
    if not token or not isinstance(token, str):
        return False, ""
    parts = token.strip().split(':')
    if len(parts) != 3:
        return False, ""
    username, ts_str, sig = parts
    try:
        ts = int(ts_str)
    except ValueError:
        return False, ""
    
    # Check expiration (30 days validity, 5 mins future clock skew allowance)
    if time.time() - ts > SESSION_EXPIRY_SECONDS or ts > time.time() + 300:
        return False, ""
    
    payload = f"{username}:{ts}"
    expected_sig = hmac.new(SESSION_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return False, ""
    return True, username
