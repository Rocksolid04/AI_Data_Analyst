
import os

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

load_dotenv()


def _get_fernet() -> Fernet:
    key = os.getenv("DATABASE_ENCRYPTION_KEY")

    if not key:
        raise RuntimeError(
            "DATABASE_ENCRYPTION_KEY is not configured."
        )

    try:
        return Fernet(key.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "DATABASE_ENCRYPTION_KEY is invalid."
        ) from exc


def encrypt_password(password: str) -> str:
    """Encrypt a database password before storing it."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")

    encrypted = _get_fernet().encrypt(password.encode("utf-8"))
    return encrypted.decode("utf-8")


def decrypt_password(encrypted_password: str) -> str:
    """Decrypt a stored database password when connecting."""
    if not isinstance(encrypted_password, str) or not encrypted_password:
        raise ValueError("Encrypted password must be a non-empty string.")

    try:
        decrypted = _get_fernet().decrypt(
            encrypted_password.encode("utf-8")
        )
        return decrypted.decode("utf-8")
    except (InvalidToken, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "Could not decrypt the stored database password. "
            "Check the encryption key configuration."
        ) from exc
