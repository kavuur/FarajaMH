from argon2 import PasswordHasher
from argon2.low_level import Type
ph = PasswordHasher(time_cost=3, memory_cost=64*1024, parallelism=2, hash_len=32, type=Type.ID)

def hash_password(pw: str) -> str:
    return ph.hash(pw)

def verify_password(hash_: str, pw: str) -> bool:
    try:
        return ph.verify(hash_, pw)
    except Exception:
        return False
