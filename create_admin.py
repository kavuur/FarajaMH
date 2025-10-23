# scripts/create_admin.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # allow "from models import ..."
from models import SessionLocal, User, Role, user_roles, init_db
try:
    from security import hash_password           # if argon2 is set up
except Exception:
    # fallback to werkzeug if you didn't set up security.py
    from werkzeug.security import generate_password_hash as hash_password


def ensure_role(db, name: str) -> Role:
    """Make sure a role (e.g., 'admin') exists in the DB."""
    r = db.query(Role).filter_by(name=name).first()
    if not r:
        r = Role(name=name)
        db.add(r)
        db.commit()
    return r


def main(email: str, password: str):
    """Create an admin user with given email and password."""
    init_db()
    db = SessionLocal()
    try:
        # Create user if not already in DB
        user = db.query(User).filter_by(email=email).first()
        if not user:
            user = User(
                email=email,
                password_hash=hash_password(password),
                email_verified=True,
                is_active=True
            )
            db.add(user)
            db.commit()
            print(f"✅ Created user {email}")
        else:
            print(f"ℹ️ User {email} already exists")

        # Ensure roles exist
        admin_role = ensure_role(db, "admin")
        clinician_role = ensure_role(db, "clinician")

        # Attach admin role
        already = db.execute(
            user_roles.select().where(
                (user_roles.c.user_id == user.id) & (user_roles.c.role_id == admin_role.id)
            )
        ).first()
        if not already:
            db.execute(user_roles.insert().values(user_id=user.id, role_id=admin_role.id))
            db.commit()
            print(f"✅ Granted admin role to {email}")
        else:
            print(f"ℹ️ {email} already has admin role")

        # Optionally also grant clinician role
        already_c = db.execute(
            user_roles.select().where(
                (user_roles.c.user_id == user.id) & (user_roles.c.role_id == clinician_role.id)
            )
        ).first()
        if not already_c:
            db.execute(user_roles.insert().values(user_id=user.id, role_id=clinician_role.id))
            db.commit()
            print(f"✅ Granted clinician role to {email}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/create_admin.py admin@gmail.com 'Admin123!'", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1].strip().lower(), sys.argv[2])
