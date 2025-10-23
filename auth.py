from flask import Blueprint, jsonify, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import SessionLocal, User, Role, user_roles
from security import hash_password, verify_password

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        return db.get(User, int(user_id))
    finally:
        db.close()

def grant_role(db, user: User, role_name: str):
    role = db.query(Role).filter_by(name=role_name).first()
    if role and not any(r.id == role.id for r in user.roles):
        db.execute(user_roles.insert().values(user_id=user.id, role_id=role.id))
        db.commit()

@auth_bp.post("/signup")  # switch to invite-only later if you want
def signup():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password required"}), 400
    db = SessionLocal()
    try:
        if db.query(User).filter_by(email=email).first():
            return jsonify({"ok": False, "error": "Email already registered"}), 409
        u = User(email=email, password_hash=hash_password(password), email_verified=False)
        db.add(u); db.commit()
        grant_role(db, u, "clinician")
        return jsonify({"ok": True})
    finally:
        db.close()

@auth_bp.post("/login")
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    db = SessionLocal()
    try:
        u = db.query(User).filter_by(email=email).first()
        if not u or not verify_password(u.password_hash, password) or not u.is_active:
            return jsonify({"ok": False, "error": "Invalid credentials"}), 401
        login_user(u, remember=bool(data.get("remember")))
        return jsonify({"ok": True, "user": {"email": u.email, "roles": [r.name for r in u.roles]}})
    finally:
        db.close()

@auth_bp.post("/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"ok": True})

@auth_bp.get("/me")
def me():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "user": {"email": current_user.email, "roles": [r.name for r in current_user.roles]}})
