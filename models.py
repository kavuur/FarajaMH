# models.py
import os
import uuid as _uuid
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ForeignKey, Integer,
    Boolean, Table, UniqueConstraint, JSON, Index, text
)
from sqlalchemy.orm import (
    sessionmaker, declarative_base, relationship, scoped_session
)

DB_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    UniqueConstraint("user_id", "role_id", name="uq_user_role"),
)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    owner_user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    messages = relationship("Message", back_populates="conversation",
                            cascade="all, delete-orphan",
                            order_by="Message.created_at.asc()")
    screenings = relationship("ScreeningEvent", back_populates="conversation",
                              cascade="all, delete-orphan",
                              order_by="ScreeningEvent.created_at.desc()")
    def __repr__(self) -> str:
        return f"<Conversation id={self.id} owner_user_id={self.owner_user_id}>"

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True, nullable=False)
    role = Column(String, index=True, nullable=False)
    type = Column(String, default="message", nullable=False)
    message = Column(Text, nullable=True)
    timestamp = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    conversation = relationship("Conversation", back_populates="messages")

    # NEW (optional) FAISS fields
    faiss_question_id = Column(String(128), index=True, nullable=True)
    faiss_category = Column(String(32), index=True, nullable=True)
    faiss_is_answer = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_messages_conv_faisscat", "conversation_id", "faiss_category"),
    )

    def __repr__(self) -> str:
        return (f"<Message id={self.id} conv={self.conversation_id} role={self.role} "
                f"type={self.type} faiss_q={self.faiss_question_id} "
                f"faiss_cat={self.faiss_category} is_ans={self.faiss_is_answer}>")

class ScreeningEvent(Base):
    __tablename__ = "screening_events"
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True, nullable=True)
    overall_flag = Column(String, index=True)
    results_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    conversation = relationship("Conversation", back_populates="screenings")
    def __repr__(self) -> str:
        return f"<ScreeningEvent id={self.id} conv={self.conversation_id} flag={self.overall_flag}>"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    roles = relationship("Role", secondary=user_roles, back_populates="users", lazy="joined")
    @property
    def is_authenticated(self): return True
    @property
    def is_anonymous(self): return False
    def get_id(self): return str(self.id)
    def has_role(self, name: str) -> bool: return any(r.name == name for r in self.roles)
    def __repr__(self) -> str: return f"<User id={self.id} email={self.email}>"

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True)
    name = Column(String(32), unique=True, nullable=False)
    users = relationship("User", secondary=user_roles, back_populates="roles")
    def __repr__(self) -> str: return f"<Role id={self.id} name={self.name}>"

class ConversationOwner(Base):
    __tablename__ = "conversation_owners"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True, nullable=False, unique=True)
    owner_user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    def __repr__(self) -> str:
        return f"<ConversationOwner conv={self.conversation_id} owner={self.owner_user_id}>"

def _seed_roles():
    db = SessionLocal()
    try:
        existing = {r.name for r in db.query(Role).all()}
        for name in ("clinician", "admin"):
            if name not in existing:
                db.add(Role(name=name))
        db.commit()
    finally:
        db.close()

def _auto_migrate_messages_sqlite():
    """Add FAISS columns on SQLite if they don't exist (idempotent, safe)."""
    if not DB_URL.startswith("sqlite"):
        return
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(messages);")}
        alters = []
        if "faiss_question_id" not in cols:
            alters.append("ALTER TABLE messages ADD COLUMN faiss_question_id VARCHAR(128)")
        if "faiss_category" not in cols:
            alters.append("ALTER TABLE messages ADD COLUMN faiss_category VARCHAR(32)")
        if "faiss_is_answer" not in cols:
            alters.append("ALTER TABLE messages ADD COLUMN faiss_is_answer BOOLEAN DEFAULT 0 NOT NULL")
        for sql in alters:
            conn.exec_driver_sql(sql)

def init_db():
    """Create tables (first run) and ensure required columns exist."""
    Base.metadata.create_all(bind=engine)
    _auto_migrate_messages_sqlite()
    _seed_roles()

def create_conversation(owner_user_id: int | None = None) -> str:
    db = SessionLocal()
    try:
        cid = str(_uuid.uuid4())
        db.add(Conversation(id=cid, owner_user_id=owner_user_id))
        db.commit()
        return cid
    finally:
        db.close()

def log_message(
    conversation_id: str,
    role: str,
    message: str | None,
    timestamp: str | None,
    type_: str = "message",
    *,
    faiss_question_id: str | None = None,
    faiss_category: str | None = None,
    faiss_is_answer: bool = False,
) -> str:
    db = SessionLocal()
    try:
        mid = str(_uuid.uuid4())
        db.add(Message(
            id=mid,
            conversation_id=conversation_id,
            role=role,
            type=type_,
            message=message,
            timestamp=timestamp,
            faiss_question_id=faiss_question_id,
            faiss_category=faiss_category,
            faiss_is_answer=faiss_is_answer,
        ))
        db.commit()
        return mid
    finally:
        db.close()

def list_conversations():
    db = SessionLocal()
    try:
        return db.query(Conversation).order_by(Conversation.created_at.desc()).all()
    finally:
        db.close()

def get_conversation_messages(conversation_id: str):
    db = SessionLocal()
    try:
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
    finally:
        db.close()
