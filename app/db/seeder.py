import json
import os
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document


def seed_docs_if_empty(db: Session, docs_path: str | None = None) -> list[Document]:
    count = db.query(Document).count()
    if count > 0:
        return db.query(Document).all()

    target_path = docs_path or settings.docs_seed_path
    if not os.path.exists(target_path):
        # Fallback search if path is relative
        fallback = os.path.join(os.path.dirname(__file__), "..", "..", "starter_kit", "data", "docs.json")
        if os.path.exists(fallback):
            target_path = fallback

    if not os.path.exists(target_path):
        return []

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs_data = data.get("docs", [])
    created = []
    for item in docs_data:
        doc = Document(
            id=item["doc_id"],
            title=item["title"],
            content=item["content"],
        )
        db.add(doc)
        created.append(doc)

    db.commit()
    for doc in created:
        db.refresh(doc)
    return created
