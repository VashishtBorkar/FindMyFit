"""Print a concise summary of the configured catalog database."""

from sqlalchemy import func

from findmyfit.config import Settings
from findmyfit.db.models import Embedding, Image, Model
from findmyfit.db.session import create_session_factory


def main() -> None:
    settings = Settings.from_env()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        models = session.query(Model).all()
        print(f"Images: {session.query(Image).count()}")
        print(f"Embeddings: {session.query(Embedding).count()}")
        print("Models:")
        for model in models:
            count = session.query(Embedding).filter_by(model_id=model.id).count()
            print(
                f"  {model.name}/{model.version}: dim={model.embedding_dim}, "
                f"embeddings={count}"
            )
        print("Categories:")
        for category, count in (
            session.query(Image.category, func.count(Image.id))
            .group_by(Image.category)
            .order_by(Image.category)
        ):
            print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
