"""Create the configured catalog database schema."""

from findmyfit.config import Settings
from findmyfit.db.models import Base
from findmyfit.db.session import create_engine_for_url


def main() -> None:
    settings = Settings.from_env()
    Base.metadata.create_all(bind=create_engine_for_url(settings.database_url))
    print("Database schema is ready.")


if __name__ == "__main__":
    main()
