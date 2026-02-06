from sqlalchemy import (
    Column, String, Integer, LargeBinary, ForeignKey,
    DateTime, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True)  
    file_path = Column(String, nullable=False)
    category = Column(String, nullable=False)
    hash = Column(String, nullable=True)
    embeddings = relationship("Embedding", back_populates="image")


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    description = Column(String, nullable=True)

    embeddings = relationship("Embedding", back_populates="model")
    __table_args__ = (
    UniqueConstraint("name", "version", name="uix_model_name_version"),
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)

    image_id = Column(String, ForeignKey("images.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    vector = Column(LargeBinary, nullable=False)
    dim = Column(Integer, nullable=False)
    dtype = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now())

    image = relationship("Image", back_populates="embeddings")
    model = relationship("Model", back_populates="embeddings")

    __table_args__ = (UniqueConstraint("image_id", "model_id", name="uix_image_model"),)
