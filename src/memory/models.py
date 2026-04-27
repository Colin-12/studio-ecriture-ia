"""SQLAlchemy models for structured novel memory."""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Novel(Base):
    __tablename__ = "novels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    chapters: Mapped[List["Chapter"]] = relationship(
        back_populates="novel", cascade="all, delete-orphan"
    )
    characters: Mapped[List["Character"]] = relationship(
        back_populates="novel", cascade="all, delete-orphan"
    )
    locations: Mapped[List["Location"]] = relationship(
        back_populates="novel", cascade="all, delete-orphan"
    )
    events: Mapped[List["Event"]] = relationship(
        back_populates="novel", cascade="all, delete-orphan"
    )
    setup_payoffs: Mapped[List["SetupPayoff"]] = relationship(
        back_populates="novel", cascade="all, delete-orphan"
    )


class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    novel_id: Mapped[int] = mapped_column(ForeignKey("novels.id"), nullable=False)
    number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    novel: Mapped["Novel"] = relationship(back_populates="chapters")
    events: Mapped[List["Event"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan"
    )
    setups: Mapped[List["SetupPayoff"]] = relationship(
        back_populates="setup_chapter",
        foreign_keys="SetupPayoff.setup_chapter_id",
    )
    payoffs: Mapped[List["SetupPayoff"]] = relationship(
        back_populates="payoff_chapter",
        foreign_keys="SetupPayoff.payoff_chapter_id",
    )


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    novel_id: Mapped[int] = mapped_column(ForeignKey("novels.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    novel: Mapped["Novel"] = relationship(back_populates="characters")


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    novel_id: Mapped[int] = mapped_column(ForeignKey("novels.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    novel: Mapped["Novel"] = relationship(back_populates="locations")
    events: Mapped[List["Event"]] = relationship(back_populates="location")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    novel_id: Mapped[int] = mapped_column(ForeignKey("novels.id"), nullable=False)
    chapter_id: Mapped[int] = mapped_column(ForeignKey("chapters.id"), nullable=False)
    location_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("locations.id"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sequence_order: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    novel: Mapped["Novel"] = relationship(back_populates="events")
    chapter: Mapped["Chapter"] = relationship(back_populates="events")
    location: Mapped[Optional["Location"]] = relationship(back_populates="events")


class SetupPayoff(Base):
    __tablename__ = "setup_payoffs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    novel_id: Mapped[int] = mapped_column(ForeignKey("novels.id"), nullable=False)
    setup_chapter_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("chapters.id"), nullable=True
    )
    payoff_chapter_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("chapters.id"), nullable=True
    )
    setup_text: Mapped[str] = mapped_column(Text, nullable=False)
    payoff_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    novel: Mapped["Novel"] = relationship(back_populates="setup_payoffs")
    setup_chapter: Mapped[Optional["Chapter"]] = relationship(
        back_populates="setups", foreign_keys=[setup_chapter_id]
    )
    payoff_chapter: Mapped[Optional["Chapter"]] = relationship(
        back_populates="payoffs", foreign_keys=[payoff_chapter_id]
    )
