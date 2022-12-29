# coding: utf-8
from sqlalchemy import CHAR, Column, String, Time
from sqlalchemy.dialects.mysql import DATETIME, SMALLINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Alarm(Base):
    __tablename__ = 'alarm'

    time = Column(DATETIME(fsp=3), primary_key=True, nullable=False)
    info = Column(String(32, 'utf8mb4_unicode_ci'), primary_key=True, nullable=False)
    val = Column(CHAR(1, 'utf8mb4_unicode_ci'))


class Counter(Base):
    __tablename__ = 'counter'

    time = Column(DATETIME(fsp=3), primary_key=True, nullable=False)
    info = Column(String(32, 'utf8mb4_unicode_ci'), primary_key=True, nullable=False)
    val = Column(SMALLINT(5))


class Timer(Base):
    __tablename__ = 'timer'

    time = Column(DATETIME(fsp=3), primary_key=True, nullable=False)
    info = Column(String(32, 'utf8mb4_unicode_ci'), primary_key=True, nullable=False)
    val = Column(Time)
