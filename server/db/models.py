from fastapi import FastAPI
from sqlalchemy import Column, Integer, String, BLOB, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = 'User'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    password = Column(String(50), nullable=False)
    data = relationship('UserData', back_populates='user')


class UserData(Base):
    __tablename__ = 'UserData'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('User.id'), nullable=False, unique=True)
    image = Column(BLOB, nullable=False)
    bbox = Column(String(1000), nullable=False)
    encoding = Column(String(5000), nullable=False)
    user = relationship('User', back_populates='data')
