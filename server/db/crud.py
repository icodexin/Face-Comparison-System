from sqlalchemy.orm import Session
from . import schemas, models


def create_user(db: Session, user: schemas.CreateUser):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_user_data(db: Session, user_data: schemas.CreateUserData):
    db_data = models.UserData(**user_data.dict())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data


def get_user_by_name(db: Session, name: str):
    return db.query(models.User).filter(models.User.name == name).first()


def get_user_data(db: Session, user_id: int):
    return db.query(models.UserData).filter(models.UserData.user_id == user_id).first()