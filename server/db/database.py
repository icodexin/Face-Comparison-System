from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Connect to the MySQL database
DATABASE_URL = 'mysql+pymysql://root:hyang48%40sql@localhost/FCS'
engine = create_engine(DATABASE_URL, pool_recycle=3600)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=True)

# Create SQLAlchemy models for User and UserData tables
Base = declarative_base()
Base.metadata.create_all(engine)
