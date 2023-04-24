from sqlalchemy import create_engine, Column, Integer, Float, Date
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///oil_prices.db')
Base = declarative_base()

class OilPrices(Base):
    __tablename__ = 'oil_prices'

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    price = Column(Float)