from sqlalchemy import Boolean, Column, Integer, String
from database import Base



class Notes(Base):
    __tablename__ = 'notes'

    id = Column(Integer,primary_key = True,index = True)
    userid = Column(Integer)
    title = Column(String(50))
    description = Column(String(100))
    content = Column(String(100))
    

    
    