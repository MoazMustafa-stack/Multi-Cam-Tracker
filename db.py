from sqlalchemy import create_engine, Column, Integer, String, Float,DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///tracks.db")

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key = True)
    global_id = Column(Integer)
    camera_id = Column(String)
    local_track_id = Column(Integer)
    frame_index = Column(Integer)
    timestamp = Column(DateTime)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    confidence = Column(Float)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
