from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Medicament(Base):
    __tablename__ = 'medicaments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    
    # Classification
    laboratory = Column(Text)
    theraputic_class = Column(Text)
    pharmaco_class = Column(Text)
    dci = Column(Text)
    
    # Details
    commercial_name = Column(Text)
    dci_code = Column(Text)
    form = Column(Text)
    dosage = Column(Text)
    conditioning = Column(Text)
    
    # Status & Pricing
    type = Column(Text)
    list = Column(Text)
    country = Column(Text)
    marketed = Column(Boolean)
    reimbursable = Column(Boolean)
    reference_price = Column(Float)
    ppa_indicative = Column(Text)
    registration_num = Column(Text)
    
    # External Links
    notice_link = Column(Text)
    img_link = Column(Text)

# Update with your actual database credentials
DATABASE_URL = "postgresql://postgres:amine@localhost:5432/pharmnet"

def get_engine():
    return create_engine(DATABASE_URL)

Session = sessionmaker(bind=get_engine())

def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)
