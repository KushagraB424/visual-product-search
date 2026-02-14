from sqlalchemy import Column, Integer, String, Float, Text
from app.database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String, index=True)  # e.g., "Top", "Bottom", "Dress"
    price = Column(Float)
    currency = Column(String, default="USD")
    image_url = Column(String)  # URL to the image stored in S3/Cloudinary
    product_url = Column(String) # Link to buy the item
    description = Column(Text, nullable=True)