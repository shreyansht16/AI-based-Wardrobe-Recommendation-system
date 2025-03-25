import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import webcolors
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import webcolors
from colour import Color

Base = declarative_base()

class Clothes(Base):
    __tablename__ = 'clothes'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    classification = Column(String, nullable=False)
    color = Column(String, nullable=False)
    view_count = Column(Integer, default=0)  # Add this line for view count


engine = create_engine('sqlite:///wardrobe.db')
Session = sessionmaker(bind=engine)


def color_to_rgb(color_name):
    try:
        # First, try to convert using webcolors
        rgb = webcolors.name_to_rgb(color_name)
    except ValueError:
        try:
            # If webcolors fails, try using the colour library
            rgb = tuple(int(x * 255) for x in Color(color_name).rgb)
        except ValueError:
            # If both fail, raise an exception
            raise ValueError(f"Invalid color name: {color_name}")
    return rgb

def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized successfully")

def save_to_db(filename, classification, color):
    session = Session()
    new_cloth = Clothes(filename=filename, classification=classification, color=color)
    session.add(new_cloth)
    session.commit()
    session.close()

from sqlalchemy.orm import sessionmaker

# Create a session factory
Session = sessionmaker(bind=engine)

def find_matching_cloth(target_color, weather):
    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2))
    
    best_match = None
    min_distance = float('inf')

    # Open a new session
    session = Session()  # Keep the session open during the search process

    try:
        clothes = session.query(Clothes).all()  # Query all clothes

        for cloth in clothes:
            cloth_color = color_to_rgb(cloth.color)
            if cloth_color is None:
                continue  # Skip if color conversion failed

            distance = color_distance(target_color, cloth_color)

            # Check if the cloth is suitable for the weather and has the minimum distance
            if is_suitable_for_weather(cloth.classification, weather) and distance < min_distance:
                min_distance = distance
                best_match = cloth

        # If a best match is found, update its view count and commit the change
        if best_match:
            best_match.view_count = (best_match.view_count or 0) + 1
            session.commit()  # Save the updated view count to the database

        # Return the filename and classification of the best match, if any
        return (best_match.filename, best_match.classification) if best_match else None

    except Exception as e:
        print(f"Error during cloth matching: {e}")
        return None
    
    finally:
        # Always close the session after the operation is done
        session.close()






def get_all_clothes():
    session = Session()
    clothes = session.query(Clothes).all()
    session.close()
    return clothes

def is_suitable_for_weather(classifications, weather):
    WEATHER_SUITABILITY = {
        'sunny': ['t-shirt', 'shorts', 'dress', 'jersey', 'skirt', 'sandal', 'chain_mail', 'cowboy_hat', 'bikini', 'swimsuit', 'cap'],
        'rainy': ['raincoat', 'umbrella', 'boots', 'jacket', 'cardigan', 'sweatshirt', 'hat'],
        'cold': ['sweater', 'coat', 'scarf', 'gloves', 'trench_coat', 'cloak', 'velvet', 'suit']
    }
    
    weather_items = WEATHER_SUITABILITY.get(weather, [])
    # Checking if any of the weather items are present in the classification string (case-insensitive)
    return any(item.lower() in classifications for item in weather_items)


if __name__ == '__main__':
    init_db()
    print("Database operations module is running")
