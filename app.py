import os
import numpy as np
from sqlalchemy.orm import sessionmaker
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, Column, Integer, String
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import webcolors
from db_operations import init_db, save_to_db, find_matching_cloth
from db_operations import init_db, save_to_db, find_matching_cloth, get_all_clothes
from colour import Color
from db_operations import Clothes
from sqlalchemy import desc

app = Flask(__name__)
engine = create_engine('sqlite:///wardrobe.db')

Session = sessionmaker(bind=engine)
CLOTHES_FOLDER = os.path.join('static', 'clothes')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = CLOTHES_FOLDER

model = MobileNetV2(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return [pred[1] for pred in decoded_preds]  

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

def get_dominant_color(image_path):
    image = Image.open(image_path)
    image = image.resize((100, 100))
    image_array = np.array(image)
    image_array = image_array.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(image_array)
    
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color.astype(int)

def get_colour_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        return closest_colour(rgb_tuple)
    
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        color_name = request.form['color'].lower()
        weather = request.form['weather'].lower()
        
        try:
            target_color = color_to_rgb(color_name)
            matching_cloth = find_matching_cloth(target_color, weather)
            if matching_cloth:
                filename, classification = matching_cloth
                image_url = url_for('static', filename=f'clothes/{filename}')
                return render_template('index.html', result=f"Recommended clothing item: {classification}", image_url=image_url)
            else:
                return render_template('index.html', result="No suitable clothing found for the given color and weather")
        except ValueError as e:
            return render_template('index.html', result=str(e))
    return render_template('index.html', result=None)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Classify the uploaded image
            classifications = classify_image(file_path)
            
            # Get the dominant color
            dominant_color = get_dominant_color(file_path)
            color_name = get_colour_name(tuple(dominant_color))
            
            new_filename = f"{classifications[0]}_{filename}"
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            os.rename(file_path, new_file_path)
            
            save_to_db(new_filename, classifications[0], color_name)
            
            return render_template('upload_result.html', filename=new_filename, classifications=classifications, color=color_name)
    return render_template('upload.html')

@app.route('/skip', methods=['GET'])
def skip():
    return redirect(url_for('index'))

@app.route('/see_all_clothes')
def see_all_clothes():
    session = Session()
    
    # Get all clothes
    clothes = session.query(Clothes).all()
    
    # Get top 5 popular clothes by view count
    popular_clothes = session.query(Clothes).order_by(desc(Clothes.view_count)).limit(5).all()
    
    session.close()
    
    return render_template('all_clothes.html', clothes=clothes, popular_clothes=popular_clothes)

if __name__ == '__main__':
    app.run(debug=True, port=5003)