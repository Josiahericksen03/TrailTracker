import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed, FileRequired
from database import create_connection, create_collection, register_user, login_user
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import pytesseract

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = create_connection()
create_collection(db)

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 6  # Number of animal classes
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('animal_model.pth', map_location=device))
model = model.to(device)
model.eval()

class_names = ['Bear', 'Boar', 'Bobcat', 'Deer', 'Turkey', 'Unidentifiable']

def recognize_animal(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def apply_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst

def extract_metadata(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video {filepath}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / frame_rate

    camera_id = "Unknown"
    animal = "Unidentifiable"
    pulled_data = ""

    success, frame = cap.read()
    frame_number = 0

    if success:
        img = frame
        height, width, _ = img.shape
        # Define the ROI for the bottom 10th of the screen and the far right
        roi_height = height // 10
        roi_width = width // 6  # Adjusted to be smaller and more precise
        cropped_img = img[height - roi_height:height, width - roi_width:width]
        masked_img = apply_mask(cropped_img)
        preprocessed_img = preprocess_image(masked_img)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789CF:/.°'  # Whitelist characters
        text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
        pulled_data = text.strip()

        # Print the OCR text for debugging
        print(f"OCR Text: {text}")

        # Extract the camera ID
        for line in text.split("\n"):
            if len(line.strip()) == 4 and line.strip().isdigit():
                camera_id = line.strip()
                break

        if camera_id == "Unknown" or camera_id == "":
            # Retry with a slightly adjusted region
            roi_width = width // 8
            cropped_img = img[height - roi_height:height, width - roi_width:width]
            masked_img = apply_mask(cropped_img)
            preprocessed_img = preprocess_image(masked_img)
            text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
            pulled_data = text.strip()
            print(f"Retry OCR Text: {text}")
            for line in text.split("\n"):
                if len(line.strip()) == 4 and line.strip().isdigit():
                    camera_id = line.strip()
                    break

        temp_filepath = f'temp_frame_{frame_number}.jpg'
        cv2.imwrite(temp_filepath, frame)
        detected_animal = recognize_animal(temp_filepath)
        if detected_animal != "Unidentifiable":
            animal = detected_animal

        os.remove(temp_filepath)

    cap.release()
    return duration, camera_id, animal, pulled_data

def process_video(filepath):
    try:
        duration, camera_id, animal, pulled_data = extract_metadata(filepath)
        username = session.get('username')
        if username:
            db.users.update_one(
                {'username': username},
                {'$push': {'uploads': {'filepath': filepath, 'duration': duration, 'camera_id': camera_id, 'animal': animal, 'pulled_data': pulled_data}}}
            )
        return jsonify({'status': 'success', 'message': 'Video processed successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


class UploadForm(FlaskForm):
    file = FileField('Video File', validators=[FileRequired(), FileAllowed(['mp4', 'avi', 'mov'], 'Videos only!')])
    submit = SubmitField('Upload')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = process_video(filepath)
        flash(result.json['message'])
        return redirect(url_for('upload'))
    return render_template('upload.html', title='Upload', form=form)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']

        success, message = register_user(db, username, password, name, email)
        if success:
            session['username'] = username
            flash(message)
            return redirect(url_for('home'))
        else:
            flash(message)
            return redirect(url_for('signup'))

    return render_template('signup.html', title='Sign Up')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        print(f"Attempting to log in user: {username}")
        success, user = login_user(db, username, password)
        if success:
            print(f"Login successful for user: {username}")
            session['username'] = username
            flash('Login successful')
            return redirect(url_for('home'))
        else:
            print("Login failed: Invalid username or password")
            flash('Invalid username or password')
            return redirect(url_for('login'))

    return render_template('login.html', title='Log In')

@app.route('/home')
def home():
    if 'username' in session:
        username = session['username']
        return render_template('home.html', username=username, title='Home')
    else:
        flash('You need to log in first')
        return redirect(url_for('login'))

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    flash('You have been logged out')
    return redirect(url_for('login'))

@app.route('/users')
def users():
    if 'username' not in session:
        flash('You need to log in first')
        return redirect(url_for('login'))

    all_users = db.users.find()
    return render_template('users.html', users=all_users, title='User List')

@app.route('/save_location', methods=['POST'])
def save_location():
    data = request.get_json()
    username = session.get('username')
    if username:
        db.users.update_one(
            {'username': username},
            {'$set': {'location': {'latitude': data['latitude'], 'longitude': data['longitude']}}}
        )
        return jsonify({'status': 'success', 'message': 'Location saved successfully'})
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/save_pin', methods=['POST'])
def save_pin():
    data = request.get_json()
    username = session.get('username')
    if username:
        db.users.update_one(
            {'username': username},
            {'$push': {'gps_pins': data}}
        )
        return jsonify({'status': 'success', 'message': 'Pin saved successfully'})
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/get_pins', methods=['GET'])
def get_pins():
    username = session.get('username')
    if username:
        user = db.users.find_one({'username': username})
        if user and 'gps_pins' in user:
            return jsonify({'status': 'success', 'pins': user['gps_pins']})
        return jsonify({'status': 'error', 'message': 'No pins found'})
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/delete_pin/<camera_id>', methods=['DELETE'])
def delete_pin(camera_id):
    username = session.get('username')
    if username:
        db.users.update_one(
            {'username': username},
            {'$pull': {'gps_pins': {'camera_id': camera_id}}}
        )
        return jsonify({'status': 'success', 'message': 'Pin deleted successfully'})
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/profile')
def profile():
    if 'username' in session:
        username = session['username']
        user = db.users.find_one({'username': username})
        return render_template('profile.html', user=user, title='Profile')
    else:
        flash('You need to log in first')
        return redirect(url_for('login'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' in session:
        username = session['username']
        user = db.users.find_one({'username': username})

        if request.method == 'POST':
            name = request.form['name']
            email = request.form['email']

            db.users.update_one(
                {'username': username},
                {'$set': {'name': name, 'email': email}}
            )
            flash('Settings updated successfully')
            return redirect(url_for('settings'))

        return render_template('settings.html', user=user, title='Settings')
    else:
        flash('You need to log in first')
        return redirect(url_for('login'))


@app.route('/get_uploads_by_camera/<camera_id>', methods=['GET'])
def get_uploads_by_camera(camera_id):
    username = session.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({'username': username})
    if not user or 'uploads' not in user:
        return jsonify({'status': 'error', 'message': 'No uploads found'})

    uploads = [upload for upload in user['uploads'] if upload.get('camera_id') == camera_id]
    return jsonify({'status': 'success', 'uploads': uploads})

@app.route('/update_pin/<camera_id>', methods=['PUT'])
def update_pin(camera_id):
    data = request.get_json()
    username = session.get('username')
    if username:
        db.users.update_one(
            {'username': username, 'gps_pins.camera_id': camera_id},
            {'$set': {'gps_pins.$.name': data['name'], 'gps_pins.$.camera_id': data['camera_id']}}
        )
        return jsonify({'status': 'success', 'message': 'Pin updated successfully'})
    return jsonify({'status': 'error', 'message': 'User not logged in'})
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({'username': session['username']})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})

    new_username = request.form.get('username')
    if not new_username:
        return jsonify({'status': 'error', 'message': 'Username is required'})

    # Update username
    db.users.update_one({'username': session['username']}, {'$set': {'username': new_username}})
    session['username'] = new_username  # Update session with new username

    # Update profile picture if provided
    profile_picture = request.files.get('profile_picture')
    if profile_picture:
        filename = secure_filename(profile_picture.filename)
        profile_picture_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        profile_picture.save(profile_picture_path)
        profile_picture_url = url_for('uploaded_file', filename=filename)
        db.users.update_one({'username': session['username']}, {'$set': {'profile_picture_url': profile_picture_url}})
    else:
        profile_picture_url = user.get('profile_picture_url', url_for('static', filename='default_profile.png'))

    return jsonify({'status': 'success', 'username': new_username, 'profile_picture_url': profile_picture_url})

# Serve the uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
