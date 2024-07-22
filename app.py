import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, request, send_file
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from password_hashing import hash_password, verify_password
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
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import io

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
    date = "Unknown"
    time = "Unknown"

    success, frame = cap.read()
    frame_number = 0

    if success:
        img = frame
        height, width, _ = img.shape
        # Define the ROI for the bottom 10th of the screen and the far right for camera ID
        roi_height = height // 10
        camera_id_width = width // 6  # Adjusted to be smaller and more precise
        camera_id_img = img[height - roi_height:height, width - camera_id_width:width]

        # Define the ROI for the date and time
        date_time_height = height // 12
        date_time_width = width // 4  # Adjusted width for date and time
        date_img = img[height - date_time_height:height,
                   width // 3:width // 3 + date_time_width - 10]  # Slightly less on the right
        time_img = img[height - date_time_height:height, width // 3 + date_time_width:width - 10]  # More on the right

        # Process the camera ID image
        masked_camera_id_img = apply_mask(camera_id_img)
        preprocessed_camera_id_img = preprocess_image(masked_camera_id_img)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789CFPMA:/.°'  # Whitelist characters
        camera_id_text = pytesseract.image_to_string(preprocessed_camera_id_img, config=custom_config).strip()

        # Process the date image
        masked_date_img = apply_mask(date_img)
        preprocessed_date_img = preprocess_image(masked_date_img)
        date_text = pytesseract.image_to_string(preprocessed_date_img, config=custom_config).strip()

        # Process the time image
        masked_time_img = apply_mask(time_img)
        preprocessed_time_img = preprocess_image(masked_time_img)
        time_text = pytesseract.image_to_string(preprocessed_time_img, config=custom_config).strip()

        # Print the OCR text for debugging
        print(f"OCR Text for Camera ID: {camera_id_text}")
        print(f"OCR Text for Date: {date_text}")
        print(f"OCR Text for Time: {time_text}")

        # Extract the camera ID
        for line in camera_id_text.split("\n"):
            if len(line.strip()) == 4 and line.strip().isdigit():
                camera_id = line.strip()
                break

        # Extract the date and time
        if "/" in date_text:
            date = date_text.split()[0]
        if ":" in time_text:
            time = time_text.split()[0]

        temp_filepath = f'temp_frame_{frame_number}.jpg'
        cv2.imwrite(temp_filepath, frame)
        detected_animal = recognize_animal(temp_filepath)
        if detected_animal != "Unidentifiable":
            animal = detected_animal

        os.remove(temp_filepath)

    cap.release()
    return duration, camera_id, animal, pulled_data, date, time


def convert_to_mp4(filepath, filename):
    clip = VideoFileClip(filepath)
    new_filename = os.path.splitext(filename)[0] + ".mp4"
    new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    clip.write_videofile(new_filepath, codec='libx264')
    return new_filename


def process_video(filepath, filename):
    try:
        if not filename.lower().endswith('.mp4'):
            new_filename = convert_to_mp4(filepath, filename)
            os.remove(filepath)  # Remove the original file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            filename = new_filename

        duration, camera_id, animal, pulled_data, date, time = extract_metadata(filepath)
        username = session.get('username')
        if username:
            db.users.update_one(
                {'username': username},
                {'$push': {
                    'uploads': {'filepath': filename, 'duration': duration, 'camera_id': camera_id, 'animal': animal,
                                'pulled_data': pulled_data, 'date': date, 'time': time}}}
            )
        return {'status': 'success', 'message': 'Video processed successfully'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


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
        result = process_video(filepath, filename)
        flash(result['message'])
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
    file_ext = os.path.splitext(filename)[1].lower()
    mimetype = 'video/quicktime' if file_ext == '.mov' else 'video/mp4'
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mimetype)


@app.route('/change_password', methods=['POST'])
def change_password():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({'username': session['username']})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})

    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    if not current_password or not new_password or not confirm_password:
        return jsonify({'status': 'error', 'message': 'All fields are required'})

    if not verify_password(user['password'], current_password):
        return jsonify({'status': 'error', 'message': 'Current password is incorrect'})

    if new_password != confirm_password:
        return jsonify({'status': 'error', 'message': 'New passwords do not match'})

    new_password_hash = hash_password(new_password)
    db.users.update_one({'username': session['username']}, {'$set': {'password': new_password_hash}})

    return jsonify({'status': 'success', 'message': 'Password changed successfully'})


@app.route('/delete_upload', methods=['DELETE'])
def delete_upload():
    data = request.get_json()
    filepath = data.get('filepath')
    username = session.get('username')

    if not filepath or not username:
        return jsonify({'status': 'error', 'message': 'Invalid request'})

    user = db.users.find_one({'username': username})
    if not user or 'uploads' not in user:
        return jsonify({'status': 'error', 'message': 'No uploads found'})

    db.users.update_one(
        {'username': username},
        {'$pull': {'uploads': {'filepath': filepath}}}
    )

    full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    if os.path.exists(full_filepath):
        os.remove(full_filepath)

    return jsonify({'status': 'success', 'message': 'Upload deleted successfully'})

@app.route('/update_upload', methods=['PUT'])
def update_upload():
    data = request.get_json()
    filepath = data.get('filepath')
    camera_id = data.get('camera_id')
    animal = data.get('animal')
    date = data.get('date')
    time = data.get('time')
    username = session.get('username')

    if not filepath or not username or not camera_id or not animal or not date or not time:
        return jsonify({'status': 'error', 'message': 'Invalid request'})

    user = db.users.find_one({'username': username})
    if not user or 'uploads' not in user:
        return jsonify({'status': 'error', 'message': 'No uploads found'})

    db.users.update_one(
        {'username': username, 'uploads.filepath': filepath},
        {'$set': {
            'uploads.$.camera_id': camera_id,
            'uploads.$.animal': animal,
            'uploads.$.date': date,
            'uploads.$.time': time
        }}
    )

    return jsonify({'status': 'success', 'message': 'Upload updated successfully'})

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    fig.savefig(output, format='png')
    output.seek(0)
    return send_file(output, mimetype='image/png')

def create_figure():
    fig, ax = plt.subplots()
    ax.bar(['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'], [12, 19, 3, 5, 2, 3])
    ax.set_xlabel('Colors')
    ax.set_ylabel('Votes')
    ax.set_title('Votes by Color')
    return fig

@app.route('/get_uploads', methods=['GET'])
def get_uploads():
    username = session.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    sort_by = request.args.get('sort_by', 'camera')
    filter_value = request.args.get('filter', None)

    user = db.users.find_one({'username': username})
    if not user or 'uploads' not in user:
        return jsonify({'status': 'error', 'message': 'No uploads found'})

    uploads = user['uploads']

    data = []
    if sort_by == 'animal' and filter_value:
        camera_counts = {}
        for upload in uploads:
            if upload.get('animal') == filter_value:
                camera_id = upload.get('camera_id')
                camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
        data = [{'label': camera_id, 'count': count} for camera_id, count in camera_counts.items()]

    elif sort_by == 'camera':
        animal_types = ['Bear', 'Boar', 'Deer', 'Bobcat', 'Turkey', 'Unidentified']
        camera_counts = {}
        for upload in uploads:
            camera_id = upload.get('camera_id')
            if filter_value and camera_id != filter_value:
                continue
            animal = upload.get('animal')
            if camera_id not in camera_counts:
                camera_counts[camera_id] = {animal: 0 for animal in animal_types}
            camera_counts[camera_id][animal] = camera_counts[camera_id].get(animal, 0) + 1
        for camera_id, counts in camera_counts.items():
            for animal, count in counts.items():
                data.append({'label': animal, 'count': count})

    return jsonify(data)


@app.route('/get_camera_ids', methods=['GET'])
def get_camera_ids():
    username = session.get('username')

    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({'username': username})
    if not user or 'uploads' not in user:
        return jsonify({'status': 'error', 'message': 'No uploads found'})

    camera_ids = list(set(upload['camera_id'] for upload in user['uploads']))
    return jsonify(camera_ids)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
