import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, MultipleFileField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed, FileRequired
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import pytesseract
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

API_URL = 'http://localhost:5001/api'

# Setup MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client.trailcamapp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_classes = 6  # Number of animal classes
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('animal_model.pth', map_location=device))
model = model.to(device)
model.eval()

class_names = ['Bear', 'Boar', 'Bobcat', 'Deer', 'Turkey', 'Unidentifiable']

class UploadForm(FlaskForm):
    files = MultipleFileField('Video or Image Files', validators=[
        FileRequired(),
        FileAllowed(['mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg'], 'Videos and images only!')
    ])
    submit = SubmitField('Upload')


# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

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


def apply_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    return cv2.merge(rgba, 4)


def extract_metadata(filepath):
    """Extract metadata like animal type and camera details from a single frame in a video or an image."""
    camera_id = "Unknown"
    animal = "Unidentifiable"
    pulled_data = ""
    date = "Unknown"
    time = "Unknown"

    img = cv2.imread(filepath)
    height, width, _ = img.shape

    # Camera ID extraction
    camera_id_img = img[height - height // 10:height, width - width // 6:width]
    masked_camera_id_img = apply_mask(camera_id_img)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789CFPMA:/.°'
    camera_id_text = pytesseract.image_to_string(masked_camera_id_img, config=custom_config).strip()

    for line in camera_id_text.split("\n"):
        if len(line.strip()) == 4 and line.strip().isdigit():
            camera_id = line.strip()
            break

    # Date and Time extraction
    date_time_height = height // 12
    date_time_width = width // 4
    date_img = img[height - date_time_height:height, width // 3:width // 3 + date_time_width - 10]
    time_img = img[height - date_time_height:height, width // 3 + date_time_width - 30:width - 20]

    # Preprocess for OCR
    masked_date_img = apply_mask(date_img)
    preprocessed_date_img = preprocess_image(masked_date_img)
    date_text = pytesseract.image_to_string(preprocessed_date_img, config=custom_config).strip()

    masked_time_img = apply_mask(time_img)
    preprocessed_time_img = preprocess_image(masked_time_img)
    time_text = pytesseract.image_to_string(preprocessed_time_img, config=custom_config).strip()

    if "/" in date_text:
        date = date_text.split()[0]
    if ":" in time_text:
        time = time_text.split()[0]

    # Detect animal
    detected_animal = recognize_animal(filepath)
    if detected_animal != "Unidentifiable":
        animal = detected_animal

    return camera_id, animal, pulled_data, date, time


def process_image(filepath, filename):
    try:
        camera_id, animal, pulled_data, date, time = extract_metadata(filepath)
        upload_data = {
            "filepath": filename,
            "camera_id": camera_id,
            "animal": animal,
            "pulled_data": pulled_data,
            "date": date,
            "time": time
        }
        return save_upload_to_db(upload_data)
    except Exception as e:
        print(f'Error in process_image: {e}')
        return {'status': 'error', 'message': str(e)}


def process_video(filepath, filename):
    try:
        # Open video file
        cap = cv2.VideoCapture(filepath)

        # Move to the 10th frame
        frame_count = 0
        target_frame = 10  # The frame we want to capture

        while frame_count < target_frame:
            success, frame = cap.read()
            if not success:
                return {'status': 'error', 'message': 'Could not read the 10th frame'}
            frame_count += 1

        # Define the image file path to save the frame
        image_filename = f"{os.path.splitext(filename)[0]}_frame.jpg"
        image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        # Save the 10th frame as an image
        cv2.imwrite(image_filepath, frame)

        # Extract metadata from the saved image
        camera_id, animal, pulled_data, date, time = extract_metadata(image_filepath)
        upload_data = {
            "filepath": image_filename,
            "camera_id": camera_id,
            "animal": animal,
            "pulled_data": pulled_data,
            "date": date,
            "time": time
        }

        # Clean up: Release the video capture and delete the original video file
        cap.release()
        os.remove(filepath)  # Remove the video file

        # Save the processed image data to the database
        return save_upload_to_db(upload_data)
    except Exception as e:
        print(f'Error in process_video: {e}')
        return {'status': 'error', 'message': str(e)}


def save_upload_to_db(upload_data):
    username = session.get('username')
    if username:
        db.users.update_one(
            {"username": username},
            {"$push": {"uploads": upload_data}}
        )
        return {'status': 'success', 'message': 'Upload successful'}
    else:
        return {'status': 'error', 'message': 'User not logged in'}


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if 'username' not in session:
        flash('You need to log in first')
        return redirect(url_for('login'))

    username = session['username']
    user = db.users.find_one({"username": username})

    if not user:
        flash('User not found')
        return redirect(url_for('login'))

    if form.validate_on_submit():
        messages = []  # Collect messages for each file
        for file in form.files.data:  # Access form.files instead of form.file
            filename = secure_filename(file.filename)

            # Check if the file already exists in the database
            existing_file = db.users.find_one({"uploads.filepath": filename})
            if existing_file:
                messages.append(f'File {filename} already exists. Skipping upload for this file.')
                continue  # Skip existing files

            # Save and process each file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Determine file type and process accordingly
            if is_image_file(filename):
                result = process_image(filepath, filename)
            elif is_video_file(filename):
                result = process_video(filepath, filename)
            else:
                messages.append(f'Unsupported file format: {filename}. Skipping this file.')
                continue  # Skip unsupported formats

            # Append success or error message
            messages.append(result['message'])

        # Flash each collected message for display
        for message in messages:
            flash(message)
        return redirect(url_for('upload'))

    return render_template('upload.html', title='Upload', form=form, user=user)



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

        data = {
            'username': username,
            'password': password,
            'name': name,
            'email': email
        }
        
        # Add debug logging
        print(f"Making API call for registration...")
        result = call_api('users/register', 'post', data)
        print(f"API result: {result}")

        if result is None:
            print("API call returned None")
            flash('An unexpected error occurred. Please try again.')
            return redirect(url_for('signup'))

        # Check the exact message from the API
        print(f"Checking message: {result.get('message')}")
        print(f"Expected message: User {username} registered successfully!")
        
        if result.get('message') == f'User {username} registered successfully!':
            print("Registration successful, setting session")
            session['username'] = username
            print("Redirecting to home")
            return redirect(url_for('home'))
        else:
            print(f"Registration failed with message: {result.get('message')}")
            flash(result.get('message', 'An unexpected error occurred. Please try again.'))
            return redirect(url_for('signup'))

    return render_template('signup.html', title='Sign Up')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        data = {
            'username': username,
            'password': password
        }

        # Log the data being sent to the API
        print(f'Sending login request with data: {data}')

        result = call_api('users/login', 'post', data)

        # Log the API response
        print(f'API response: {result}')

        flash(result['message'])
        if result['message'] == 'Login successful':
            session['username'] = username
            return redirect(url_for('home'))
        else:
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
    return jsonify({'status': 'success', 'message': 'Logged out successfully'})


@app.route('/users')
def users():
    if 'username' not in session:
        flash('You need to log in first')
        return redirect(url_for('login'))

    result = call_api('users', 'get')
    all_users = result.get('users', [])
    return render_template('users.html', users=all_users, title='User List')

@app.route('/save_location', methods=['POST'])
def save_location():
    data = request.get_json()
    username = session.get('username')
    if username:
        result = call_api('users/save_location', 'post', data)
        return jsonify(result)
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/save_pin', methods=['POST'])
def save_pin():
    data = request.get_json()
    username = session.get('username')
    if username:
        data['username'] = username
        result = call_api('users/save_pin', 'post', data)
        return jsonify(result)
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/get_pins', methods=['GET'])
def get_pins():
    username = session.get('username')
    if username:
        data = {'username': username}
        result = call_api('users/get_pins', 'post', data)
        return jsonify(result)
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/profile')
def profile():
    if 'username' in session:
        username = session['username']
        user = db.users.find_one({"username": username})
        if not user:
            flash('User not found')
            return redirect(url_for('login'))

        return render_template('profile.html', user=user, title='Profile')
    else:
        flash('You need to log in first')
        return redirect(url_for('login'))

@app.route('/get_uploads_by_camera/<camera_id>', methods=['GET'])
def get_uploads_by_camera(camera_id):
    username = session.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({"username": username})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})

    uploads = [upload for upload in user['uploads'] if upload['camera_id'] == camera_id]
    return jsonify({'status': 'success', 'uploads': uploads})

@app.route('/update_pin/<camera_id>', methods=['PUT'])
def update_pin(camera_id):
    data = request.get_json()
    username = session.get('username')
    if username:
        data['username'] = username
        result = call_api(f'users/update_pin/{camera_id}', 'put', data)
        return jsonify(result)
    return jsonify({'status': 'error', 'message': 'User not logged in'})

@app.route('/delete_pin/<camera_id>', methods=['DELETE'])
def delete_pin(camera_id):
    username = session.get('username')
    if username:
        print(f"Deleting pin for user: {username}, camera_id: {camera_id}")
        result = call_api(f'users/delete_pin/{camera_id}', 'delete', {'username': username})
        print(f"API call result: {result}")
        if result:
            print(f"API result status: {result.get('status')}, message: {result.get('message')}")
            if result.get('status') == 'success':
                return jsonify({'status': 'success', 'message': 'Pin deleted successfully'})
        return jsonify(result)
    print("User not logged in")
    return jsonify({'status': 'error', 'message': 'User not logged in'})
@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    current_username = session['username']
    new_username = request.form.get('username')
    profile_picture = request.files.get('profile_picture')

    update_data = {'current_username': current_username, 'username': new_username}

    if profile_picture:
        filename = secure_filename(profile_picture.filename)
        profile_picture_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        profile_picture.save(profile_picture_path)
        update_data['profile_picture_url'] = url_for('uploaded_file', filename=filename, _external=True)

    try:
        print('Sending data to API:', update_data)
        response = requests.post('http://localhost:5001/api/users/update_profile', data=update_data)
        if response.status_code == 200:
            session['username'] = new_username  # Update session with new username if changed
            return response.json()
        else:
            print('API error response:', response.json())
            return response.json(), response.status_code
    except Exception as e:
        print('Error during API call:', e)
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/change_password', methods=['POST'])
def change_password():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    data = request.get_json()
    data['username'] = session['username']
    result = call_api('users/change_password', 'post', data)
    return jsonify(result)

# Update call_api function if not already updated
def call_api(endpoint, method='get', data=None):
    api_url = f'http://localhost:5001/api/{endpoint}'
    print(f"Calling API: {api_url} with method: {method} and data: {data}")
    
    try:
        if method.lower() == 'get':
            response = requests.get(api_url)
        elif method.lower() == 'post':
            response = requests.post(api_url, json=data)
        elif method.lower() == 'put':
            response = requests.put(api_url, json=data)
        elif method.lower() == 'delete':
            response = requests.delete(api_url, json=data)
        else:
            return None

        print(f"API call to {api_url} returned status {response.status_code}")
        
        # Handle both 200 and 201 status codes as success
        if response.status_code in [200, 201]:
            return response.json()
        
        return None
    except Exception as e:
        print(f"API call failed with error: {str(e)}")
        return None


@app.route('/update_upload', methods=['PUT'])
def update_upload():
    data = request.get_json()
    username = session.get('username')

    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({"username": username})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})

    db.users.update_one(
        {'username': username, 'uploads.filepath': data['filepath']},
        {'$set': {
            'uploads.$.camera_id': data['camera_id'],
            'uploads.$.animal': data['animal'],
            'uploads.$.date': data['date'],
            'uploads.$.time': data['time']
        }}
    )
    return jsonify({'status': 'success', 'message': 'Upload updated successfully'})

@app.route('/delete_upload', methods=['DELETE'])
def delete_upload():
    data = request.get_json()
    filepath = data.get('filepath')
    username = session.get('username')

    if not filepath or not username:
        return jsonify({'status': 'error', 'message': 'Invalid request'})

    user = db.users.find_one({"username": username})
    if user:
        db.users.update_one(
            {"username": username},
            {"$pull": {"uploads": {"filepath": filepath}}}
        )
        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
        if os.path.exists(full_filepath):
            os.remove(full_filepath)
        return jsonify({'status': 'success', 'message': 'Upload deleted successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'User not found'})
@app.route('/api/users/get_uploads', methods=['POST'])
def get_uploads():
    sort_by = request.json.get('sort_by')
    filter_value = request.json.get('filter_value')
    username = request.json.get('username')

    user = db.users.find_one({"username": username})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404

    uploads = user.get('uploads', [])
    animal_names = ['Bear', 'Boar', 'Turkey', 'Deer', 'Bobcat', 'Other']

    def format_data(grouped_data, keys):
        formatted_data = []
        for key in keys:
            formatted_data.append({
                'label': key,
                'count': grouped_data.get(key, 0)
            })
        return formatted_data

    if sort_by == 'camera' and filter_value:
        filtered_uploads = [upload for upload in uploads if upload['camera_id'] == filter_value]
        grouped_uploads = {}
        for upload in filtered_uploads:
            animal = upload['animal'] if upload['animal'] != 'Unidentifiable' else 'Other'
            if animal not in grouped_uploads:
                grouped_uploads[animal] = 0
            grouped_uploads[animal] += 1
        formatted_data = format_data(grouped_uploads, animal_names)
        return jsonify({'status': 'success', 'uploads': formatted_data}), 200
    elif sort_by == 'animal' and filter_value:
        cameras = [pin['camera_id'] for pin in user['gps_pins']]
        filtered_uploads = [upload for upload in uploads if upload['animal'] == filter_value]
        grouped_uploads = {camera: 0 for camera in cameras}
        for upload in filtered_uploads:
            grouped_uploads[upload['camera_id']] += 1
        formatted_data = [{'label': camera, 'count': count} for camera, count in grouped_uploads.items()]
        return jsonify({'status': 'success', 'uploads': formatted_data}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Invalid sort option or filter value'}), 400



@app.route('/api/users/get_camera_ids', methods=['GET'])
def get_camera_ids():
    username = request.args.get('username')
    user = db.users.find_one({"username": username})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404

    # Get camera IDs from gps_pins instead of uploads
    camera_ids = [pin['camera_id'] for pin in user['gps_pins']]
    return jsonify({'status': 'success', 'camera_ids': camera_ids}), 200

@app.route('/get_latest_uploads', methods=['GET'])
def get_latest_uploads():
    username = session.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'User not logged in'})

    user = db.users.find_one({"username": username})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})

    # Convert ObjectId to string in uploads
    uploads = user.get('uploads', [])
    for upload in uploads:
        if '_id' in upload:
            upload['_id'] = str(upload['_id'])

    return jsonify({
        'status': 'success',
        'uploads': uploads
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
##############################
