from pymongo import MongoClient, errors
from password_hashing import hash_password, verify_password
import datetime
from flask import url_for

def create_connection():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.trailcamapp
    return db

def create_collection(db):
    if "users" not in db.list_collection_names():
        db.create_collection("users")
        db.users.create_index("username", unique=True)
        print("Collection 'users' created successfully.")
    else:
        print("Collection 'users' already exists.")

def register_user(db, username, password, name, email):
    hashed_password = hash_password(password)
    try:
        db.users.insert_one({
            "username": username,
            "password": hashed_password,
            "name": name,
            "email": email,
            "photos": [],
            "gps_pins": [],
            "uploads": [],
            "scan_history": [],
            "profile_picture_url": url_for('static', filename='default_profile.png')  # Add default profile picture
        })
        return True, f"User {username} registered successfully!"
    except errors.DuplicateKeyError:
        return False, "Username already exists. Try a different one."

def login_user(db, username, provided_password):
    user = db.users.find_one({"username": username})
    if user:
        print(f"User found: {user}")
        print(f"Stored password hash: {user['password']}")
        print(f"Provided password: {provided_password}")
        if verify_password(user['password'], provided_password):
            print("Password verified")
            return True, user
        else:
            print("Password verification failed")
            return False, None
    else:
        print("User not found")
        return False, None

def log_scan(db, username, filename):
    scan_entry = {
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename
    }
    db.users.update_one(
        {'username': username},
        {'$push': {'scan_history': scan_entry}}
    )
