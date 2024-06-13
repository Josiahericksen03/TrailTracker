from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from database import create_connection, create_collection, register_user, login_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'

db = create_connection()
create_collection(db)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
