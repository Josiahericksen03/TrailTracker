# Trail Tracker

A web application for managing trail camera footage with AI-powered animal detection and GPS-based camera tracking.

## What It Does

Trail Tracker helps you organize and analyze trail camera data. Upload images and videos from your trail cameras, and the app will automatically extract metadata (camera ID, date, time) and identify animals using AI. You can manage your camera locations on an interactive map, view upload history, and see statistics about your wildlife sightings.

## Features

### 🎯 Core Functionality
- **AI Animal Detection**: Identifies Bear, Boar, Bobcat, Deer, Turkey, and Unidentifiable animals using a trained ResNet18 model
- **Metadata Extraction**: Uses OCR to extract camera ID, date, and time from trail camera images
- **Video Processing**: Extracts the 10th frame from videos for analysis
- **Camera Management**: Add camera locations with GPS coordinates and manage them on a map
- **Upload History**: View all your uploads with filtering and editing capabilities

### 🗺️ Interactive Map
- **GPS Location**: Get your current location coordinates
- **Camera Pins**: See all your camera locations on an interactive map
- **Map Layers**: Switch between satellite, topographic, street, and humanitarian views
- **Weather Overlay**: Optional weather data display

### 📊 Data Analytics
- **Statistics Dashboard**: View animal sightings by camera or species
- **Interactive Charts**: Visual charts showing your data
- **Filtering**: Sort by camera ID or animal type
- **Data Management**: Edit upload information and delete files

### 👤 User Management
- **User Accounts**: Register and login system
- **Profile Settings**: Update username, profile picture, and password
- **Session Management**: Stay logged in across browser sessions
- **Personal Data**: Each user has their own uploads and camera network

### 🖼️ Media Management
- **File Viewer**: Built-in viewer for images and videos
- **Multiple Uploads**: Upload several files at once
- **File Organization**: Files are organized by camera and date
- **Edit Metadata**: Update camera ID, animal type, date, and time for uploads

## Tech Stack

### Frontend
- **HTML5/CSS3**: Responsive web design with modern styling
- **JavaScript (ES6+)**: Interactive client-side functionality
- **Chart.js**: Data visualization and analytics
- **Leaflet.js**: Interactive mapping and GPS functionality

### Backend
- **Flask**: Python web framework for API and server-side logic
- **MongoDB**: NoSQL database for flexible data storage
- **PyMongo**: MongoDB driver for Python

### AI/ML
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision models and transforms
- **ResNet18**: Pre-trained neural network for animal classification
- **OpenCV**: Image and video processing
- **Tesseract OCR**: Text extraction from images

### Media Processing
- **FFmpeg**: Video processing and frame extraction
- **Pillow (PIL)**: Image processing and manipulation
- **MoviePy**: Video editing and processing

### Development Tools
- **Flask-WTF**: Form handling and validation
- **Alembic**: Database migrations
- **Requests**: HTTP client for API calls

## How to Run Locally

### Prerequisites
- Python 3.8+
- Node.js 14+
- MongoDB
- FFmpeg
- Tesseract OCR

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Josiahericksen03/TrailTracker
   cd Trailtracker
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install system dependencies**
   ```bash
   # macOS
   brew install mongodb-community ffmpeg tesseract
   
   # Ubuntu/Debian
   sudo apt-get install mongodb ffmpeg tesseract-ocr
   
   # Windows
   # Download and install MongoDB, FFmpeg, and Tesseract separately
   ```

4. **Start MongoDB**
   ```bash
   # macOS
   brew services start mongodb-community
   
   # Ubuntu/Debian
   sudo systemctl start mongod
   
   # Windows
   # Start MongoDB service from Windows Services
   ```

5. **Set up the Backend API**
   ```bash
   # Navigate to the backend directory (separate repository)
   cd ../trailtracker-backend
   npm install
   node server.js
   ```
   *Note: The backend runs on port 5001 and handles user authentication and additional API endpoints.*

6. **Start the Flask application**
   ```bash
   # Return to the main project directory
   cd ../Trailtracker
   python app.py
   ```

7. **Access the application**
   - Open your browser and go to `http://localhost:8080`
   - Register a new account or log in with existing credentials
   - Start uploading trail camera footage!

### Configuration

The application uses these default settings:
- **Database**: MongoDB running on localhost:27017
- **Upload folder**: `uploads/` directory in the project
- **AI model**: `animal_model.pth` (43MB trained model)
- **Backend API**: `http://localhost:5001/api`

### File Structure
```
Trailtracker/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── animal_model.pth       # Trained AI model (43MB)
├── templates/            # HTML templates (home, profile, upload, etc.)
├── static/              # CSS, JS, images, and static assets
├── uploads/             # Uploaded media files
├── migrations/          # Database migrations
├── scripts/            # Utility scripts
├── TODO.md             # Development notes and tasks
└── README.md           # This file
```

## Current Status

This is a working application with the following implemented features:
- ✅ User authentication and profile management
- ✅ AI-powered animal detection
- ✅ Interactive map with camera pin management
- ✅ Upload and processing of images/videos
- ✅ Data analytics and filtering
- ✅ Mobile-responsive design

## Known Issues

- Video processing can be slow for large files
- Some OCR text extraction may not work perfectly on all camera types
- Weather overlay requires API key configuration

## Support

For issues or questions, please open an issue in the repository. 