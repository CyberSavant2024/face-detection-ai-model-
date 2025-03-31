"""
Simplified version of the app that uses basic OpenCV face detection instead of face_recognition
Use this if you have trouble installing face_recognition or dlib
"""

import os
import sys

# Add the project root directory to the Python path
# This ensures that the modules directory can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import json
import datetime
import cv2
import numpy as np
import shutil
import base64
from modules.database import Database
from flask_cors import CORS

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)  # Enable CORS for React development
db = Database('attendance_db.sqlite')

# Ensure required directories exist
os.makedirs('data/student_images', exist_ok=True)
os.makedirs('data/models', exist_ok=True)

# Simple face detector using OpenCV's Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def base64_to_image(base64_string):
    """Convert base64 string to an OpenCV image"""
    # Remove the data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def detect_faces(image):
    """Simple face detection using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

@app.route('/api/students', methods=['GET'])
def get_students():
    students = db.get_all_students()
    return jsonify(students)

@app.route('/api/register_student', methods=['POST'])
def register_student():
    name = request.form.get('name')
    student_id = db.add_student(name)
    
    # Save images
    image_count = int(request.form.get('image_count', 0))
    
    save_path = f'data/student_images/{student_id}'
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(image_count):
        image_data = request.form.get(f'image_{i}')
        if image_data:
            # Convert base64 to image and save
            image = base64_to_image(image_data)
            cv2.imwrite(f'{save_path}/{i}.jpg', image)
    
    # Handle uploaded files
    uploaded_files = request.files.getlist('uploaded_images')
    for i, file in enumerate(uploaded_files):
        if file.filename:
            file_path = f'{save_path}/{image_count + i}.jpg'
            file.save(file_path)
    
    return jsonify({'success': True, 'student_id': student_id})

@app.route('/api/train_model', methods=['POST'])
def train_model():
    # In simple mode, we don't actually train a model
    # Just return success as long as there are student images
    student_dirs = os.listdir('data/student_images')
    success = len(student_dirs) > 0
    return jsonify({'success': success})

@app.route('/api/take_attendance', methods=['POST'])
def take_attendance():
    date = request.form.get('date')
    
    # Since we can't recognize specific students in simple mode,
    # we'll just mark attendance for all students if faces are detected
    face_detected = False
    
    # Check if captured image is provided
    image_data = request.form.get('image')
    if image_data:
        image = base64_to_image(image_data)
        face_detected = detect_faces(image)
    
    # Check if uploaded image is provided
    if not face_detected and 'uploaded_image' in request.files:
        uploaded_file = request.files['uploaded_image']
        if uploaded_file.filename:
            # Save temporarily
            temp_path = 'temp_attendance.jpg'
            uploaded_file.save(temp_path)
            
            # Process the image
            image = cv2.imread(temp_path)
            if image is not None:
                face_detected = detect_faces(image)
            
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    recognized_students = []
    if face_detected:
        # Get all student IDs
        students = db.get_all_students()
        recognized_students = [str(student['id']) for student in students]
        
        # Record attendance
        for student_id in recognized_students:
            db.mark_attendance(student_id, date)
    
    return jsonify({'success': True, 'recognized': recognized_students})

@app.route('/api/attendance_data', methods=['GET'])
def get_attendance_data():
    date = request.args.get('date', datetime.date.today().isoformat())
    attendance_data = db.get_attendance_by_date(date)
    
    # Get student details for recognized students
    attendance_with_details = []
    for record in attendance_data:
        student_details = db.get_student_by_id(record['student_id'])
        combined = {**record, **student_details}
        attendance_with_details.append(combined)
    
    return jsonify(attendance_with_details)

@app.route('/api/reset_system', methods=['POST'])
def reset_system():
    try:
        # Drop and recreate database
        db.reset_database()
        
        # Remove all student images
        if os.path.exists('data/student_images'):
            shutil.rmtree('data/student_images')
            os.makedirs('data/student_images')
        
        # Remove trained models
        if os.path.exists('data/models'):
            shutil.rmtree('data/models')
            os.makedirs('data/models')
        
        return jsonify({'success': True, 'message': 'System reset successful'})
    except Exception as e:
        print(f"Error during reset: {e}")
        return jsonify({'success': False, 'message': f'Error during reset: {str(e)}'})

@app.route('/data/student_images/<path:filename>')
def student_images(filename):
    return send_from_directory('data/student_images', filename)

# Remove the capture_sequence endpoint

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    db.setup_database()  # Ensure database is set up
    app.run(debug=True, host='0.0.0.0', port=5000)
