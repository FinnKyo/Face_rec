from flask import render_template
import os

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_directories(app):
    """Ensure all required directories exist"""
    required_dirs = [
        app.config['UPLOAD_FOLDER'],
        os.path.join(app.config['UPLOAD_FOLDER'], 'visitors'),
        app.config['DATASET_FOLDER'],
        app.config['MODELS_FOLDER'],
        app.config['ACCESS_LOGS_FOLDER']
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def handle_error(e, status_code=500):
    """Handle errors consistently across the application"""
    return render_template('error.html', error=str(e), code=status_code), status_code