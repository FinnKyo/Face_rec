import os

class Config:
    SECRET_KEY = 'your-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///face-recognition.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    APPLICATION_ROOT = '/face-recognize'
    
    # Face Recognition Settings
    DATASET_FOLDER = "static/dataset"
    MODELS_FOLDER = "static/models"
    ACCESS_LOGS_FOLDER = "static/access_logs"
    
    # Make sure required directories exist
    @staticmethod
    def init_app(app):
        for folder in [Config.UPLOAD_FOLDER, Config.DATASET_FOLDER, 
                       Config.MODELS_FOLDER, Config.ACCESS_LOGS_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                