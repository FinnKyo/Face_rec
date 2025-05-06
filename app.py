from flask import Flask, redirect, url_for
from flask_socketio import SocketIO
from flask_login import LoginManager
from config import Config
from models import db, User
from utils import create_directories

# Import blueprints
from auth import auth_bp
from admin import admin_bp
from client import client_bp
from face_recognition import face_bp

def create_app():
    # Initialize Flask application
    app = Flask(__name__, static_url_path='/face-recognize/static')
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @app.route('/')
    def default():
        return redirect(url_for('auth.login'))
    
    # Setup login manager
    login_manager = LoginManager(app)
    login_manager.login_view = 'auth.login'
    
 
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(client_bp)
    app.register_blueprint(face_bp)
    
    # Create required directories
    Config.init_app(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app, socketio

app, socketio = create_app()

if __name__ == '__main__':
    socketio.run(app, host='localhost', port=5002, debug=True, allow_unsafe_werkzeug=True)

