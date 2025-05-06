from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'client', 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    client_info = db.relationship('ClientInfo', uselist=False, back_populates='user', cascade='all, delete-orphan')
    admin_info = db.relationship('AdminInfo', uselist=False, back_populates='user', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ClientInfo(db.Model):
    __tablename__ = 'client_info'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    title = db.Column(db.String(50))  

    # Relationship
    user = db.relationship('User', back_populates='client_info')

class AdminInfo(db.Model):
    __tablename__ = 'admin_info'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    title = db.Column(db.String(50)) 

    # Relationship
    user = db.relationship('User', back_populates='admin_info')

class DoorLog(db.Model):
    __tablename__ = 'door_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    visitor_name = db.Column(db.String(100), nullable=True)
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    door_action = db.Column(db.String(20), nullable=False)  # 'open', 'close'
    notes = db.Column(db.String(200), nullable=True)  # For visitor purpose
    image_path = db.Column(db.String(100), nullable=True)
    # Relationship
    user = db.relationship('User', backref=db.backref('door_logs', lazy=True))

class AuthorizedFace(db.Model):
    """Model for storing authorized faces information"""
    __tablename__ = 'authorized_faces'
    
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(100), unique=True, nullable=False)  # Internal identifier
    display_name = db.Column(db.String(100), nullable=False)  # Displayed name
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Can be null for visitors
    is_authorized = db.Column(db.Boolean, default=True, nullable=False)
    face_count = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    last_updated = db.Column(db.DateTime, nullable=True)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('faces', lazy=True))
    
    def __repr__(self):
        return f'<AuthorizedFace {self.display_name}>'