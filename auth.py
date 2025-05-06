from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User, AdminInfo, ClientInfo

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/face-recognize/')
def default():
    return redirect(url_for('auth.login'))

@auth_bp.route('/face-recognize/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('用户名已存在!', 'danger')
            return redirect(url_for('auth.register'))
        
        # Create new user
        new_user = User(username=username, role=role)
        new_user.set_password(password)
        
        # Add additional user info
        if role == 'client':
            client_info = ClientInfo(user=new_user)
            db.session.add(client_info)
        elif role == 'admin':
            admin_info = AdminInfo(user=new_user)
            db.session.add(admin_info)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('注册成功，请登录!', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')

@auth_bp.route('/face-recognize/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('登录成功!', 'success')
            
            # Redirect based on user role
            if user.role == 'admin':
                return redirect(url_for('admin.dashboard'))
            elif user.role == 'client':
                return redirect(url_for('client.dashboard'))
        else:
            flash('用户名或密码错误!', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/face-recognize/logout')
@login_required
def logout():
    logout_user()
    flash('已成功退出登录', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/face-recognize/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin.dashboard'))
    elif current_user.role == 'client':
        return redirect(url_for('client.dashboard'))