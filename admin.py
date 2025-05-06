from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from models import db, User, DoorLog
from werkzeug.utils import secure_filename
from datetime import datetime
import os

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/face-recognize/admin/dashboard')
@login_required
def dashboard():
    if current_user.role != 'admin':
        flash('无权限访问!', 'danger')
        return redirect(url_for('auth.dashboard'))
    
    users = User.query.all()  
    return render_template('admin/dashboard.html', users=users)

@admin_bp.route('/face-recognize/admin/visitor_control')
@login_required
def visitor_control():
    if current_user.role != 'admin' and current_user.role != 'client':
        flash('没有权限访问此页面!', 'danger')
        return redirect(url_for('auth.dashboard'))
    
    door_logs = DoorLog.query.order_by(DoorLog.entry_time.desc()).limit(100).all()
    return render_template('admin/visitor_control.html', door_logs=door_logs)

@admin_bp.route('/face-recognize/api/door_control', methods=['POST'])
@login_required
def door_control():
    # Check if user is admin
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    # Get data from request
    data = request.get_json()
    action = data.get('action')
    
    if action not in ['open', 'close']:
        return jsonify({'success': False, 'error': '无效操作'}), 400
    
    try:
        # Create a door log entry for the action
        door_log = DoorLog(
            user_id=current_user.id,
            door_action=action,
            entry_time=datetime.utcnow()
        )
        db.session.add(door_log)
        db.session.commit()
        
        # Here add any hardware control code
        # This is a placeholder for actual door control logic
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@admin_bp.route('/face-recognize/api/add_visitor_and_open_door', methods=['POST'])
@login_required
def add_visitor_and_open_door():
    # Check if user is admin
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    try:
        # Get visitor data
        visitor_name = request.form.get('name')
        visitor_purpose = request.form.get('purpose', '')
        
        if not visitor_name:
            return jsonify({'success': False, 'error': '访客姓名不能为空'}), 400
        
        # Handle uploaded image
        if 'image' in request.files:
            visitor_image = request.files['image']
            if visitor_image and allowed_file(visitor_image.filename):
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"visitor_{visitor_name}_{timestamp}.jpg"
                
                image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'visitors', filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                visitor_image.save(image_path)
        
        # Create a door log entry for the visitor with the open action
        door_log = DoorLog(
            visitor_name=visitor_name,
            door_action='open',
            entry_time=datetime.utcnow(),
            notes=visitor_purpose
        )
        db.session.add(door_log)
        db.session.commit()
        
        # Here add any hardware control code to actually open the door
        # This is a placeholder for actual door control logic
        
        return jsonify({'success': True, 'log_id': door_log.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
'''
@admin_bp.route('/face-recognize/api/door_logs', methods=['GET'])
@login_required
def get_door_logs():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403

    try:
        door_logs = DoorLog.query.order_by(DoorLog.entry_time.desc()).limit(100).all()
        logs_data = []

        for log in door_logs:
            logs_data.append({
                'id': log.id,
                'user': log.user.username if log.user else None,
                'visitor_name': log.visitor_name,
                'door_action': log.door_action,
                'entry_time': log.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'notes': log.notes
            })

        return jsonify({'success': True, 'door_logs': logs_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
'''
@admin_bp.route('/face-recognize/admin/door_logs', methods=['GET'])
@login_required
def door_logs():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403

    try:
        door_logs = DoorLog.query.order_by(DoorLog.entry_time.desc()).limit(100).all()
        return render_template('admin/door_log.html', door_logs=door_logs)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@admin_bp.route('/face-recognize/admin/delete_user/<int:user_id>', methods=['GET'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('无权限!', 'danger')
        return redirect(url_for('admin.dashboard'))
    
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('用户已删除!', 'success')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/face-recognize/admin/edit_user', methods=['POST'])
@login_required
def edit_user():
    if not request.json:
        return jsonify({"success": False, "error": "无效请求"}), 400

    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
    
    user_id = request.json.get('user_id')
    username = request.json.get('username')
    role = request.json.get('role')

    user = User.query.get(user_id)
    if not user:
        return jsonify({"success": False, "error": "用户不存在"}), 404
    
    user.username = username
    user.role = role
    db.session.commit()

    return jsonify({"success": True})

# Helper function
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS