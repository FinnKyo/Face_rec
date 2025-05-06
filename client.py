from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user

client_bp = Blueprint('client', __name__)

@client_bp.route('/face-recognize/client/dashboard')
@login_required
def dashboard():
    if current_user.role != 'client':
        flash('没有权限访问此页面!', 'danger')
        return redirect(url_for('auth.dashboard'))
    
    return redirect(url_for('admin.visitor_control'))