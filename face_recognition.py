from flask import Blueprint, request, jsonify, url_for, current_app, render_template, flash, redirect
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from models import db, DoorLog, User, AuthorizedFace
import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import base64
import shutil
import json

face_bp = Blueprint('face', __name__)

# Global variables for face recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_count = 0
current_person = None

@face_bp.route('/face-recognize/admin/face_management', methods=['GET'])
@login_required
def face_management():
    print("aisudhiahs")

    if current_user.role != 'admin':
        flash('无权限操作', 'danger')
        return redirect(url_for('auth.dashboard'))

    try:
        # Get all registered faces
        faces = AuthorizedFace.query.all()
        

        return render_template('admin/face_management.html', faces=faces)
    except Exception as e:
        flash(f'错误: {str(e)}', 'danger')
        return redirect(url_for('auth.dashboard'))

@face_bp.route('/face-recognize/admin/face_learning', methods=['GET'])
@login_required
def face_learning():
    if current_user.role != 'admin' and current_user.role != 'client':
        flash('无权限操作', 'danger')
        return redirect(url_for('auth.dashboard'))

    try:
        # Get all users for the dropdown
        users = User.query.all()
        return render_template('admin/face_learning.html', users=users)
    except Exception as e:
        flash(f'错误: {str(e)}', 'danger')
        return redirect(url_for('auth.dashboard'))

@face_bp.route('/face-recognize/api/list_registered_faces', methods=['GET'])
@login_required
def list_registered_faces():
    print("jasdnansd")
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
        
    try:
        # Get all authorized faces from database
        authorized_faces = AuthorizedFace.query.all()
        print("authorized_faces > ",authorized_faces)
        
        # Prepare response
        faces_data = []
        for face in authorized_faces:
            # Get user information if associated with a user
            user_info = None
            if face.user_id:
                user = User.query.get(face.user_id)
                if user:
                    user_info = {
                        'id': user.id,
                        'username': user.username,
                        'role': user.role
                    }
            
            # Get sample image path
            dataset_folder = current_app.config['DATASET_FOLDER']
            person_folder = os.path.join(dataset_folder, face.label)
            sample_image = None
            
            if os.path.exists(person_folder):
                image_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg') and not f.startswith('face_')]
                if image_files:
                    sample_image = url_for('static', filename=f"dataset/{face.label}/{image_files[0]}")
            
            faces_data.append({
                'id': face.id,
                'label': face.label,
                'display_name': face.display_name,
                'is_authorized': face.is_authorized,
                'user': user_info,
                'sample_image': sample_image,
                'face_count': face.face_count,
                'created_at': face.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'last_updated': face.last_updated.strftime('%Y-%m-%d %H:%M:%S') if face.last_updated else None
            })
        
        print("faces_data > ",faces_data)
        return jsonify({
            'success': True, 
            'faces': faces_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/api/toggle_authorization', methods=['POST'])
@login_required
def toggle_authorization():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    try:
        data = request.get_json()
        face_id = data.get('face_id')
        
        if not face_id:
            return jsonify({'success': False, 'error': '未提供面部ID'}), 400
            
        face = AuthorizedFace.query.get(face_id)
        if not face:
            return jsonify({'success': False, 'error': '面部不存在'}), 404
            
        # Toggle authorization status
        face.is_authorized = not face.is_authorized
        face.last_updated = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f"已{'授权' if face.is_authorized else '取消授权'} {face.display_name}",
            'is_authorized': face.is_authorized
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/api/delete_face', methods=['POST'])
@login_required
def delete_face():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    try:
        data = request.get_json()
        face_id = data.get('face_id')
        
        if not face_id:
            return jsonify({'success': False, 'error': '未提供面部ID'}), 400
            
        face = AuthorizedFace.query.get(face_id)
        if not face:
            return jsonify({'success': False, 'error': '面部不存在'}), 404
        
        # Get the face label
        face_label = face.label
        
        # Delete face directory
        dataset_folder = current_app.config['DATASET_FOLDER']
        person_folder = os.path.join(dataset_folder, face_label)
        if os.path.exists(person_folder):
            shutil.rmtree(person_folder)
        
        # Delete database record
        db.session.delete(face)
        db.session.commit()
        
        # Retrain model if there are still other faces
        if AuthorizedFace.query.count() > 0:
            _train_model_internal()
        else:
            # Remove model files if no more faces
            models_folder = current_app.config['MODELS_FOLDER']
            model_file = os.path.join(models_folder, "face_recognizer.yml")
            labels_file = os.path.join(models_folder, "face_labels.pickle")
            
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(labels_file):
                os.remove(labels_file)
        
        return jsonify({
            'success': True,
            'message': f"已删除 {face.display_name} 的面部数据"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/api/rename_face', methods=['POST'])
@login_required
def rename_face():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    try:
        data = request.get_json()
        face_id = data.get('face_id')
        new_name = data.get('display_name')
        
        if not face_id or not new_name:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400
            
        face = AuthorizedFace.query.get(face_id)
        if not face:
            return jsonify({'success': False, 'error': '面部不存在'}), 404
            
        # Update display name
        face.display_name = new_name
        face.last_updated = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f"已更新显示名称为 {new_name}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/api/get_face_samples', methods=['GET'])
@login_required
def get_face_samples():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    try:
        face_id = request.args.get('face_id')
        
        if not face_id:
            return jsonify({'success': False, 'error': '未提供面部ID'}), 400
            
        face = AuthorizedFace.query.get(face_id)
        if not face:
            return jsonify({'success': False, 'error': '面部不存在'}), 404
            
        # Get all face samples
        dataset_folder = current_app.config['DATASET_FOLDER']
        person_folder = os.path.join(dataset_folder, face.label)
        
        samples = []
        if os.path.exists(person_folder):
            for file in os.listdir(person_folder):
                if file.endswith('.jpg'):
                    file_path = os.path.join(person_folder, file)
                    # Get file creation time
                    ctime = os.path.getctime(file_path)
                    # Convert to datetime
                    create_date = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
                    
                    samples.append({
                        'filename': file,
                        'url': url_for('static', filename=f"dataset/{face.label}/{file}"),
                        'create_date': create_date,
                        'is_face': 'face_' in file
                    })
                    
        # Sort by creation date
        samples.sort(key=lambda x: x['create_date'], reverse=True)
        
        return jsonify({
            'success': True,
            'face_id': face_id,
            'label': face.label,
            'display_name': face.display_name,
            'samples': samples
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/api/delete_face_sample', methods=['POST'])
@login_required
def delete_face_sample():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'error': '无权限操作'}), 403
    
    try:
        data = request.get_json()
        face_id = data.get('face_id')
        filename = data.get('filename')
        
        if not face_id or not filename:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400
            
        face = AuthorizedFace.query.get(face_id)
        if not face:
            return jsonify({'success': False, 'error': '面部不存在'}), 404
            
        # Delete the file
        dataset_folder = current_app.config['DATASET_FOLDER']
        file_path = os.path.join(dataset_folder, face.label, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Update face count if it was a face sample
            if 'face_' in filename:
                face.face_count = max(0, face.face_count - 1)
                face.last_updated = datetime.utcnow()
                db.session.commit()
                
                # Retrain model if there are still faces
                if face.face_count > 0:
                    _train_model_internal()
                    
            return jsonify({
                'success': True,
                'message': f"已删除样本 {filename}",
                'face_count': face.face_count
            })
        else:
            return jsonify({
                'success': False,
                'error': '文件不存在'
            }), 404
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/capture', methods=['POST'])
@login_required
def capture_image():
    if current_user.role != 'admin' and current_user.role != 'client':
        return jsonify({"success": False, "error": "无权限操作"}), 403
    
    global image_count, current_person
    
    # Make sure we have a current person
    if not current_person:
        return jsonify({"success": False, "error": "请先开始捕获会话"}), 400
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({"success": False, "error": "未收到图像数据"}), 400
        
        # Debug
        print(f"Received data URL length: {len(image_data)}")
        
        # Properly handle data URL format
        if 'data:image/' in image_data:
            # Split header and data
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data
            
        # Debug
        print(f"Base64 data length: {len(encoded)}")
        
        # Decode base64 data
        try:
            image_bytes = base64.b64decode(encoded)
            print(f"Decoded bytes length: {len(image_bytes)}")
            
            if len(image_bytes) == 0:
                return jsonify({"success": False, "error": "解码后的图像数据为空"}), 400
                
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            print(f"NumPy array shape: {nparr.shape}")
            
            # Decode the image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"success": False, "error": "无法解码图像数据，请检查图像格式"}), 400
                
            print(f"Decoded image shape: {frame.shape}")
            
            # Now process the image as before
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Save the image with bounding boxes drawn
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        except base64.binascii.Error as e:
            return jsonify({"success": False, "error": f"Base64解码错误: {str(e)}"}), 400

        # Create person folder if it doesn't exist
        dataset_folder = current_app.config['DATASET_FOLDER']
        person_folder = os.path.join(dataset_folder, current_person)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Save the image in the person's folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{current_person}_{timestamp}.jpg"
        filepath = os.path.join(person_folder, filename)
        
        cv2.imwrite(filepath, frame)
        
        # Also save the aligned face for better recognition
        face_count = 0
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            # Resize to a standard size (for better training)
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            # Save the face ROI
            face_filename = f"{current_person}_face_{timestamp}_{image_count}.jpg"
            face_filepath = os.path.join(person_folder, face_filename)
            cv2.imwrite(face_filepath, face_roi_resized)
            
            # Update count
            image_count += 1
            face_count += 1
        
        # Get relative path for URL
        relative_path = os.path.join('dataset', current_person, filename).replace('\\', '/')
        
        # Construct the URL correctly
        app_root = current_app.config.get('APPLICATION_ROOT', '')
        static_url = current_app.static_url_path
        
        # Construct the full URL path
        image_url = f"{static_url}/{relative_path}"
        
        # Update face count in database
        face = AuthorizedFace.query.filter_by(label=current_person).first()
        if face:
            face.face_count += face_count
            face.last_updated = datetime.utcnow()
            db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "图像捕获成功!",
            "image_path": image_url,
            "count": image_count,
            "faces_detected": len(faces)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@face_bp.route('/face-recognize/api/start_capture', methods=['POST'])
@login_required
def start_capture():
    if current_user.role != 'admin' and current_user.role != 'client':
        return jsonify({"success": False, "error": "无权限操作"}), 403
    
    global image_count, current_person
    
    data = request.get_json()
    user_id = data.get('user_id')  # Can be None for visitors
    display_name = data.get('display_name')
    is_new = data.get('is_new', True)
    face_id = data.get('face_id')  # For recapture
    use_current_user = data.get('use_current_user', False)
    
    if use_current_user:
        display_name = current_user.username
        user_id = current_user.id

    if not display_name:
        return jsonify({"success": False, "error": "请提供显示名称"}), 400
    
    # For recapture of existing face
    if not is_new and face_id:
        face = AuthorizedFace.query.get(face_id)
        if not face:
            return jsonify({"success": False, "error": "面部不存在"}), 404
            
        # Reset counter and set current person
        image_count = 0
        current_person = face.label
        
        # Create person folder if it doesn't exist
        dataset_folder = current_app.config['DATASET_FOLDER']
        person_folder = os.path.join(dataset_folder, current_person)
        
        # If folder exists, remove previous data
        if os.path.exists(person_folder):
            try:
                shutil.rmtree(person_folder)
            except Exception as e:
                return jsonify({"success": False, "error": f"无法清除旧数据: {str(e)}"}), 500
        
        # Create new folder
        os.makedirs(person_folder)
        
        # Reset face count
        face.face_count = 0
        face.last_updated = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            "success": True, 
            "message": f"开始为 {display_name} 重新捕获人脸图像",
            "face_id": face_id,
            "is_new": False
        })
    
    # For new face
    if is_new:
        # Generate a unique label/identifier for the person
        if user_id:
            user = User.query.get(user_id)
            if not user:
                return jsonify({"success": False, "error": "用户不存在"}), 404
                
            # Use username as base for label
            base_label = secure_filename(user.username.lower())
        else:
            # For visitors or non-users
            base_label = secure_filename(display_name.lower())
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        label = f"{base_label}_{timestamp}"
        
        # Reset counter and set current person
        image_count = 0
        current_person = label
        
        # Create person folder
        dataset_folder = current_app.config['DATASET_FOLDER']
        person_folder = os.path.join(dataset_folder, current_person)
        
        # Ensure folder doesn't exist
        if os.path.exists(person_folder):
            return jsonify({"success": False, "error": "此标识已存在，请使用不同的名称"}), 400
        
        # Create new folder
        os.makedirs(person_folder)
        
        # Create new authorized face record
        face = AuthorizedFace(
            label=current_person,
            display_name=display_name,
            user_id=user_id,
            is_authorized=True,  # Default to authorized
            face_count=0,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        db.session.add(face)
        db.session.commit()
        
        return jsonify({
            "success": True, 
            "message": f"开始为 {display_name} 捕获人脸图像",
            "face_id": face.id,
            "is_new": True
        })
    
    return jsonify({"success": False, "error": "无效的请求参数"}), 400

def _train_model_internal():
    """Internal function for model training"""
    try:
        # Check if OpenCV face module is available
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Prepare data
        faces = []
        labels = []
        label_ids = {}
        current_id = 0
        new_faces = []  # Track newly added faces
        
        dataset_folder = current_app.config['DATASET_FOLDER']
        models_folder = current_app.config['MODELS_FOLDER']
        model_file = os.path.join(models_folder, "face_recognizer.yml")
        labels_file = os.path.join(models_folder, "face_labels.pickle")
        last_trained_file = os.path.join(models_folder, "last_trained.txt")
        
        # Ensure models folder exists
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        
        # Get last training timestamp
        last_trained_time = None
        if os.path.exists(last_trained_file):
            with open(last_trained_file, 'r') as f:
                try:
                    last_trained_time = datetime.fromisoformat(f.read().strip())
                except (ValueError, TypeError):
                    last_trained_time = None
        
        # Get all authorized faces
        authorized_faces = AuthorizedFace.query.filter_by(is_authorized=True).all()
        
        print("authorized_faces", authorized_faces)
        # Map labels to IDs
        for face in authorized_faces:
            label_ids[face.label] = current_id
            
            # Check if this face is new since last training
            if last_trained_time and face.created_at and face.created_at > last_trained_time:
                new_faces.append({
                    "id": face.id,
                    "label": face.label,
                    "display_name": face.display_name,
                    "face_count": face.face_count,
                    "created_at": face.created_at.isoformat()
                })
            
            current_id += 1
        
        # Process all face images in the dataset folder for authorized faces
        for face in authorized_faces:
            person_folder = os.path.join(dataset_folder, face.label)
            if os.path.exists(person_folder):
                for file in os.listdir(person_folder):
                    if file.endswith(".jpg") and "face_" in file:
                        path = os.path.join(person_folder, file)
                        face_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:
                            # Add face and label to training data
                            faces.append(face_img)
                            labels.append(label_ids[face.label])
        
        if not faces:
            return False, "没有找到足够的面部图像进行训练", []
        
        # Save the label mapping
        with open(labels_file, 'wb') as f:
            pickle.dump(label_ids, f)
        
        # Train the recognizer
        recognizer.train(faces, np.array(labels))
        
        # Save the trained model
        recognizer.save(model_file)
        
        # Update last trained timestamp
        with open(last_trained_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        return True, "模型训练成功", new_faces
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e), []

@face_bp.route('/face-recognize/api/train_model', methods=['POST'])
@login_required
def train_model():
    if current_user.role != 'admin' and current_user.role != 'client':
        return jsonify({"success": False, "error": "无权限操作"}), 403
    
    try:
        success, message, new_faces = _train_model_internal()
        
        if not success:
            return jsonify({
                "success": False,
                "error": message
            }), 500
        
        # Count total people and faces
        total_people = AuthorizedFace.query.filter_by(is_authorized=True).count()
        total_faces = 0
        
        for face in AuthorizedFace.query.filter_by(is_authorized=True).all():
            total_faces += face.face_count
        
        return jsonify({
            "success": True,
            "message": "训练完成!",
            "total_faces": total_faces,
            "total_people": total_people,
            "new_faces": new_faces  # Include new faces in the response
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    
@face_bp.route('/face-recognize/api/verify_access_live', methods=['POST'])
def verify_access_live():
    """
    API endpoint for live face verification that can be called from
    any client device (including non-admin users)
    """
    try:
        # Check if model exists
        models_folder = current_app.config['MODELS_FOLDER']
        model_file = os.path.join(models_folder, "face_recognizer.yml")
        labels_file = os.path.join(models_folder, "face_labels.pickle")
        
        if not os.path.exists(model_file) or not os.path.exists(labels_file):
            return jsonify({"success": False, "error": "没有训练模型"})
         
        # Load the face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file)
        
        # Load the label mapping
        with open(labels_file, 'rb') as f:
            label_ids = pickle.load(f)
            # Invert the dictionary to get id -> name mapping
            id_to_label = {v: k for k, v in label_ids.items()}
        
        # Get image data
        if request.content_type and 'application/json' in request.content_type:
            # Handle base64 encoded image
            data = request.get_json()
            image_data = data.get('image_data')
            
            if not image_data:
                return jsonify({"success": False, "error": "没有提供图像数据"}), 400
                
            # Handle data URL format
            if 'data:image/' in image_data:
                header, encoded = image_data.split(',', 1)
            else:
                encoded = image_data
                
            # Decode base64 data
            try:
                image_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"success": False, "error": f"解码图像失败: {str(e)}"}), 400
                
        elif 'image' in request.files:
            # Handle file upload
            image_file = request.files['image']
            image_stream = image_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"success": False, "error": "没有提供图像"}), 400
            
        if frame is None:
            return jsonify({"success": False, "error": "无法解码图像"}), 400
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Initialize variables
        authorized_access = False
        detected_faces = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            
            # Predict using the recognizer
            label, confidence = recognizer.predict(face_roi_resized)
            
            # Determine if it's a recognized person 
            recognized = False
            person_label = None
            person_name = "Unknown"
            is_authorized = False
            
            if confidence < 70:  # Threshold can be adjusted
                person_label = id_to_label.get(label)
                
                if person_label:
                    # Look up in database to get the person's display name and authorization status
                    face_record = AuthorizedFace.query.filter_by(label=person_label).first()
                    if face_record:
                        recognized = True
                        person_name = face_record.display_name
                        is_authorized = face_record.is_authorized
                        
                        if is_authorized:
                            authorized_access = True
            
            # Determine color based on recognition and authorization
            if recognized and is_authorized:
                color = (0, 255, 0)  # Green for authorized
            elif recognized and not is_authorized:
                color = (0, 255, 255)  # Yellow for recognized but not authorized
            else:
                color = (0, 0, 255)  # Red for unknown
                
            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{person_name} ({confidence:.1f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
            # Add to detected faces list
            detected_faces.append({
                'name': person_name,
                'confidence': float(confidence),
                'recognized': recognized,
                'is_authorized': is_authorized if recognized else False,
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            })
        
        # Only save image and create log entry if there are faces detected
        if detected_faces:
            # Create a timestamped image for logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"access_attempt_{timestamp}.jpg"
            log_folder = current_app.config['ACCESS_LOGS_FOLDER']
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            filepath = os.path.join(log_folder, filename)
            
            cv2.imwrite(filepath, frame)
            
            # Create a door log entry
            action = "PASS" if authorized_access else "REJECT"
            
            # Initialize user and visitor variables
            user_id = None
            visitor_name = None
            
            # Process detected faces to separate users and visitors
            for face in detected_faces:
                if face['recognized'] and face['is_authorized']:
                    # This is an authorized user
                    # Try to find user in the database
                    user = User.query.filter_by(username=face['name']).first()
                    if user:
                        user_id = user.id
                elif face['name'] != "Unknown":
                    # This is a visitor (not an authorized user)
                    if visitor_name is None:
                        visitor_name = face['name']
                    else:
                        visitor_name += f", {face['name']}"
            
            # If no user was found but we're logged in, use the current user
            if user_id is None and current_user.is_authenticated:
                user_id = current_user.id
                
            print("vjvjhvjvhjv",user_id)
            door_log = DoorLog(
                user_id=user_id,
                visitor_name=visitor_name,  # Only include visitor names, not authorized users
                door_action=action,
                entry_time=datetime.utcnow(),
                notes=f"Live recognition: {'Authorized' if authorized_access else 'Unauthorized'}",
                image_path=f"access_logs/{filename}"
            )
            db.session.add(door_log)
            db.session.commit()
            
            image_path = url_for('static', filename=f"access_logs/{filename}")
        else:
            image_path = None
        
        print(detected_faces)
        return jsonify({
            "success": True,
            "authorized": authorized_access,
            "detected_faces": detected_faces,
            "image_path": image_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@face_bp.route('/face-recognize/api/access_logs', methods=['GET'])
@login_required
def get_access_logs():
    """Retrieve access logs with optional filtering"""
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        visitor_name = request.args.get('visitor_name')
        door_action = request.args.get('door_action')
        
        # Build query
        query = DoorLog.query
        
        # Apply filters
        if start_date:
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(DoorLog.entry_time >= start_datetime)
            
        if end_date:
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
            # Add one day to include the entire end date
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            query = query.filter(DoorLog.entry_time <= end_datetime)
            
        if visitor_name:
            query = query.filter(DoorLog.visitor_name.ilike(f'%{visitor_name}%'))
            
        if door_action:
            query = query.filter(DoorLog.door_action == door_action)
            
        # Order by most recent first
        query = query.order_by(DoorLog.entry_time.desc())
        
        # Paginate
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        logs = pagination.items
        
        # Prepare response
        logs_data = []
        for log in logs:
            # Get user info
            user_info = None
            if log.user_user_id:
                user = User.query.get(log.user_user_id)
                if user:
                    user_info = {
                        'id': user.id,
                        'username': user.username,
                    }
                    
            # Build log entry data
            log_data = {
                'id': log.id,
                'visitor_name': log.visitor_name,
                'door_action': log.door_action,
                'entry_time': log.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'notes': log.notes,
                'user': user_info,
                'image_path': url_for('static', filename=log.image_path) if log.image_path else None
            }
            
            logs_data.append(log_data)
            
        return jsonify({
            'success': True,
            'logs': logs_data,
            'total': pagination.total,
            'pages': pagination.pages,
            'current_page': page
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/admin/access_logs', methods=['GET'])
@login_required
def access_logs():
    if current_user.role != 'admin':
        flash('无权限操作', 'danger')
        return redirect(url_for('auth.dashboard'))
        
    return render_template('admin/access_logs.html')

@face_bp.route('/face-recognize/api/delete_access_log', methods=['POST'])
@login_required
def delete_access_log():
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        data = request.get_json()
        log_id = data.get('log_id')
        
        if not log_id:
            return jsonify({"success": False, "error": "未提供日志ID"}), 400
            
        log = DoorLog.query.get(log_id)
        if not log:
            return jsonify({"success": False, "error": "日志不存在"}), 404
            
        # Delete image if it exists
        if log.image_path:
            image_path = os.path.join(current_app.static_folder, log.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
                
        # Delete log entry
        db.session.delete(log)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "日志已删除"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@face_bp.route('/face-recognize/api/get_system_status', methods=['GET'])
@login_required
def get_system_status():
    """Get system status information about the face recognition system"""
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        # Check if model exists
        models_folder = current_app.config['MODELS_FOLDER']
        model_file = os.path.join(models_folder, "face_recognizer.yml")
        labels_file = os.path.join(models_folder, "face_labels.pickle")
        
        model_exists = os.path.exists(model_file) and os.path.exists(labels_file)
        
        # Get counts
        total_faces = AuthorizedFace.query.count()
        authorized_faces = AuthorizedFace.query.filter_by(is_authorized=True).count()
        total_access_logs = DoorLog.query.count()
        
        # Get recent logs
        recent_logs = DoorLog.query.order_by(DoorLog.entry_time.desc()).limit(5).all()
        recent_logs_data = []
        
        for log in recent_logs:
            log_data = {
                'id': log.id,
                'visitor_name': log.visitor_name,
                'door_action': log.door_action,
                'entry_time': log.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'image_path': url_for('static', filename=log.image_path) if log.image_path else None
            }
            recent_logs_data.append(log_data)
            
        # Get model information
        model_info = {}
        if model_exists:
            try:
                # Get model file size and date
                model_size = os.path.getsize(model_file)
                model_modified = datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
                
                # Load label mapping to get number of people in model
                with open(labels_file, 'rb') as f:
                    label_ids = pickle.load(f)
                    
                model_info = {
                    'size': model_size,
                    'modified': model_modified,
                    'people_count': len(label_ids)
                }
            except Exception as e:
                model_info = {
                    'error': str(e)
                }
                
        return jsonify({
            'success': True,
            'model_exists': model_exists,
            'total_faces': total_faces,
            'authorized_faces': authorized_faces,
            'total_access_logs': total_access_logs,
            'recent_logs': recent_logs_data,
            'model_info': model_info
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@face_bp.route('/face-recognize/api/compare_faces', methods=['POST'])
@login_required
def compare_faces():
    """Compare two face images to determine if they are the same person"""
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        # Check for face_recognition library
        try:
            import face_recognition
        except ImportError:
            return jsonify({
                "success": False,
                "error": "未安装face_recognition库。请安装face_recognition: pip install face_recognition"
            }), 500
            
        # Get images from request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({"success": False, "error": "请提供两张人脸图像"}), 400
            
        image1_file = request.files['image1']
        image2_file = request.files['image2']
        
        # Read images
        image1_stream = image1_file.read()
        image2_stream = image2_file.read()
        
        # Convert to numpy arrays
        image1_array = np.frombuffer(image1_stream, np.uint8)
        image2_array = np.frombuffer(image2_stream, np.uint8)
        
        # Decode images
        image1 = cv2.imdecode(image1_array, cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(image2_array, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            return jsonify({"success": False, "error": "无法解码图像"}), 400
            
        # Convert from BGR to RGB (face_recognition expects RGB)
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        image1_face_locations = face_recognition.face_locations(image1_rgb)
        image2_face_locations = face_recognition.face_locations(image2_rgb)
        
        if not image1_face_locations:
            return jsonify({"success": False, "error": "在第一张图像中未检测到人脸"}), 400
            
        if not image2_face_locations:
            return jsonify({"success": False, "error": "在第二张图像中未检测到人脸"}), 400
            
        # Get face encodings
        image1_encodings = face_recognition.face_encodings(image1_rgb, image1_face_locations)
        image2_encodings = face_recognition.face_encodings(image2_rgb, image2_face_locations)
        
        if not image1_encodings or not image2_encodings:
            return jsonify({"success": False, "error": "无法提取人脸特征"}), 400
            
        # Compare faces
        # Compare each face in image1 with each face in image2
        match_results = []
        
        for i, encoding1 in enumerate(image1_encodings):
            for j, encoding2 in enumerate(image2_encodings):
                # Calculate face distance
                face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
                # Convert to similarity (lower distance = higher similarity)
                similarity = 1 - face_distance
                
                # Determine if match (threshold can be adjusted)
                is_same_person = face_distance < 0.6
                
                match_results.append({
                    'face1_index': i,
                    'face2_index': j,
                    'similarity': float(similarity),
                    'is_match': bool(is_same_person)
                })
        
        # Find best match
        if match_results:
            best_match = max(match_results, key=lambda x: x['similarity'])
        else:
            best_match = None
            
        return jsonify({
            "success": True,
            "matches": match_results,
            "best_match": best_match,
            "face1_count": len(image1_face_locations),
            "face2_count": len(image2_face_locations)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@face_bp.route('/face-recognize/api/bulk_authorization', methods=['POST'])
@login_required
def bulk_authorization():
    """Bulk authorize or deauthorize multiple faces"""
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        data = request.get_json()
        face_ids = data.get('face_ids', [])
        authorize = data.get('authorize', True)
        
        if not face_ids:
            return jsonify({"success": False, "error": "未提供面部ID列表"}), 400
            
        # Update authorization status for all faces
        for face_id in face_ids:
            face = AuthorizedFace.query.get(face_id)
            if face:
                face.is_authorized = authorize
                face.last_updated = datetime.utcnow()
                
        db.session.commit()
        
        # Retrain model
        _train_model_internal()
        
        return jsonify({
            "success": True,
            "message": f"已{'授权' if authorize else '取消授权'}选中的 {len(face_ids)} 个面部"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@face_bp.route('/face-recognize/api/bulk_delete', methods=['POST'])
@login_required
def bulk_delete():
    """Bulk delete multiple faces"""
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        data = request.get_json()
        face_ids = data.get('face_ids', [])
        
        if not face_ids:
            return jsonify({"success": False, "error": "未提供面部ID列表"}), 400
            
        deleted_count = 0
        dataset_folder = current_app.config['DATASET_FOLDER']
        
        # Delete all specified faces
        for face_id in face_ids:
            face = AuthorizedFace.query.get(face_id)
            if face:
                # Delete face directory if it exists
                person_folder = os.path.join(dataset_folder, face.label)
                if os.path.exists(person_folder):
                    shutil.rmtree(person_folder)
                    
                # Delete database record
                db.session.delete(face)
                deleted_count += 1
                
        db.session.commit()
        
        # Retrain model if there are still faces
        if AuthorizedFace.query.count() > 0:
            _train_model_internal()
        else:
            # Remove model files if no more faces
            models_folder = current_app.config['MODELS_FOLDER']
            model_file = os.path.join(models_folder, "face_recognizer.yml")
            labels_file = os.path.join(models_folder, "face_labels.pickle")
            
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(labels_file):
                os.remove(labels_file)
        
        return jsonify({
            "success": True,
            "message": f"已删除 {deleted_count} 个面部数据"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@face_bp.route('/face-recognize/api/face_search', methods=['POST'])
@login_required
def face_search():
    """Search for a face in the database using an uploaded image"""
    if current_user.role != 'admin':
        return jsonify({"success": False, "error": "无权限操作"}), 403
        
    try:
        # Check if model exists
        models_folder = current_app.config['MODELS_FOLDER']
        model_file = os.path.join(models_folder, "face_recognizer.yml")
        labels_file = os.path.join(models_folder, "face_labels.pickle")
        
        if not os.path.exists(model_file) or not os.path.exists(labels_file):
            return jsonify({"success": False, "error": "没有训练模型"}), 400
            
        # Load the face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file)
        
        # Load the label mapping
        with open(labels_file, 'rb') as f:
            label_ids = pickle.load(f)
            # Invert the dictionary to get id -> name mapping
            id_to_label = {v: k for k, v in label_ids.items()}
            
        # Get image from form data
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "没有提供图像"}), 400
            
        image_file = request.files['image']
        
        # Read the image
        image_stream = image_file.read()
        nparr = np.frombuffer(image_stream, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "无法解码图像"}), 400
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        results = []
        
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (200, 200))
            
            # Draw rectangle on the image
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Predict using the recognizer
            label, confidence = recognizer.predict(face_roi_resized)
            
            # Get potential matches
            matches = []
            
            if confidence < 100:  # Consider all possible matches
                person_label = id_to_label.get(label)
                
                if person_label:
                    # Find the face in database
                    face_record = AuthorizedFace.query.filter_by(label=person_label).first()
                    if face_record:
                        # Get user info if available
                        user_info = None
                        if face_record.user_id:
                            user = User.query.get(face_record.user_id)
                            if user:
                                user_info = {
                                    'id': user.id,
                                    'username': user.username,
                                }
                                
                        # Get a sample image
                        dataset_folder = current_app.config['DATASET_FOLDER']
                        person_folder = os.path.join(dataset_folder, face_record.label)
                        sample_image = None
                        
                        if os.path.exists(person_folder):
                            image_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg') and not f.startswith('face_')]
                            if image_files:
                                sample_image = url_for('static', filename=f"dataset/{face_record.label}/{image_files[0]}")
                                
                        # Add to matches
                        matches.append({
                            'id': face_record.id,
                            'label': face_record.label,
                            'display_name': face_record.display_name,
                            'is_authorized': face_record.is_authorized,
                            'confidence': float(confidence),
                            'similarity': max(0, 100 - confidence),
                            'user': user_info,
                            'sample_image': sample_image
                        })
                        
            # Find other potential matches
            # For each face in the database, compute similarity
            for face_record in AuthorizedFace.query.all():
                # Skip if already in matches
                if matches and matches[0]['id'] == face_record.id:
                    continue
                    
                # Get sample face images for comparison
                dataset_folder = current_app.config['DATASET_FOLDER']
                person_folder = os.path.join(dataset_folder, face_record.label)
                
                if os.path.exists(person_folder):
                    # Find a face image
                    face_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg') and f.startswith('face_')]
                    if face_files:
                        # Load first face image
                        face_path = os.path.join(person_folder, face_files[0])
                        comparison_face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                        
                        if comparison_face is not None:
                            # Resize to match
                            comparison_face_resized = cv2.resize(comparison_face, (200, 200))
                            
                            # Compare using various metrics
                            # 1. Mean Squared Error
                            mse = np.sum((face_roi_resized.astype("float") - comparison_face_resized.astype("float")) ** 2)
                            mse /= float(face_roi_resized.shape[0] * face_roi_resized.shape[1])
                            
                            # Convert to similarity score (100 = exact match, 0 = completely different)
                            mse_similarity = max(0, 100 - min(100, mse / 100))
                            
                            # Only include if similarity is above threshold
                            if mse_similarity > 40:  # Threshold can be adjusted
                                # Get user info if available
                                user_info = None
                                if face_record.user_id:
                                    user = User.query.get(face_record.user_id)
                                    if user:
                                        user_info = {
                                            'id': user.id,
                                            'username': user.username,
                                        }
                                        
                                # Get a sample image
                                sample_image = None
                                image_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg') and not f.startswith('face_')]
                                if image_files:
                                    sample_image = url_for('static', filename=f"dataset/{face_record.label}/{image_files[0]}")
                                    
                                # Add to matches
                                matches.append({
                                    'id': face_record.id,
                                    'label': face_record.label,
                                    'display_name': face_record.display_name,
                                    'is_authorized': face_record.is_authorized,
                                    'confidence': float(100 - mse_similarity),
                                    'similarity': float(mse_similarity),
                                    'user': user_info,
                                    'sample_image': sample_image
                                })
            
            # Sort matches by similarity (highest first)
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Add face result
            results.append({
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'matches': matches[:5]  # Top 5 matches
            })
        
        # Save the image with face rectangles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_search_{timestamp}.jpg"
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        
        cv2.imwrite(filepath, frame)
        
        image_url = url_for('static', filename=f"uploads/{filename}")
        
        return jsonify({
            "success": True,
            "image_path": image_url,
            "faces_detected": len(faces),
            "results": results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500