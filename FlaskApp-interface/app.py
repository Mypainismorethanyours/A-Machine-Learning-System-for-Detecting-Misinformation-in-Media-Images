#interface for handling user image upload and displaying the results of the model inference on that image
#clc
from flask import Flask, render_template, request, jsonify
import requests
import os
import uuid
from werkzeug.utils import secure_filename
import time
import logging
 
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'development-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
API_URL = os.environ.get('API_URL', 'http://fastapi-inference:8080')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Short ID for tracking requests
    
    logger.info(f"[{request_id}] New prediction request started")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    if file and allowed_file(file.filename):
        # Time the file save operation
        save_start = time.time()
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        save_time = time.time() - save_start
        
        # Log file info
        file_size = os.path.getsize(filepath) / 1024  # Size in KB
        logger.info(f"[{request_id}] File saved: {filename}, Size: {file_size:.2f}KB, Save time: {save_time:.3f}s")
        
        # Call FastAPI service
        try:
            api_start = time.time()
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{API_URL}/predict/file", files=files)
            api_time = time.time() - api_start
            
            if response.status_code == 200:
                prediction_data = response.json()
                total_time = time.time() - start_time
                
                # Log performance metrics
                model_time = prediction_data.get('processing_time', 0)
                label = prediction_data.get('label', 'unknown')
                
                logger.info(f"[{request_id}] Prediction complete - "
                          f"Label: {label}, "
                          f"Upload: {save_time:.3f}s, "
                          f"API: {api_time:.3f}s, "
                          f"Model: {model_time:.3f}s, "
                          f"Total: {total_time:.3f}s")
                
                return jsonify({
                    'success': True,
                    'label': label,
                    'reasoning': prediction_data.get('reasoning', ''),
                })
            else:
                logger.error(f"[{request_id}] API error: {response.status_code}")
                return jsonify({'error': 'Error getting prediction'})
                
        except Exception as e:
            logger.error(f"[{request_id}] Exception: {str(e)}")
            return jsonify({'error': str(e)})
    else:
        logger.warning(f"[{request_id}] Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

    # Add a test comment
echo "# Test comment" >> FASTAPI-Inference/app.py