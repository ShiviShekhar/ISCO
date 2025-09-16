from flask import Flask, jsonify, request, send_from_directory
import os
import pandas as pd
from werkzeug.utils import secure_filename

# Import the main functions from our scripts
from feature_engineering import create_feature_dataset
from model_training import train_model
from optimization import run_optimization

# --- Define Absolute Paths ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_CURRENT_DIR, '..')
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_PUBLIC_DIR = os.path.join(_ROOT_DIR, 'public')

# Initialize the Flask application
# For Netlify, the 'public' folder is served automatically.
# For local dev, we configure it as the static folder.
app = Flask(__name__, static_folder=_PUBLIC_DIR, static_url_path='')

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = _DATA_DIR

@app.route('/', methods=['GET'])
def index():
    """Render the main UI page from the public directory."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle uploading of seller and account CSV files."""
    if 'sellers_file' not in request.files or 'accounts_file' not in request.files:
        return jsonify({"error": "Missing seller or account file."}), 400

    sellers_file = request.files['sellers_file']
    accounts_file = request.files['accounts_file']

    if sellers_file.filename == '' or accounts_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        sellers_filename = secure_filename('sellers.csv')
        accounts_filename = secure_filename('accounts.csv')
        
        sellers_file.save(os.path.join(app.config['UPLOAD_FOLDER'], sellers_filename))
        accounts_file.save(os.path.join(app.config['UPLOAD_FOLDER'], accounts_filename))
        
        return jsonify({"message": "Files uploaded successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save files: {str(e)}"}), 500

@app.route('/run-optimization', methods=['POST'])
def run_optimization_endpoint():
    """API endpoint to trigger the full pipeline and return the plan."""
    try:
        create_feature_dataset()
        train_model()
        results = run_optimization()

        if results['status'] in ['OPTIMAL', 'FEASIBLE']:
            assignments_df = pd.DataFrame(results['assignments'])
            results_path = os.path.join(app.config['UPLOAD_FOLDER'], 'assignment_plan.csv')
            assignments_df.to_csv(results_path, index=False)
            
            return jsonify(results), 200
        else:
            error_message = results.get('message', 'Optimization failed to find a solution.')
            return jsonify({"error": error_message, "status": results['status']}), 500

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/download-results', methods=['GET'])
def download_results():
    """Provide the optimization results file for download."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'assignment_plan.csv', as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Results file not found. Please run the optimization first."}), 404

# This block is for local development only and will not run on Netlify
if __name__ == '__main__':
    app.run(debug=True)
