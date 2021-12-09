import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
import pandas as pd 
import librosa
import soundfile as sf
import numpy as np
from os import path

UPLOAD_FOLDER = 'uploads'
# Check if the upload folder exists and if not create one in the root directory
if(path.exists(UPLOAD_FOLDER) == False):
    os.mkdir(UPLOAD_FOLDER)
    print("Uploads directory created")

ALLOWED_EXTENSIONS = {'wav', 'mp3',"ogg"}

print("Extracting features..")
features_df1 = pd.read_csv("features.csv") 
print("Extracting features done..")

# Create Flask App
app = Flask(__name__)


# Limit content size
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_features(file_name):

    if file_name: 
        X, sample_rate = sf.read(file_name, dtype='float32')

    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],
                secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results',
                filename=filename))
    return render_template("index.html")

def get_numpy_array(features_df):

    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())
    # encode classification labels
    le = LabelEncoder()
    # one hot encoded labels
    yy = to_categorical(le.fit_transform(y))
    return X,yy,le

def class_label(argument):
    classes = {
        0: "Doorbell",
        1: "Rain",
        2: "Pressure-Cooker",
        3: "Baby-Cry",
        4: "Water=Overflow"
    }
    return classes.get(argument, "Unidentified Sound")

# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args['filename']
    # Compute audio signal features
    X, y, le = get_numpy_array(features_df1)
    model = load_model("trained_cnn.h5")
    prediction_feature = get_features(filename)
    prediction_feature = np.expand_dims(np.array([prediction_feature]),axis=2)
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    final_pred = class_label(predicted_class[0])

    # Delete uploaded file
    os.remove(filename)
    # Render results
    return render_template("results.html",
        filename=filename,
        predictions_to_render=final_pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))