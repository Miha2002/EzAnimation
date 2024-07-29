# Partea de Web Dev
from flask import Flask, Response, render_template, request, redirect, url_for, send_file, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
from utils import *
import base64
import tempfile


# Configurarea Flask
def create_app():
    app = Flask(__name__)
    app.secret_key = 'AnaAreMere'
    return app


app = create_app()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4'}


def image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS


def video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    input = db.Column(db.LargeBinary)
    output = db.Column(db.LargeBinary)
    model_option = db.Column(db.String(50))
    background_option = db.Column(db.String(50))
    upload_background = db.Column(db.LargeBinary)
    can_delete = db.Column(db.Integer)


def clean_db():
    # Schimba ultimul rand folosit din db
    upload_id = session.get('upload_id')
    data = Upload.query.get(upload_id)
    if data is not None:
        data.can_delete = 1
    # Sterge datele folosite din baza de date
    Upload.query.filter_by(can_delete=1).delete()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# Favicon
@app.route('/favicon.ico')
def favicon():
    return url_for('static', filename='static/styles/img/favicon.ico')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Curata baza de date
        clean_db()

        if 'file' not in request.files or request.files['file'].filename == '':
            flash('No image/video uploaded', 'error')
            return redirect(request.url)

        file = request.files['file']

        model_option = request.form['model_option']
        background_option = request.form['background_option']

        background = request.files['background']

        if image_file(background.filename):
            upload = Upload(filename=file.filename,
                            input=file.read(),
                            output=None,
                            model_option=model_option,
                            background_option=background_option,
                            upload_background=background.read(),
                            can_delete=0
                            )
        else:
            upload = Upload(filename=file.filename,
                            input=file.read(),
                            output=None,
                            model_option=model_option,
                            background_option=background_option,
                            upload_background=None,
                            can_delete=0
                            )

        db.session.add(upload)
        db.session.commit()
        session['upload_id'] = upload.id
        return redirect(url_for('processing'))
    return render_template('upload.html')


@app.route('/processing')
def processing():
    # Sterge datele folosite din baza de date
    Upload.query.filter_by(can_delete=1).delete()
    db.session.commit()
    return render_template('processing.html')


@app.route('/process')
def process():
    upload_id = session.get('upload_id')
    upload = Upload.query.get(upload_id)

    if upload and image_file(upload.filename):
        # Verificam daca e imagine sau video
        data = upload.input
        # binary -> base64
        image_base64 = base64.b64encode(data).decode('utf-8')
        image_data = base64.b64decode(image_base64)

        # base64 -> numpy array pentru a folosi OpenCV
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv.imdecode(np_arr, cv.IMREAD_COLOR)

        # Procesare imagine / frame
        if upload.upload_background is None or not upload.upload_background:
            result_img = animate_frame(image, model=upload.model_option, bg=upload.background_option)
        else:
            # binary -> base64
            bg_base64 = base64.b64encode(upload.upload_background).decode('utf-8')
            bg_data = base64.b64decode(bg_base64)
            # base64 -> numpy array pentru a folosi OpenCV
            bg_np_arr = np.frombuffer(bg_data, np.uint8)
            bg = cv.imdecode(bg_np_arr, cv.IMREAD_COLOR)

            result_img = animate_frame(image, model=upload.model_option, bg=bg)

        # Facem encode-ing inapoi in bytes pentru a salva poza in baza de date (blob)
        upload.output = cv.imencode('.jpg', result_img)[1].tobytes()
        db.session.commit()

    if upload and video_file(upload.filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(upload.input)
            temp_video_file.flush()

            # Procesare video
            if upload.upload_background is None or not upload.upload_background:
                result_vid = animate_video(temp_video_file.name, model=upload.model_option, bg=upload.background_option)
            else:
                # binary -> base64
                bg_base64 = base64.b64encode(upload.upload_background).decode('utf-8')
                bg_data = base64.b64decode(bg_base64)
                # base64 -> numpy array pentru a folosi OpenCV
                bg_np_arr = np.frombuffer(bg_data, np.uint8)
                bg = cv.imdecode(bg_np_arr, cv.IMREAD_COLOR)
                result_vid = animate_video(temp_video_file.name, model=upload.model_option, bg=bg)

            upload.output = result_vid
        db.session.commit()

    return redirect(url_for('download'))


@app.route('/download', methods=['GET', 'POST'])
def download():
    upload_id = session.get('upload_id')
    upload = Upload.query.get(upload_id)
    if request.method == 'POST':
        upload.can_delete = 1
        db.session.commit()
        # descarcare fisier
        return send_file(BytesIO(upload.output), download_name=upload.filename, as_attachment=True)
    else:
        return render_template('download.html', upload_id=upload_id)


# Webcam cu procesare dupa inregistrare video --------------------------------------------------------------------------

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


is_recording = False
out = None
temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)


def gen_frames():
    camera = cv.VideoCapture(0)
    global is_recording, out
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            if is_recording:
                out.write(frame)
            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, out, temp_output
    if not is_recording:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(temp_output.name, fourcc, 24.0, (640, 480))
        is_recording = True
    return '', 204


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, out, temp_output
    if is_recording:
        out.release()
        is_recording = False
        with open(temp_output.name, 'rb') as f:
            clean_db()

            processed_video_data = f.read()
            upload = Upload(filename='webcam_rec.mp4',
                            input=processed_video_data,
                            output=None
                            )
            db.session.add(upload)
            db.session.commit()
            session['upload_id'] = upload.id
    return '', 204


@app.route('/process_recording', methods=['GET', 'POST'])
def process_recording():
    if request.method == 'POST':
        model_option = request.form['model_option']
        background_option = request.form['background_option']

        background = request.files['background']

        upload_id = session.get('upload_id')
        upload = Upload.query.get(upload_id)

        if upload:
            upload.model_option = model_option
            upload.background_option = background_option

            # Daca fisierul e imagine se salveaza in db
            if image_file(background.filename):
                upload.upload_background = background.read()
            db.session.commit()
    return redirect(url_for('processing'))


# Webcam cu procesare live --------------------------------------------------------------------------

@app.route('/live_animation')
def live_animation():
    session['upload_id'] = -1
    return render_template('live_animation.html')


is_recording = False
out = None
temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)


def live_animation_frames(model_option, background):
    camera = cv.VideoCapture(0)
    global is_recording, out
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            if is_recording:
                start = time.time()
                # Procesam imaginea in mijlocul procesului de redare a webcam-ului
                frame = animate_frame(frame, model=model_option, bg=background)
                end = time.time()
                framet = round((end-start) * 12)
                for i in range(framet):
                    out.write(frame)

            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/live_feed')
def live_feed():
    model_option = request.args.get('model_option')
    background_option = request.args.get('background_option')

    upload_id = session.get('upload_id')
    if not upload_id == -1:
        upload = Upload.query.get(upload_id)
        if upload.upload_background is not None:
            bg_base64 = base64.b64encode(upload.upload_background).decode('utf-8')
            bg_data = base64.b64decode(bg_base64)
            bg_np_arr = np.frombuffer(bg_data, np.uint8)
            bg = cv.imdecode(bg_np_arr, cv.IMREAD_COLOR)
            return Response(live_animation_frames(model_option, bg),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(live_animation_frames(model_option, background_option), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload-background', methods=['POST'])
def upload_background():
    if 'upload_background' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['upload_background']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if image_file(file.filename):
        # Salveaza in db
        new_upload = Upload(
            upload_background=file.read()
        )
        db.session.add(new_upload)
        db.session.commit()
        session['upload_id'] = new_upload.id

        return jsonify({'success': True, 'id': new_upload.id}), 200
    else:
        return jsonify({'error': 'File is not a supported image type.'}), 400


@app.route('/start_recording_live', methods=['POST'])
def start_recording_live():
    global is_recording, out, temp_output
    if not is_recording:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(temp_output.name, fourcc, 12.0, (640, 480))
        is_recording = True
    return '', 204


@app.route('/stop_recording_live', methods=['POST'])
def stop_recording_live():
    global is_recording, out, temp_output
    upload_id = session.get('upload_id')
    if is_recording:
        out.release()
        is_recording = False
        with open(temp_output.name, 'rb') as f:
            processed_video_data = f.read()
            if not upload_id == -1:
                # cu imagine de fundal de la utilizator
                upload = Upload.query.get(upload_id)
                upload.filename = 'webcam_rec.mp4'
                upload.input = processed_video_data
                upload.output = processed_video_data
            else:
                clean_db()
                # fara imagine de fundal de la utilizator
                upload = Upload(filename='webcam_rec.mp4',
                                input=processed_video_data,
                                output=processed_video_data)
            db.session.add(upload)
            db.session.commit()
            session['upload_id'] = upload.id
    return '', 204
