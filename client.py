import subprocess

from flask import Flask, request, send_file, render_template
from util import convert_pca_weights_to_speaker_embedding, to_bytes
import os
import logging
import io
import tempfile
import json
import requests
import base64
import numpy as np
import torch

from datetime import datetime

################
# Constants
################
N_PCA = 10

app = Flask(__name__)
# Load the models here

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

@app.route('/')
def show_editor():
    return render_template('client.html')

def get_empty_embedding():
    b = io.BytesIO()
    #np.save(b, torch.nn.functional.normalize(torch.randn(1,256)).numpy())
    np.save(b, np.zeros((1,256)))
    b.seek(0)
    return b

@app.route('/api/tts_lipsync/synthesize_batched', methods=["POST", "GET"])
def generate():
    begin_time = datetime.now()
    app.logger.info("Synthesizing stimulus...")
    if request.method == 'POST':
        data = request.json
    else:
        data = request.args.to_dict()
    assert "text" in data
    assert "pca_weights" in data
    assert "model_name" in data
    assert "face" in data
    key = 'temp'

    data["normalize_embedding"] = data["normalize_embedding"] == 'true'
    print(data["normalize_embedding"])
    pca_weights = data["pca_weights"]


    with tempfile.TemporaryDirectory() as out_dir:
        output_file = out_dir + '/' + pca_weights + '.mp4'

        if type(pca_weights) == str:
            pca_weights = json.loads(pca_weights)

        face_path = out_dir + '/face'
        with open(face_path, 'wb') as f:
            f.write(base64.b64decode(data["reference_image"]))

        try:
            response = requests.post(
                data["server"] + "/api/tts_lipsync/synthesize_batched",
                files={
                    "text": data['text'],
                    "spk_emb": to_bytes(convert_pca_weights_to_speaker_embedding(
                        pca_weights,
                        data["model_name"],
                        data["normalize_embedding"]
                    )),
                    "style_emb": get_empty_embedding(),
                    "face": open(face_path, 'rb'),
                },
                headers={
                    'voice': 'vits',
                    'vocoder': 'identity'
                }
            )
            app.logger.info(f'The request took {datetime.now() - begin_time}')

            with open(output_file, 'wb') as f:
                f.write(response.content)
            #tmp_file = output_file.replace(".mp4", "_tmp.mp4")
            #subprocess.call(f'ffmpeg -y -i {output_file} {tmp_file}', shell=True)
            return send_file(
                output_file,
                #tmp_file,
                mimetype='video/mp4',
                as_attachment=True,
                attachment_filename=key
            )
        except Exception as e:
            app.logger.error(e)

        #with open(output_file, 'rb') as bites:

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    app.run(debug=True, host=args.host, port=args.port)
