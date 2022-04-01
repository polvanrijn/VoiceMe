from flask import Flask, request, send_file, jsonify, make_response
import base64
import logging
import io
import struct
import json

from datetime import datetime

################
# Load VITS
################
from scipy.io.wavfile import write

from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

import numpy as np
import cv2, os
import subprocess
import torch
import platform
import tempfile

from Wav2Lip import audio
from Wav2Lip.face_detection import FaceAlignment, LandmarksType
from Wav2Lip.models import Wav2Lip


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


class CustomVITS(SynthesizerTrn):
    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None,
              spk_emb=None):
        assert sum([1 for s in [sid, spk_emb] if s is not None]) == 1, \
            "You can either set the sid or speaker embedding."

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            if sid is not None:
                g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            elif spk_emb is not None:
                g = spk_emb.unsqueeze(-1)
            else:
                NotImplementedError('This may not happen')
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1,
                                                                           2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1,
                                                                                 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)


######################
# Globals
######################
checkpoint_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"
STATIC = True  # If it is an image
FPS = 25  # Default: Only for images
WAV2LIP_BATCH_SIZE = 128  # Default: Batch size for Wav2Lip model
MEL_STEP_SIZE = 16
RESIZE_FACTOR = 1

# Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. Useful if
# multiple face present. -1 implies the value will be auto-inferred based on height, width
CROP = [0, -1, 0, -1]

# Specify a constant bounding box for the face. Use only as a last resort if the face is not detected. Also, might work
# only if the face is not moving around much. Syntax: (top, bottom, left, right).
BOX = [-1, -1, -1, -1]

# Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg. Use if you get a
# flipped result, despite feeding a normal looking video
ROTATE = False

# Prevent smoothing face detections over a short temporal window
NOSMOOTH = False

IMG_SIZE = 96


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

    batch_size = 16

    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
    for rect, image in zip(predictions, images):
        if rect is None:
            # cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not NOSMOOTH: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if BOX[0] == -1:
        if not STATIC:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = BOX
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if STATIC else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= WAV2LIP_BATCH_SIZE:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, IMG_SIZE // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, IMG_SIZE // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def synthesize(audio_path, face_path, outfile):
    with tempfile.TemporaryDirectory() as out_dir:
        ext = face_path.split('.')[1]
        if not os.path.isfile(face_path):
            raise ValueError('--face argument must be a valid path to video/image file')
        elif ext in ['jpg', 'png', 'jpeg']:
            face_path_video = face_path.replace(ext, 'mp4')
            subprocess.call(
                f'ffmpeg  -hide_banner -loglevel error -y -i {face_path} -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" {face_path_video}',
                shell=shell)
            face_path = face_path_video

        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if RESIZE_FACTOR > 1:
                frame = cv2.resize(frame, (frame.shape[1] // RESIZE_FACTOR, frame.shape[0] // RESIZE_FACTOR))

            if ROTATE:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = CROP
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

        print("Number of frames available for inference: " + str(len(full_frames)))

        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -hide_banner -loglevel error -y -i {} -strict -2 {}'.format(audio_path,
                                                                                          out_dir + '/temp.wav')

            subprocess.call(command, shell=shell)
            audio_path = out_dir + '/temp.wav'

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + MEL_STEP_SIZE > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + MEL_STEP_SIZE])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        gen = datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(out_dir + '/result.avi',
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()
        print('Creating video')
        command = 'ffmpeg -hide_banner -loglevel error -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path,
                                                                                                   out_dir + '/result.avi',
                                                                                                   outfile)
        subprocess.call(command, shell=shell)


def make_batch_file(in_files, output_path):
    with open(output_path, "wb") as output:
        for in_file in in_files:
            b = os.path.getsize(in_file)
            output.write(struct.pack("I", b))
            with open(in_file, "rb") as i:
                output.write(i.read())


def normalize(x):
    return x / np.linalg.norm(x)


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


@app.route('/api/tts_lipsync/synthesize_batched', methods=["POST"])
def generate():
    begin_time = datetime.now()
    app.logger.info("Synthesizing stimulus...")
    data = request.files

    app.logger.info("Data received...")
    app.logger.info(request.files)
    assert "text" in data
    assert "spk_emb" in data
    assert "face" in data
    key = 'tmp'

    text = data['text'].read().decode("utf-8")
    app.logger.info(text)

    with tempfile.TemporaryDirectory() as out_dir:

        output_file = os.path.join(out_dir, key)
        bname = key.split('.')[0]

        embeddings_filename = os.path.join(out_dir, "speaker_embeddings.npy")
        with open(embeddings_filename, "wb") as f:
            f.write(data['spk_emb'].read())
        speaker_embeddings = np.load(embeddings_filename)
        app.logger.info(speaker_embeddings)

        reference_filename = os.path.join(out_dir, "reference.png")
        with open(reference_filename, "wb") as f:
            f.write(data['face'].read())
        video_paths = []
        for i in range(speaker_embeddings.shape[0]):
            speaker_embedding = speaker_embeddings[i]
            print(speaker_embedding.shape)

            spk_emb = torch.from_numpy(speaker_embedding.astype(np.float32)).cuda()[None]

            tmp_wav_path = os.path.join(out_dir, bname + '_' + str(i) + '.wav')

            stn_tst = get_text(text, hps)
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                audio = \
                net_g.infer(x_tst, x_tst_lengths, spk_emb=spk_emb, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[
                    0][
                    0, 0].data.cpu().float().numpy()

            # Generate wav
            write(tmp_wav_path, hps.data.sampling_rate, audio)

            tmp_video_path = tmp_wav_path.replace('.wav', '.mp4')
            video_paths.append(tmp_video_path)
            synthesize(tmp_wav_path, reference_filename, tmp_video_path)

        # merge videos together
        if len(video_paths) == 1:
            output_file = tmp_video_path
        else:
            make_batch_file(video_paths, output_file)
        app.logger.info(f'Elapsed time: {datetime.now() - begin_time}')
        with open(output_file, 'rb') as bites:
            return send_file(
                io.BytesIO(bites.read()),
                attachment_filename=key,
                mimetype='video/mp4'
            )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file("vits/configs/vctk_base.json")
    net_g = CustomVITS(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint("vits/pretrained_vctk.pth", net_g, None)

    ################
    # Wav2Lib
    ################
    shell = platform.system() != 'Windows'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} for inference.'.format(device))

    # initialize Wav2Lib model
    model = load_model(checkpoint_path)
    print("Model loaded")

    app.run(debug=True, host=args.host, port=args.port)
