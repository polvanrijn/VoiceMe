import io
import numpy as np
from datetime import datetime
import requests
from util import convert_pca_weights_to_speaker_embedding, to_bytes
import tempfile
def get_zero_embedding():
    b = io.BytesIO()
    np.save(b, np.zeros((1, 256)))
    b.seek(0)
    return b

start = datetime.now()
url = "http://0.0.0.0:31337"
image_file = 'stills/speaker01.png'
text = "The quick brown fox jumps over the lazy dog."
weights = [[-0.8] * 10]
speaker_embedding = convert_pca_weights_to_speaker_embedding(weights, 'vctk-vits')
with tempfile.TemporaryDirectory() as out_dir:
    try:
        response = requests.post(
            url + "/api/tts_lipsync/synthesize_batched",
            files={
                "text": text,
                "spk_emb": to_bytes(speaker_embedding),
                "style_emb": get_zero_embedding(),
                "face": open(image_file, "rb"),
            },
            headers={
                'voice': 'vits',
                'vocoder': 'identity'
            }
        )
        with open('test.mp4', 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(e)
print(f'The request took {datetime.now() - start}')


