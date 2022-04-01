from glob import glob
import requests
import subprocess
from os.path import basename, exists
import time

URL = 'https://speaker-face-exp.s3.amazonaws.com/'
for img_path in glob('../stills/*.png'):
    out_path = '../images/toonify/' + basename(img_path)
    if not exists(out_path):
        r = requests.post(
            "https://api.deepai.org/api/toonify",
            data={
                'image': URL + img_path,
            },
            headers={'api-key': ''} # TODO use your API key
        )
        print(r.json())
        url = r.json()['output_url']

        subprocess.call(f'curl {url} --output {out_path}', shell=True)
        time.sleep(2)

