import argparse
import os
import pandas as pd

import nemo.collections.asr as nemo_asr

from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description='Downloads and processes Mozilla Common Voice dataset.')
parser.add_argument('--language', type=str, required=True, help='Which language to download.'
                                   'check https://commonvoice.mozilla.org/en/datasets for more language codes')
parser.add_argument('--version', default='cv-corpus-8.0-2022-01-19',
                    type=str, help='Version of the dataset (obtainable via https://commonvoice.mozilla.org/en/datasets')
args = parser.parse_args()


COMMON_VOICE_URL = f"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" \
                   "{}/{}-{}.tar.gz".format(args.version, args.version, args.language)

FNULL = open(os.devnull, 'w')


def main():
    tar_filename = os.path.basename(COMMON_VOICE_URL)
    folder_name = tar_filename.replace('-' + args.language, '').replace('.tar.gz', '')

    os.makedirs(folder_name, exist_ok=True)

    csv_output = '../pca/%s-%s.csv' % (folder_name, args.language)

    if os.path.exists(csv_output):
        print("Corpus already preprocessed")
    else:
        if os.path.exists(tar_filename):
            print('Find existing folder {}'.format(folder_name))
        else:
            print("Could not find Common Voice, Downloading corpus...")
            subprocess.Popen(['wget', COMMON_VOICE_URL]).communicate()

            # Extract tar
            subprocess.Popen(['tar', '-zxvf', tar_filename]).communicate()

        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="speakerverification_speakernet"
        )
        # Read all tsv files
        train = pd.read_csv('%s/%s/train.tsv' % (folder_name, args.language), sep='\t')
        dev = pd.read_csv('%s/%s/dev.tsv' % (folder_name, args.language), sep='\t')
        test = pd.read_csv('%s/%s/test.tsv' % (folder_name, args.language), sep='\t')

        all_data = pd.concat([train, dev, test], axis=0)
        assert all_data.shape[0] == train.shape[0] + dev.shape[0] + test.shape[0]

        print('The language %s contains %d unique voices.' % (args.language, all_data.shape[0]))
        subset = all_data.drop_duplicates(subset=['client_id'])

        df = pd.DataFrame({})
        for _, row in tqdm(subset.iterrows()):
            sound_filename = row['path']
            mp3_path = '%s/%s/clips/%s' % (folder_name, args.language, sound_filename)

            wav_path = mp3_path.replace('.mp3', '.wav')
            if not os.path.exists(wav_path):
                subprocess.call(['sox', mp3_path, wav_path])

            embs = speaker_model.get_embedding(wav_path).cpu().detach().numpy()

            new_row = pd.DataFrame([{
                **dict(zip(['F' + str(i + 1) for i in range(256)], embs[0])),
                **dict(row)
            }])

            df = df.append(new_row, ignore_index=True)

        df.to_csv(csv_output, index=False)


if __name__ == "__main__":
    main()
