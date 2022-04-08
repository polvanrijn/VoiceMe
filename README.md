# 🗣 VoiceMe: *Personalized voice generation in TTS*
[![arXiv](https://img.shields.io/badge/arXiv-2203.15379-b31b1b.svg)](https://arxiv.org/abs/2203.15379)

> **Abstract**
>
> Novel text-to-speech systems can generate entirely new voices that were not seen during training. However, it remains a difficult task to efficiently create personalized voices from a high dimensional speaker space. In this work, we use speaker embeddings from a state-of-the-art speaker verification model (SpeakerNet) trained on thousands of speakers to condition a TTS model. We employ a human sampling paradigm to explore this speaker latent space. We show that users can create voices that fit well to photos of faces, art portraits, and cartoons. We recruit online participants to collectively manipulate the voice of a speaking face. We show that (1) a separate group of human raters confirms that the created voices match the faces, (2) speaker gender apparent from the face is well-recovered in the voice, and (3) people are consistently moving towards the real voice prototype for the given face. Our results demonstrate that this technology can be applied in a wide number of applications including character voice development in audiobooks and games, personalized speech assistants, and individual voices for people with speech impairment.


    
## Demos
- 📢 [Demo website](https://polvanrijn.github.io/VoiceMe/)
- 🔇 Unmute to listen to the videos on Github:

https://user-images.githubusercontent.com/19990643/160813289-fb467e5d-7526-425e-986d-8d827808369e.mp4

https://user-images.githubusercontent.com/19990643/160814914-bc32f156-270c-4415-a489-fa0a8460d441.mp4

## Preprocessing
Setup the repository
````shell
git clone https://github.com/polvanrijn/VoiceMe.git
cd VoiceMe
main_dir=$PWD

preprocessing_env="$main_dir/preprocessing-env"
conda create --prefix $preprocessing_env python=3.7
conda activate $preprocessing_env
pip install Cython
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
pip install requests
````

### Create face styles
We used the same sentence ("Kids are talking by the door", neutral recording) from the RAVDESS corpus from all 24 
speakers. You can download all videos by running `download_RAVDESS.sh`. However, the stills used in the 
paper are also part of the repository (`stills`). We can create the AI Gahaku styles by running 
`python ai_gahaku.py` and the toonified version by running `python toonify.py` (you need to add 
your API key).

### Obtain the PCA space
The model used in the paper was trained on SpeakerNet embeddings, so we to extract the embeddings from a dataset. Here
we use the commonvoice data. To download it, run: `python preprocess_commonvoice.py --language en`

To extract the principal components, run `compute_pca.py`.

## Synthesis
### Setup
We'll assume, you'll setup a remote instance for synthesis. Clone the repo and setup the virtual environment:
````shell
git clone https://github.com/polvanrijn/VoiceMe.git
cd VoiceMe
main_dir=$PWD

synthesis_env="$main_dir/synthesis-env"
conda create --prefix $synthesis_env python=3.7
conda activate $synthesis_env

##############
# Setup Wav2Lip
##############
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# Install Requirements
pip install -r requirements.txt
pip install opencv-python-headless==4.1.2.30
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "face_detection/detection/sfd/s3fd.pth"  --no-check-certificate

# Install as package
mv ../setup_wav2lip.py setup.py
pip install -e .
cd ..


##############
# Setup VITS
##############
git clone https://github.com/jaywalnut310/vits
cd vits

# Install Requirements
pip install -r requirements.txt

# Install monotonic_align
mv monotonic_align ../monotonic_align

# Download the VCTK checkpoint
pip install gdown
gdown 'https://drive.google.com/file/d/11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru'

# Install as package
mv ../setup_vits.py setup.py
pip install -e .

cd ../monotonic_align
python setup.py build_ext --inplace
cd ..


pip install flask
pip install wget

````

You'll need to do the last step manually (let me know if you know an automatic way). Download the
checkpoint `wav2lip_gan.pth` from [here](https://github.com/Rudrabha/Wav2Lip) and put it in `Wav2Lip/checkpoints`. Make 
sure you have `espeak` installed and it is in `PATH`.

### Running
Start the remote service (I used port 31337)
```shell
python server.py --port 31337
```

You can send an example request locally, by running (don't forget to change host and port accordingly):
```shell
python request_demo.py
```

We also made a small 'playground' so you can see how slider values will influence the voice. Start the local flask app called `client.py`.

### Experiment
The GSP experiment cannot be shared at this moment, as PsyNet is still under development.





