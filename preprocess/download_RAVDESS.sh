for i in $(seq -f "%02g" 1 24)
do
  url=https://zenodo.org/record/1188976/files/Video_Speech_Actor_$i.zip
  filename=Video_Speech_Actor_$i.zip
  echo "Download $url"
  curl $url --output $filename

  unzip $filename
done