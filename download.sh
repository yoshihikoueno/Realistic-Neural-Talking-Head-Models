#!/bin/bash
FILE_ID=1vdFz4sh23hC_KIQGJjwbTfUdPG-aYor8
FILE_NAME=model_weights.tar

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=$FILE_ID" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=$FILE_ID" -o $FILE_NAME
