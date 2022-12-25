#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=10SO0P5Ompp2fck5fhJXL2pDgCIgUnPNO -O data.zip
    unzip data.zip
fi