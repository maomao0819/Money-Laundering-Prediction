#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1l_hUANVarwHy7jmo4xgCHSFBaeBVmjkE -O data.zip
    unzip data.zip
fi