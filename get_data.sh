#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=16PPRUpJVrtAlg65KoJsfPVAmiWp_Kw2C -O data.zip
    unzip data.zip
fi