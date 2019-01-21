#!/bin/bash

all:
    echo "nothing to do..."

download:
    echo "downloading models"
	if [ ! -f vggish_model.ckpt ]; then
	    echo "vggish_model.ckpt not found"
	    curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
	if [ ! -f vggish_pca_params.npz ]; then
	    echo "vggish_pca_params.npz not found"
		curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

test: *.py
	python3 vggish_smoke_test.py

clean:
	rm -f *.aux *.log *.synctex.gz *.out *.blg
