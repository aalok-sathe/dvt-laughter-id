#!/bin/bash

all:
	echo "nothing to do..."

download:
	echo "downloading models"
	if [ ! -e "vggish_model.ckpt" ]; then \
		echo "vggish_model.ckpt not found"
		curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
	fi
	if [ ! -e "vggish_pca_params.npz" ]; then \
		echo "vggish_pca_params.npz not found"
		curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
	fi

test: *.py
	python3 vggish_smoke_test.py

clean:
	rm -f *.ckpt *.npz
