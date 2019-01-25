#/bin/bash
# usage: ./download_vggish.sh
#   run from root of the repo
#   files are downloaded inside vggish/

echo "downloading model checkpoint and weights for VGGish"
cd "vggish/"

# first check if files already exist
if [ ! -f vggish/vggish_model.ckpt ]; then
    curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
fi
if [ ! -f vggish/vggish_pca_params.npz ]; then
    curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
fi
