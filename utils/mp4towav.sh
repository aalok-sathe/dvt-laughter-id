#/bin/bash
# usage: ./utils/mp4towav.sh BASE_DIRECTORY
#   run from root of the repo
#   where BASE_DIRECTORY is the directory housing mp4 files.
#   output is generated in BASE_DIRECTORY/../wav/

for file in "$1"/*.mp4
do
    if [ ! -f "$1/../wav/$(basename "$file" .mp4).wav" ]; then
	    ffmpeg -i "$file" -vn -acodec pcm_s16le -ac 1 "$1/../wav/$(basename "$file" .mp4).wav"
    fi
done
