#/bin/bash
# usage: ./mp4towav.sh BASE_DIRECTORY
#   where BASE_DIRECTORY is the directory housing mp4 files.
#   output is generated in the same directory as mp4 files.

for file in "$1".mp4
do
	ffmpeg -i "$file" -vn -acodec pcm_s16le -ac 1 "$(basename "$file" .mp4).wav"
done
