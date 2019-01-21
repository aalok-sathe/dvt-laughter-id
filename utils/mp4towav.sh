#/bin/bash
# usage: ./mp4towav.sh BASE_DIRECTORY
#   where BASE_DIRECTORY is the directory housing mp4 files.
#   output is generated in <directory>/../wav/

for file in "$1"/*.mp4
do
	ffmpeg -i "$file" -vn -acodec pcm_s16le -ac 1 "$1/../wav/$(basename "$file" .mp4).wav"
done
