#!/bin/bash
# this script takes three arguments: vid, aud, out and creates output at out
#   which is a video file corresponding to vid with aud audio added to it

ffmpeg -i "$1" -i "$2" -c:v libx264 -c:a libvorbis -shortest "$3"
