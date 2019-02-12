#!/bin/bash
# this script takes two arguments: vid and aud, and creates output at vid
#   which is a video file with aud audio added to it

ffmpeg -i "$1" -i "$2" -codec copy -shortest "$1"
