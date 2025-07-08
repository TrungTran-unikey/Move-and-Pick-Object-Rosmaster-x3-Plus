#!/usr/bin/env python3.8
import time
from playsound import playsound
import os

def play_sound_files():

    music_file = 'root/YOLOv11/src/yolov11/scripts/music/Daylight.wav'  # Change to your file if needed

    if os.path.exists(music_file):
        print(f"Playing sound file: {music_file}")
        playsound(music_file)
        # rospy.loginfo(f'Playing {music_file}')
        # Wait for the duration of the song (adjust as needed)
    else:
        import sys; sys.exit(f"File {music_file} does not exist. Please check the path.")


if __name__ == '__main__':
    try:
        play_sound_files()
    except Exception as e:
        print(f"An error occurred: {e}")
        pass