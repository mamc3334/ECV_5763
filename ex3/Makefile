CC = g++
CFLAGS = -Wall -Wextra -O3 -g
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: interactive_hough background_removal_video oneband cv_oneband_mp4 com_crosshair

interactive_hough: interactive_hough.cpp
	$(CC) $(CFLAGS) interactive_hough.cpp -o $@ $(OPENCV_FLAGS)

background_removal_video: background_removal_video.cpp
	$(CC) $(CFLAGS) background_removal_video.cpp -o $@ $(OPENCV_FLAGS)

oneband: oneband.cpp
	$(CC) $(CFLAGS) oneband.cpp -o $@ $(OPENCV_FLAGS)

cv_oneband_mp4: cv_oneband_mp4.cpp
	$(CC) $(CFLAGS) cv_oneband_mp4.cpp -o $@ $(OPENCV_FLAGS)

cv_oneband_frames: cv_oneband_frames.cpp
	$(CC) $(CFLAGS) cv_oneband_frames.cpp -o $@ $(OPENCV_FLAGS)

com_crosshair: com_crosshair.cpp
	$(CC) $(CFLAGS) com_crosshair.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f interactive_hough background_removal_video oneband cv_oneband_mp4 cv_oneband_frames com_crosshair
