CC = g++
CFLAGS = -Wall -Wextra -O0
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: annotated_video

annotated_video: annotated_video.cpp
	$(CC) $(CFLAGS) annotated_video.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f annotated_video
