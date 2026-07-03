CC = g++
CFLAGS = -Wall -Wextra -O0 -g
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: hough_lines_p

hough_lines_p: hough_lines_p.cpp
	$(CC) $(CFLAGS) hough_lines_p.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f hough_lines_p
