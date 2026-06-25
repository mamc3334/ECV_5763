CC = g++
CFLAGS = -Wall -Wextra -O3 -g
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: interactive_hough_lines

interactive_hough_lines: interactive_hough_lines.cpp
	$(CC) $(CFLAGS) interactive_hough_lines.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f interactive_hough_lines
