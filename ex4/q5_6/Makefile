CC = g++
CFLAGS = -Wall -Wextra -O0 -g
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: cuda_interactive_hough 

cuda_interactive_hough: cuda_interactive_hough.cpp
	$(CC) $(CFLAGS) cuda_interactive_hough.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f cuda_interactive_hough
