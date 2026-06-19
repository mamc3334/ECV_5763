CC = g++
CFLAGS = -Wall -Wextra -O0
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: interactive_edge

interactive_edge: interactive_edge.cpp
	$(CC) $(CFLAGS) interactive_edge.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f interactive_edge
