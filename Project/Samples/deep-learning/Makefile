CC = g++
CFLAGS = -Wall -Wextra -O3 -g
PROFILE = -gp # using gprof
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

SRCS = main.cpp
HEADERS = input.hpp deepFace.hpp

all: deep-recognizer

deep-recognizer: $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(SRCS) -o deep-recognizer $(OPENCV_FLAGS)

clean:
	rm -f deep-recognizer