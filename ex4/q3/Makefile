CC = g++
CFLAGS = -Wall -Wextra -O0 -g
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: skeletal

skeletal: skeletal.cpp count_finger.cpp
	$(CC) $(CFLAGS) skeletal.cpp count_finger.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f skeletal
