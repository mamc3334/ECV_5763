CC = g++
CFLAGS = -Wall -Wextra -O0 -g
PROFILE = -gp # using gprof
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
DEBUG = # -g

all: simple_detect cuda_detect hog_detect lbp_recognize.cpp

simple_detect: simple_detect.cpp
	$(CC) $(CFLAGS) simple_detect.cpp -o $@ $(OPENCV_FLAGS)

cuda_detect: cuda_detect.cpp
	$(CC) $(CFLAGS) cuda_detect.cpp -o $@ $(OPENCV_FLAGS)

hog_detect: hog_detect.cpp
	$(CC) $(CFLAGS) hog_detect.cpp -o $@ $(OPENCV_FLAGS)

lbp_recognize: lbp_recognize.cpp
	$(CC) $(CFLAGS) lbp_recognize.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f simple_detect cuda_detect hog_detect lbp_recognize
