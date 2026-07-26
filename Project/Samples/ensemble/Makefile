CC = g++
CFLAGS = -Wall -Wextra -O3 -g
PROFILE = -gp # using gprof
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
DEBUG = # -g

all: tmp ensemble

tmp: tmp.cpp
	$(CC) $(CFLAGS) -fopenmp tmp.cpp -o $@ $(OPENCV_FLAGS)

ensemble: ensemble.cpp
	$(CC) $(CFLAGS) -fopenmp ensemble.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f tmp simple_tmp threaded_recognizer openmp_recognizer ensemble
