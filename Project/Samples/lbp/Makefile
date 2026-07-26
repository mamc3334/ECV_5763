CC = g++
CFLAGS = -Wall -Wextra -O3 -g
PROFILE = -gp # using gprof
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
DEBUG = # -g

all: lbp_recognize simple_lbp_recognize threaded_recognizer openmp_recognizer ensemble

lbp_recognize: lbp_recognize.cpp
	$(CC) $(CFLAGS) lbp_recognize.cpp -o $@ $(OPENCV_FLAGS)

simple_lbp_recognize: simple_lbp_recognize.cpp
	$(CC) $(CFLAGS) simple_lbp_recognize.cpp -o $@ $(OPENCV_FLAGS)

threaded_recognizer: threaded_recognizer.cpp
	$(CC) $(CFLAGS) threaded_recognizer.cpp -o $@ $(OPENCV_FLAGS)

openmp_recognizer: openmp_recognizer.cpp
	$(CC) $(CFLAGS) -fopenmp openmp_recognizer.cpp -o $@ $(OPENCV_FLAGS)

ensemble: ensemble.cpp
	$(CC) $(CFLAGS) -fopenmp ensemble.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f lbp_recognize simple_lbp_recognize threaded_recognizer openmp_recognizer ensemble
