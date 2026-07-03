CC = g++
CFLAGS = -Wall -Wextra -O3 -g
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: hello_sift descriptor_extractor_matcher pyramids

hello_sift: hello_sift.cpp
	$(CC) $(CFLAGS) hello_sift.cpp -o $@ $(OPENCV_FLAGS)

descriptor_extractor_matcher: descriptor_extractor_matcher.cpp
	$(CC) $(CFLAGS) descriptor_extractor_matcher.cpp -o $@ $(OPENCV_FLAGS)

pyramids: pyramids.cpp
	$(CC) $(CFLAGS) pyramids.cpp -o $@ $(OPENCV_FLAGS)

clean:
	rm -f hello_sift descriptor_extractor_matcher pyramids
