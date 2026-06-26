#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[1] << " <file> [band: 0=R, 1=G, 2=B] \n";
        return -1;
    }

    int band = (argc > 2) ? stoi(argv[2]) : 0;

    string inputPath = argv[1];

    ifstream inFile(inputPath, ios::binary);
    if (!inFile) {
        cerr << "Error: Cannot open input file: " << inputPath << endl;
        return -1;
    }

    string magic;
    inFile >> magic;
    if (magic != "P6") {
        cerr << "Error: Not PPM file" << endl;
        return -1;
    }

    int width, height, maxVal;
    inFile >> width;
    inFile >> height;
    inFile >> maxVal;

    size_t numPixels = (size_t)width * height;
    vector<char> outBuf(numPixels);

    vector<char> inBuf(numPixels * 3);
    inFile.read(inBuf.data(), inBuf.size());
    
    // Get only the target channel
    for (size_t i = 0, j = band; 
            i < numPixels;
            i++, j += 3) 
    {
        outBuf[i] = inBuf[j];
    }

    size_t idx = inputPath.find_last_of(".");
    string outPath = inputPath.substr(0, idx) + ".pgm";
    
    ofstream outFile(outPath, ios::binary);
    if (!outFile) {
        cerr << "Error: Cannot open output file: " << outPath << endl;
        return -1;
    }

    outFile << "P5\n" << width << " " << height << "\n" << maxVal << "\n";
    outFile.write(outBuf.data(), outBuf.size());
    outFile.close();

    return 0;
}