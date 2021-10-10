#pragma once
#include <string>
#include <vector>
using namespace std;

struct MnistImage {
    unsigned char data[28*28];
};

class MnistDataset {
    vector<MnistImage> trainImages;
    vector<MnistImage> testImages;
    vector<unsigned char> trainLabels;
    vector<unsigned char> testLabels;

    vector<size_t> trainIndices;

    void loadImages(string imagesFile, vector<MnistImage>& images);
    void loadLabels(string labelsFile, vector<unsigned char>& labels);

public:
    MnistDataset(string datasetPath);
    void shuffle();
    void getTrainMinibatch(size_t batchSize, size_t i, vector<MnistImage>&, vector<float>&);
    size_t trainSize() {
        return trainImages.size();
    }
};