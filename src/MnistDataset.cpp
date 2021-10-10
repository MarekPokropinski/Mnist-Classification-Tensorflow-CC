#include "MnistDataset.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

thread_local std::mt19937 gen{std::random_device{}()};

/**
 * Converts 32bit value from big-endian to little-endian
 *
 * @param MSB Original value.
 * @return `MSB` converted to little-endian.
 */
unsigned int toLSB(unsigned int MSB) {
  return ((MSB & 0x000000ff) << 24) | ((MSB & 0x0000ff00) << 8) |
         ((MSB & 0x00ff0000) >> 8) | ((MSB & 0xff000000) >> 24);
}

void MnistDataset::loadImages(string imagesFile, vector<MnistImage> &images) {
  ifstream f(imagesFile, ios::binary);
  if (!f.good()) {
    cerr << "Couldn't open file: \"" << imagesFile << "\"" << endl;
    throw string("Couldn't open file: \"") + imagesFile + "\"";
  }
  unsigned int magic;
  unsigned int numOfImages;
  unsigned int numOfRows;
  unsigned int numOfCols;

  f.read((char *)&magic, sizeof(magic));
  f.read((char *)&numOfImages, sizeof(numOfImages));
  f.read((char *)&numOfRows, sizeof(numOfRows));
  f.read((char *)&numOfCols, sizeof(numOfCols));

  magic = toLSB(magic);
  numOfImages = toLSB(numOfImages);
  numOfRows = toLSB(numOfRows);
  numOfCols = toLSB(numOfCols);

  if (magic != 2051) {
    cerr << "Wrong magic number in file \"" << imagesFile
         << "\". Expected: 2051, got: " << magic << "." << endl;
  }
  if (numOfRows != 28 || numOfCols != 28) {
    cerr << "Wrong image size in file \"" << imagesFile
         << "\". Expected: 28x28, got: " << numOfRows << "x" << numOfCols << "."
         << endl;
  }
  images.resize(numOfImages);
  f.read((char *)images.data(), sizeof(MnistImage) * numOfImages);
}

void MnistDataset::loadLabels(string labelsFile,
                              vector<unsigned char> &labels) {
  ifstream f(labelsFile, ios::binary);
  if (!f.good()) {
    cerr << "Couldn't open file: \"" << labelsFile << "\"" << endl;
    throw string("Couldn't open file: \"") + labelsFile + "\"";
  }
  unsigned int magic;
  unsigned int numOfImages;

  f.read((char *)&magic, sizeof(magic));
  f.read((char *)&numOfImages, sizeof(numOfImages));

  magic = toLSB(magic);
  numOfImages = toLSB(numOfImages);

  if (magic != 2049) {
    cerr << "Wrong magic number in file \"" << labelsFile
         << "\". Expected: 2051, got: " << magic << "." << endl;
  }

  labels.resize(numOfImages);
  f.read((char *)labels.data(), numOfImages);
}

MnistDataset::MnistDataset(string datasetPath) {
  filesystem::path _datasetPath(datasetPath);

  loadImages(_datasetPath / "train-images-idx3-ubyte", trainImages);
  loadImages(_datasetPath / "t10k-images-idx3-ubyte", testImages);

  loadLabels(_datasetPath / "train-labels-idx1-ubyte", trainLabels);
  loadLabels(_datasetPath / "t10k-labels-idx1-ubyte", testLabels);

  for (size_t i = 0; i < trainImages.size(); i++) {
    trainIndices.push_back(i);
  }
}

void MnistDataset::shuffle() {
  std::random_shuffle(trainIndices.begin(), trainIndices.end());
}

void MnistDataset::getTrainMinibatch(size_t batchSize, size_t idx,
                                     vector<MnistImage> &images,
                                     vector<float> &oneHotLabels) {
  images.clear();
  oneHotLabels.clear();

  for (int i = idx * batchSize; i < (idx + 1) * batchSize; i++) {
    if (i >= trainImages.size())
      break;
    int shuffledIdx = trainIndices[i];
    images.push_back(trainImages[shuffledIdx]);
    unsigned char label = trainLabels[shuffledIdx];
    for (int j = 0; j < 10; j++) {
      oneHotLabels.push_back((unsigned char)j == label ? 1.0f : 0.0f);
    }
  }
}
