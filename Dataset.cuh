#ifndef DATASET_CUH
#define DATASET_CUH

#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include "Tensor.cuh"

static int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

static void readMNISTImages(const std::string &filename, std::vector<float> &images, int &nImages, int &nRows, int &nCols)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int magicNumber = 0;
    int numberOfImages = 0;
    int rows = 0;
    int cols = 0;

    file.read((char *)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);
    file.read((char *)&numberOfImages, sizeof(numberOfImages));
    numberOfImages = reverseInt(numberOfImages);
    file.read((char *)&rows, sizeof(rows));
    rows = reverseInt(rows);
    file.read((char *)&cols, sizeof(cols));
    cols = reverseInt(cols);

    nImages = numberOfImages;
    nRows = rows;
    nCols = cols;

    images.resize(nImages * nRows * nCols);

    for (int i = 0; i < nImages; i++)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                unsigned char temp = 0;
                file.read((char *)&temp, 1);
                images[i * rows * cols + r * cols + c] = ((float)temp) / 255.0f;
            }
        }
    }
}

static void readMNISTLabels(const std::string &filename, std::vector<float> &labels, int &nLabels)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int magicNumber = 0;
    int numberOfLabels = 0;

    file.read((char *)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);
    file.read((char *)&numberOfLabels, sizeof(numberOfLabels));
    numberOfLabels = reverseInt(numberOfLabels);

    nLabels = numberOfLabels;
    labels.resize(nLabels);

    for (int i = 0; i < nLabels; i++)
    {
        unsigned char temp = 0;
        file.read((char *)&temp, 1);
        labels[i] = (float)temp; // label is 0..9
    }
}

class MNISTDataset
{
public:
    std::vector<float> trainImages;
    std::vector<float> testImages;
    std::vector<float> trainLabels;
    std::vector<float> testLabels;

    int trainSize;
    int testSize;
    int rows;
    int cols;

    MNISTDataset(const std::string &trainImagePath,
                 const std::string &trainLabelPath,
                 const std::string &testImagePath,
                 const std::string &testLabelPath)
    {
        int dummyRows, dummyCols;
        std::cout << "Loading MNIST dataset..." << std::endl;
        readMNISTImages(trainImagePath, trainImages, trainSize, dummyRows, dummyCols);
        std::cout << "trainSize: " << trainSize << std::endl;
        readMNISTLabels(trainLabelPath, trainLabels, trainSize);
        std::cout << "trainLabels: " << trainLabels.size() << std::endl;
        readMNISTImages(testImagePath, testImages, testSize, rows, cols);
        std::cout << "testSize: " << testSize << std::endl;
        readMNISTLabels(testLabelPath, testLabels, testSize);
        std::cout << "testLabels: " << testLabels.size() << std::endl;


        // If needed, unify rows/cols from train/test
        // but MNIST is standard 28x28
        // so dummyRows==rows, dummyCols==cols, typically.
        rows = dummyRows;
        cols = dummyCols;
    }
};

#endif // DATASET_CUH
