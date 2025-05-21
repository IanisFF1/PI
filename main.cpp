#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include "PlateDetection.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

struct GroundTruthEntry {
    int x, y, width, height;
};

map<string, GroundTruthEntry> loadGroundTruth(const string& csvPath) {
    map<string, GroundTruthEntry> gt;
    ifstream file(csvPath);
    string line;
    getline(file, line); // skip header

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;
        string xStr, yStr, wStr, hStr;

        getline(ss, filename, ',');
        getline(ss, xStr, ',');
        getline(ss, yStr, ',');
        getline(ss, wStr, ',');
        getline(ss, hStr, ',');

        GroundTruthEntry entry = {
            stoi(xStr),
            stoi(yStr),
            stoi(wStr),
            stoi(hStr)
        };

        gt[filename] = entry;
    }

    return gt;
}

double computeIoU(Rect a, Rect b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);

    int interArea = max(0, x2 - x1) * max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;

    return unionArea > 0 ? (double)interArea / unionArea : 0.0;
}

int main() {
    string inputDir = "C:\\Users\\IanisFatFrumos\\Desktop\\AN 3\\Laboratoare\\PI\\Proiect1\\images\\";
    string outputDir = "C:\\Users\\IanisFatFrumos\\Desktop\\AN 3\\Laboratoare\\PI\\Proiect1\\results\\";
    string gtPath = "C:\\Users\\IanisFatFrumos\\Desktop\\AN 3\\Laboratoare\\PI\\Proiect1\\ground_truth.csv";

    fs::create_directory(outputDir);
    map<string, GroundTruthEntry> groundTruth = loadGroundTruth(gtPath);

    double totalIoU = 0.0;
    int count = 0;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        string path = entry.path().string();
        string filenameOnly = fs::path(path).filename().string();

        Mat image = imread(path);
        if (image.empty()) {
            cerr << "Failed to read image: " << path << endl;
            continue;
        }

        Mat gray = convertToGrayscale(image);
        Mat contrast = enhanceContrast(gray);
        Mat blurred = applyGaussianBlur(contrast);
        Mat edges = detectEdgesCannyLite(blurred, 50, 120);
        dilate(edges, edges, getStructuringElement(MORPH_RECT, Size(2, 2)));

        vector<Point> plate = detectBestPlate(image, edges);
        if (plate.empty()) {
            cout << "Nicio placuta detectata pentru: " << filenameOnly << endl;
            continue;
        }

        Rect detected = boundingRect(plate);

        if (groundTruth.count(filenameOnly)) {
            GroundTruthEntry gt = groundTruth[filenameOnly];
            Rect real(gt.x, gt.y, gt.width, gt.height);
            double iou = computeIoU(detected, real);
            int scorePercent = static_cast<int>(iou * 100);
            totalIoU += iou;
            count++;

            rectangle(image, real, Scalar(0, 255, 0), 2);      // ground truth – verde
            rectangle(image, detected, Scalar(0, 0, 255), 2);  // detectat – roșu

            cout << "Scor de potrivire pentru " << filenameOnly << ": " << scorePercent << "%" << endl;
        } else {
            cout << "Ground truth lipsa pentru: " << filenameOnly << endl;
        }

        string outputPath = outputDir + filenameOnly;
        imwrite(outputPath, image);
    }

    if (count > 0) {
        int avgScore = static_cast<int>((totalIoU / count) * 100);
        cout << "\nScor mediu pe toate imaginile: " << avgScore << "%" << endl;
    } else {
        cout << "Nicio imagine cu ground truth valid.\n";
    }

    return 0;
}
