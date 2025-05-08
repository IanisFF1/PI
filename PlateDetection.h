//
// Created by IanisFatFrumos on 5/6/2025.
//

#ifndef PLATEDETECTION_H
#define PLATEDETECTION_H
#include <opencv2/opencv.hpp>

#endif //PLATEDETECTION_H

using namespace cv;
using namespace std;


Mat convertToGrayscale(Mat source);
Mat applyGaussianBlur(Mat source);
Mat detectEdgesCannyLite(Mat source, int lowThreshold, int highThreshold);
double computeContourArea(vector<Point> contour);
Mat enhanceContrast(Mat source);
Mat applyThreshold(Mat source, int thresholdValue);
Mat manualDilate(Mat source);
vector<Point> detectBestPlate(const Mat& image, const Mat &edgeImage);
void drawPlateContour(Mat image, const vector<Point> &plateContour);