#include <opencv2/opencv.hpp>
#include <iostream>

#include "PlateDetection.h"

using namespace cv;

int main() {

    Mat image = imread("C:\\Users\\IanisFatFrumos\\Desktop\\AN 3\\Laboratoare\\PI\\Proiect1\\images\\masina5.bmp");
    imshow("Original image", image);
    moveWindow("Original image", image.rows/2, image.cols/2);


    Mat gray = convertToGrayscale(image);
    imshow("GrayScale Image", gray);

    Mat contrast = enhanceContrast(gray);
    imshow("Contrast Image", contrast);

    Mat binary = applyThreshold(contrast, 100);
    imshow("Binary Image", binary);

    Mat blurred = applyGaussianBlur(contrast);
    imshow("Blurred Image", blurred);

    Mat edges = detectEdgesCannyLite(blurred, 50, 120);
    imshow("Edges Canny", edges);

    dilate(edges, edges, getStructuringElement(MORPH_RECT, Size(2, 2)));
    imshow("Edges Dilated", edges);

    vector<Point> bestPlate = detectBestPlate(image, edges);
    Mat result = image.clone();
    drawPlateContour(result, bestPlate);
    imshow("Final result", result);


    waitKey(0);
    return 0;
}
