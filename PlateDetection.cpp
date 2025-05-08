

#include "PlateDetection.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;


Mat convertToGrayscale(Mat source) {

    Mat gray(source.rows, source.cols, CV_8UC1);

    for (int y = 0; y < source.rows; ++y) {
        for (int x = 0; x < source.cols; ++x) {
            Vec3b pixel = source.at<Vec3b>(y, x);
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];

            uchar grayValue = (b+g+r)/3;
            gray.at<uchar>(y, x) = grayValue;
        }
    }

    return gray;
}

Mat enhanceContrast(Mat source) {
    Mat result = source.clone();

    double minVal, maxVal;
    minMaxLoc(source, &minVal, &maxVal);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            uchar pixel = source.at<uchar>(i, j);
            result.at<uchar>(i, j) = static_cast<uchar>(255.0 * (pixel - minVal) / (maxVal - minVal));
        }
    }

    return result;
}

Mat applyThreshold(Mat source, int thresholdValue) {
    Mat binary(source.rows, source.cols, CV_8UC1);

    for (int i = 0; i < source.rows; ++i) {
        for (int j = 0; j < source.cols; ++j) {
            uchar pixel = source.at<uchar>(i, j);
            binary.at<uchar>(i, j) = (pixel > thresholdValue) ? 255 : 0;
        }
    }

    return binary;
}



Mat applyGaussianBlur(Mat source) {

    Mat blurred = source.clone();

    int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    int kernelWeight = 16;

    for (int i = 1; i < source.rows - 1; ++i) {
        for (int j = 1; j < source.cols - 1; ++j) {
            int sum = 0;

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int pixel = source.at<uchar>(i + ki, j + kj);
                    sum += pixel * kernel[ki + 1][kj + 1];
                }
            }

            blurred.at<uchar>(i, j) = static_cast<uchar>(sum / kernelWeight);
        }
    }

    return blurred;
}



Mat detectEdgesCannyLite(Mat source, int lowThreshold, int highThreshold) {
    Mat edges = Mat::zeros(source.size(), CV_8UC1);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
         {0,  0,  0},
         {1,  2,  1}
    };

    Mat gradientMagnitude(source.rows, source.cols, CV_32FC1);
    Mat gradientDirection(source.rows, source.cols, CV_32FC1);

    // 1. Calcul Gx, Gy, magnitudine și unghi
    for (int i = 1; i < source.rows - 1; ++i) {
        for (int j = 1; j < source.cols - 1; ++j) {
            float sumX = 0, sumY = 0;

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int pixel = source.at<uchar>(i + ki, j + kj);
                    sumX += pixel * Gx[ki + 1][kj + 1];
                    sumY += pixel * Gy[ki + 1][kj + 1];
                }
            }

            float magnitude = std::sqrt(sumX * sumX + sumY * sumY);
            float angle = std::atan2(sumY, sumX) * 180.0 / CV_PI;
            if (angle < 0) angle += 180;

            gradientMagnitude.at<float>(i, j) = magnitude;
            gradientDirection.at<float>(i, j) = angle;
        }
    }

    // 2. Non-Maximum Suppression
    for (int i = 1; i < source.rows - 1; ++i) {
        for (int j = 1; j < source.cols - 1; ++j) {
            float angle = gradientDirection.at<float>(i, j);
            float magnitude = gradientMagnitude.at<float>(i, j);

            float neighbor1 = 0, neighbor2 = 0;

            // Determinăm vecinii în funcție de direcție
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                neighbor1 = gradientMagnitude.at<float>(i, j - 1);
                neighbor2 = gradientMagnitude.at<float>(i, j + 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                neighbor1 = gradientMagnitude.at<float>(i - 1, j + 1);
                neighbor2 = gradientMagnitude.at<float>(i + 1, j - 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                neighbor1 = gradientMagnitude.at<float>(i - 1, j);
                neighbor2 = gradientMagnitude.at<float>(i + 1, j);
            } else if (angle >= 112.5 && angle < 157.5) {
                neighbor1 = gradientMagnitude.at<float>(i - 1, j - 1);
                neighbor2 = gradientMagnitude.at<float>(i + 1, j + 1);
            }

            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                // 3. Aplicăm prag
                if (magnitude >= highThreshold)
                    edges.at<uchar>(i, j) = 255;  // muchie clară
                else if (magnitude >= lowThreshold)
                    edges.at<uchar>(i, j) = 100;  // muchie slabă
            }
        }
    }

    // 4. Hysteresis thresholding
    for (int i = 1; i < edges.rows - 1; ++i) {
        for (int j = 1; j < edges.cols - 1; ++j) {
            if (edges.at<uchar>(i, j) == 100) {
                // Verificăm cei 8 vecini
                bool connectedToStrongEdge = false;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        if (edges.at<uchar>(i + di, j + dj) == 255) {
                            connectedToStrongEdge = true;
                            break;
                        }
                    }
                    if (connectedToStrongEdge) break;
                }

                if (connectedToStrongEdge)
                    edges.at<uchar>(i, j) = 255;  // devine muchie puternică
                else
                    edges.at<uchar>(i, j) = 0;    // eliminat
            }
        }
    }


    return edges;
}


Mat manualDilate(Mat source) {
    Mat result = source.clone();

    for (int i = 1; i < source.rows - 1; ++i) {
        for (int j = 1; j < source.cols - 1; ++j) {
            bool foundWhite = false;

            for (int ki = -1; ki <= 1 && !foundWhite; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    if (source.at<uchar>(i + ki, j + kj) == 255) {
                        foundWhite = true;
                    }
                }
            }

            result.at<uchar>(i, j) = foundWhite ? 255 : 0;
        }
    }

    return result;
}


double computeContourArea(vector<Point> contour) {
    double area = 0.0;
    int n = contour.size();

    for (int i = 0; i < n; ++i) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % n];
        area += (p1.x * p2.y) - (p2.x * p1.y);
    }

    return abs(area) / 2.0;
}


vector<Point> detectBestPlate(const Mat& image, const Mat &edgeImage) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edgeImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<Point> bestPlate;
    double maxScore = 0.0;

    int imageCenterX = image.cols / 2;
    int maxOffset = image.cols / 4;

    for (const auto& contour : contours) {
        double area = computeContourArea(contour);
        if (area < 1000 || area > 16000)
            continue;

        vector<Point> approx;
        double epsilon = 0.03 * arcLength(contour, true);
        approxPolyDP(contour, approx, epsilon, true);
        if (approx.size() != 4)
            continue;

        Rect rect = boundingRect(approx);
        float aspectRatio = float(rect.width) / rect.height;
        if (aspectRatio < 2.0 || aspectRatio > 6.0)
            continue;

        int plateCenterX = rect.x + rect.width / 2;
        if (abs(plateCenterX - imageCenterX) > maxOffset)
            continue;

        float idealAspect = 4.0f;
        float shapeScore = 1.0f - std::abs(aspectRatio - idealAspect) / idealAspect;
        float score = area * shapeScore;

        if (score > maxScore) {
            maxScore = score;
            bestPlate = approx;
        }
    }

    return bestPlate;
}

void drawPlateContour(Mat image, const vector<Point> &plateContour) {
    if (!plateContour.empty()) {
        for (size_t i = 0; i < plateContour.size(); ++i) {
            line(image, plateContour[i], plateContour[(i + 1) % plateContour.size()], Scalar(0, 255, 255), 2);
        }
        cout << "Placuta incadrata cu succes.\n";
    } else {
        cout << "Nicio placuta valida nu a fost gasita.\n";
    }
}



