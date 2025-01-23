#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

int greyscale(cv::Mat &src, cv::Mat &dst);
int Customgreyscale(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels = 10); 
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int embossEffect(cv::Mat &src, cv::Mat &dst);
int isolateColor(cv::Mat &src, cv::Mat &dst, cv::Scalar lowerBound, cv::Scalar upperBound);
int cartoonize(cv::Mat &src, cv::Mat &dst, int ksize = 15, int edgeThreshold = 30);

#define FACE_CASCADE_FILE "../resources/haarcascade_frontalface_alt2.xml"

// Function prototypes
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces);
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 20, float scale = 1.0);

// Extensions
int pencilSketch(cv::Mat &src, cv::Mat &dst);
int applySepia(cv::Mat &src, cv::Mat &dst, bool applyVignette = false); //vignetting is an extension

#endif
