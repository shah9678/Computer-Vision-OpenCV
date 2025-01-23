/*
Adit Shah
CS5330 PRCV
Create Date: Jan 15 2025 
*/

#include "filter.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

using namespace cv;

int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst = src.clone();
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            unsigned char gray = static_cast<unsigned char>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(gray, gray, gray);
        }
    }
    return 0;
}

int Customgreyscale(cv::Mat &src, cv::Mat &dst) {
    dst = src.clone();

    // Apply an even more aggressive reduction in contrast and color contribution
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            unsigned char gray = static_cast<unsigned char>(0.2 * pixel[2] + 0.2 * pixel[1] + 0.2 * pixel[0]);
            gray = static_cast<unsigned char>(gray * 0.7 + 100); 
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(gray, gray, gray);
        }
    }
    return 0;
}


int applySepia(cv::Mat &src, cv::Mat &dst, bool applyVignette) {
    dst = src.clone();
    cv::Point center(src.cols / 2, src.rows / 2);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            int blue = std::min(255, static_cast<int>(0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0]));
            int green = std::min(255, static_cast<int>(0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0]));
            int red = std::min(255, static_cast<int>(0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0]));
            
            //Extension 
            if (applyVignette) {
                double dist = cv::norm(cv::Point(j, i) - center);
                double maxDist = cv::norm(cv::Point(0, 0) - center); 
                double vignetteFactor = 1.0 - (dist / maxDist) * 0.5; 
                blue = std::min(255, static_cast<int>(blue * vignetteFactor));
                green = std::min(255, static_cast<int>(green * vignetteFactor));
                red = std::min(255, static_cast<int>(red * vignetteFactor));
            }
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(blue, green, red);
        }
    }
    return 0;
}

int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp = src.clone();
    dst = src.clone();
    // Horizontal pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            for (int c = 0; c < 3; c++) {
                dst.at<cv::Vec3b>(y, x)[c] =
                    (temp.at<cv::Vec3b>(y, x - 2)[c] +
                     2 * temp.at<cv::Vec3b>(y, x - 1)[c] +
                     4 * temp.at<cv::Vec3b>(y, x)[c] +
                     2 * temp.at<cv::Vec3b>(y, x + 1)[c] +
                     temp.at<cv::Vec3b>(y, x + 2)[c]) /
                    10;
            }
        }
    }
    // Vertical pass
    temp = dst.clone();
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int c = 0; c < 3; c++) {
                dst.at<cv::Vec3b>(y, x)[c] =
                    (temp.at<cv::Vec3b>(y - 2, x)[c] +
                     2 * temp.at<cv::Vec3b>(y - 1, x)[c] +
                     4 * temp.at<cv::Vec3b>(y, x)[c] +
                     2 * temp.at<cv::Vec3b>(y + 1, x)[c] +
                     temp.at<cv::Vec3b>(y + 2, x)[c]) /
                    10;
            }
        }
    }
    return 0;
}

int sobelX3x3(Mat &src, Mat &dst) {
    if (src.empty()) {
        return -1; // Return error if the source image is empty
    }
 
    // Create dst with the same size and type as src
    dst = Mat::zeros(src.size(), CV_16SC3); // Use 16SC3 for signed short
 
    // Sobel X kernel
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
 
    for (int i = 1; i < src.rows - 1; ++i) {
        Vec3b* prevRow = src.ptr<Vec3b>(i - 1);
        Vec3b* currRow = src.ptr<Vec3b>(i);
        Vec3b* nextRow = src.ptr<Vec3b>(i + 1);
        Vec3s* dstRow = dst.ptr<Vec3s>(i);
 
        for (int j = 1; j < src.cols - 1; ++j) {
            Vec3s sum = {0, 0, 0};
 
            for (int k = -1; k <= 1; ++k) {
                Vec3b* row = (k == -1) ? prevRow : (k == 0) ? currRow : nextRow;
                sum[0] += kernelX[k + 1][0] * row[j - 1][0] +
                          kernelX[k + 1][1] * row[j][0] +
                          kernelX[k + 1][2] * row[j + 1][0];
                sum[1] += kernelX[k + 1][0] * row[j - 1][1] +
                          kernelX[k + 1][1] * row[j][1] +
                          kernelX[k + 1][2] * row[j + 1][1];
                sum[2] += kernelX[k + 1][0] * row[j - 1][2] +
                          kernelX[k + 1][1] * row[j][2] +
                          kernelX[k + 1][2] * row[j + 1][2];
            }
 
            dstRow[j] = sum;
        }
    }
 
    return 0; // Success
}
 
int sobelY3x3(Mat &src, Mat &dst) {
    if (src.empty()) {
        return -1; // Return error if the source image is empty
    }
 
    // Create dst with the same size and type as src
    dst = Mat::zeros(src.size(), CV_16SC3); // Use 16SC3 for signed short
 
    // Sobel Y kernel
    int kernelY[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };
 
    for (int i = 1; i < src.rows - 1; ++i) {
        Vec3b* prevRow = src.ptr<Vec3b>(i - 1);
        Vec3b* currRow = src.ptr<Vec3b>(i);
        Vec3b* nextRow = src.ptr<Vec3b>(i + 1);
        Vec3s* dstRow = dst.ptr<Vec3s>(i);
 
        for (int j = 1; j < src.cols - 1; ++j) {
            Vec3s sum = {0, 0, 0};
 
            for (int k = -1; k <= 1; ++k) {
                Vec3b* row = (k == -1) ? prevRow : (k == 0) ? currRow : nextRow;
                sum[0] += kernelY[k + 1][0] * row[j - 1][0] +
                          kernelY[k + 1][1] * row[j][0] +
                          kernelY[k + 1][2] * row[j + 1][0];
                sum[1] += kernelY[k + 1][0] * row[j - 1][1] +
                          kernelY[k + 1][1] * row[j][1] +
                          kernelY[k + 1][2] * row[j + 1][1];
                sum[2] += kernelY[k + 1][0] * row[j - 1][2] +
                          kernelY[k + 1][1] * row[j][2] +
                          kernelY[k + 1][2] * row[j + 1][2];
            }
 
            dstRow[j] = sum;
        }
    }
 
    return 0; // Success
}
int magnitude(Mat &sx, Mat &sy, Mat &dst) {
    if (sx.empty() || sy.empty()) {
        return -1;
    }
    dst = Mat::zeros(sx.size(), CV_8UC3);
    for (int i = 0; i < sx.rows; ++i) {
        Vec3s* sxRow = sx.ptr<Vec3s>(i);
        Vec3s* syRow = sy.ptr<Vec3s>(i);
        Vec3b* dstRow = dst.ptr<Vec3b>(i);
        for (int j = 0; j < sx.cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                float mag = std::sqrt(sxRow[j][c] * sxRow[j][c] + syRow[j][c] * syRow[j][c]);
                dstRow[j][c] = saturate_cast<uchar>(mag);
            }
        }
    }
    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);
    float bucketSize = 255.0f / levels;
    dst = blurred.clone();
    for (int y = 0; y < blurred.rows; y++) {
        for (int x = 0; x < blurred.cols; x++) {
            for (int c = 0; c < 3; c++) {
                float pixelValue = blurred.at<cv::Vec3b>(y, x)[c];
                int quantizedValue = static_cast<int>(pixelValue / bucketSize) * bucketSize;
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(quantizedValue);
            }
        }
    }

    return 0;
}
int embossEffect(cv::Mat &src, cv::Mat &dst) {
    cv::Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);

    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    float directionX = 0.7071f, directionY = 0.7071f;

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3s gx = sobelX.at<cv::Vec3s>(y, x);
            cv::Vec3s gy = sobelY.at<cv::Vec3s>(y, x);

            for (int c = 0; c < 3; ++c) {
                int embossValue = static_cast<int>(gx[c] * directionX + gy[c] * directionY + 128);
                dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(embossValue);
            }
        }
    }
    return 0;
}
int isolateColor(cv::Mat &src, cv::Mat &dst, cv::Scalar lowerBound, cv::Scalar upperBound) {
    cv::Mat mask, gray;
    dst = src.clone();
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, lowerBound, upperBound, mask);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
    gray.copyTo(dst);
    src.copyTo(dst, mask);
    return 0;
}
int cartoonize(cv::Mat &src, cv::Mat &dst, int ksize, int edgeThreshold) {
    cv::Mat smooth;
    cv::bilateralFilter(src, smooth, ksize, ksize * 2, ksize / 2);
    cv::Mat gray, edges;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, edgeThreshold, edgeThreshold * 2);
    cv::bitwise_not(edges, edges);
    cv::Mat edgesColor;
    cv::cvtColor(edges, edgesColor, cv::COLOR_GRAY2BGR); 
    dst = smooth & edgesColor; 

    return 0;
}

//Extensions 
int pencilSketch(cv::Mat &src, cv::Mat &dst) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat invertedGray;
    cv::bitwise_not(gray, invertedGray);
    cv::Mat blurredInverted;
    cv::GaussianBlur(invertedGray, blurredInverted, cv::Size(21, 21), 0);
    cv::Mat pencilSketch;
    cv::divide(gray, 255 - blurredInverted, pencilSketch, 256);
    cv::cvtColor(pencilSketch, dst, cv::COLOR_GRAY2BGR);

    return 0;
}

int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces ) {
  // a static variable to hold a half-size image
  static cv::Mat half;
  
  // a static variable to hold the classifier
  static cv::CascadeClassifier face_cascade;

  // the path to the haar cascade file
  static cv::String face_cascade_file(FACE_CASCADE_FILE);

  if( face_cascade.empty() ) {
    if( !face_cascade.load( face_cascade_file ) ) {
      printf("Unable to load face cascade file\n");
      printf("Terminating\n");
      exit(-1);
    }
  }

  // clear the vector of faces
  faces.clear();
  
  // cut the image size in half to reduce processing time
  cv::resize( grey, half, cv::Size(grey.cols/2, grey.rows/2) );

  // equalize the image
  cv::equalizeHist( half, half );

  // apply the Haar cascade detector
  face_cascade.detectMultiScale( half, faces );

  // adjust the rectangle sizes back to the full size image
  for(int i=0;i<faces.size();i++) {
    faces[i].x *= 2;
    faces[i].y *= 2;
    faces[i].width *= 2;
    faces[i].height *= 2;
  }

  return(0);
}
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale  ) {
  // The color to draw, you can change it here (B, G, R)
  cv::Scalar wcolor(170, 120, 110);

  for(int i=0;i<faces.size();i++) {
    if( faces[i].width > minWidth ) {
      cv::Rect face( faces[i] );
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      cv::rectangle( frame, face, wcolor, 3 );
    }
  }

  return(0);
}
