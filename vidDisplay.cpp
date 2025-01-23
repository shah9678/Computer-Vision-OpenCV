/*
Adit Shah
PRCV CS5330
Date: Jan 15 2025
*/

#include "filter.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
    cv::VideoCapture capdev(0);

    if (!capdev.isOpened()) {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " x " << refS.height << std::endl;

    cv::namedWindow("Video", 1);
    cv::Mat frame, grey, processedFrame, sobelXFrame, sobelYFrame, sobelXAbs, sobelYAbs, gradientMagnitude;

    // Add variables for face detection
    bool faceDetectionEnabled = false;
    std::vector<cv::Rect> faces;

    char lastKey = ' ';

    for (;;) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        if (faceDetectionEnabled) {
            // Convert the image to greyscale
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);

            // Detect faces
            detectFaces(grey, faces);

            // Draw boxes around the faces
            drawBoxes(frame, faces);
            cv::imshow("Video", frame);
        }

        switch (lastKey) {
            case 'g': // OpenCV grayscale
                cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);
                cv::cvtColor(processedFrame, processedFrame, cv::COLOR_GRAY2BGR); 
                cv::imshow("Video", processedFrame);
                break;

            case 'h': // Custom grayscale
                Customgreyscale(frame, processedFrame);
                cv::imshow("Video", processedFrame);
                break;

            case 's': // Save image
                //cv::imwrite("saved_image.jpg", frame);
                //std::cout << "Image saved to saved_image.jpg" << std::endl;
                cv::imshow("Video", frame);
                break;

            case 'b': // Custom 5x5 blur
                blur5x5_2(frame, processedFrame);
                cv::imshow("Video", processedFrame);
                break;

            case 'r': // Apply sepia tone
                applySepia(frame, processedFrame, false);
                cv::imshow("Video", processedFrame);
                break;

            case 'x': // Apply Sobel X filter
                sobelX3x3(frame, sobelXFrame); 
                cv::convertScaleAbs(sobelXFrame, sobelXAbs); 
                cv::imshow("Video", sobelXAbs);
                break;

            case 'y': // Apply Sobel Y filter
                sobelY3x3(frame, sobelYFrame);
                cv::convertScaleAbs(sobelYFrame, sobelYAbs);
                cv::imshow("Video", sobelYAbs);
                break;

            case 'm': // Apply Gradient Magnitude
                sobelX3x3(frame, sobelXFrame);
                sobelY3x3(frame, sobelYFrame);
                magnitude(sobelXFrame, sobelYFrame, gradientMagnitude);
                cv::imshow("Video", gradientMagnitude);
                break;

            case 'l': // Blur and Quantize
                blurQuantize(frame, processedFrame, 10);
                cv::imshow("Video", processedFrame);
                break;

            case 'f': // Toggle Face Detection
                faceDetectionEnabled = !faceDetectionEnabled;
                //printf("Face detection %s\n", faceDetectionEnabled ? "enabled" : "disabled");
                break;

            case 'e': // Emboss effect
                embossEffect(frame, processedFrame);
                cv::imshow("Video", processedFrame);
                break;

            case 'c': // Isolate color
                isolateColor(frame, processedFrame, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255)); // Example: Isolate red
                cv::imshow("Video", processedFrame);
                break;

            case 't': // Cartoonize the video
                cartoonize(frame, processedFrame);
                cv::imshow("Video", processedFrame);
                break;

            // Extensions
            case 'p': // Pencil Sketch Effect
                pencilSketch(frame, processedFrame);
                cv::imshow("Video", processedFrame);
                break;

            case 'v': // Apply sepia tone with vignette
                applySepia(frame, processedFrame, true);
                cv::imshow("Video", processedFrame);
                break;

            default:
                cv::imshow("Video", frame);
                break;
        }

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key != -1) {
            lastKey = key;
        }
    }

    return 0;
}
