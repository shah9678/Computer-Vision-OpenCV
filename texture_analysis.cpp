/* CS5330 PRCV 
ADIT SHAH
JHEEL KAMDAR
DATE: 01/31/2025 */

// EXTENSION FOR TASK 4- Fourier Transform, Gabor Filter, GLCM and LBP features //



#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to compute Fourier Transform and return 16x16 Power Spectrum
Mat computeFFTFeatures(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    Mat padded; 
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    
    dft(complexI, complexI);

    // Compute power spectrum
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magnitudeSpectrum = planes[0];

    magnitudeSpectrum += Scalar::all(1);
    log(magnitudeSpectrum, magnitudeSpectrum);

    normalize(magnitudeSpectrum, magnitudeSpectrum, 0, 255, NORM_MINMAX, CV_8U);

    // Resize to 16x16
    Mat resizedSpectrum;
    resize(magnitudeSpectrum, resizedSpectrum, Size(16, 16));

    return resizedSpectrum;
}

// Function to apply Gabor filter
Mat applyGaborFilter(const Mat& src, double theta) {
    Mat kernel = getGaborKernel(Size(21, 21), 4.0, theta, 10.0, 0.5, 0, CV_32F);
    Mat filtered;
    filter2D(src, filtered, CV_32F, kernel);
    return filtered;
}

// Function to compute LBP histogram
Mat computeLBP(const Mat& src) {
    Mat gray, lbp = Mat::zeros(src.size(), CV_8U);

    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            uchar center = gray.at<uchar>(i, j);
            uchar code = 0;
            code |= (gray.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (gray.at<uchar>(i - 1, j) > center) << 6;
            code |= (gray.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (gray.at<uchar>(i, j + 1) > center) << 4;
            code |= (gray.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (gray.at<uchar>(i + 1, j) > center) << 2;
            code |= (gray.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (gray.at<uchar>(i, j - 1) > center) << 0;
            lbp.at<uchar>(i, j) = code;
        }
    }
    
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    calcHist(&lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    normalize(hist, hist, 0, 255, NORM_MINMAX);
    return hist;
}

// Function to compute GLCM features
vector<double> computeGLCM(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    Mat glcm = Mat::zeros(256, 256, CV_32F);
    for (int i = 0; i < gray.rows - 1; i++) {
        for (int j = 0; j < gray.cols - 1; j++) {
            int pixel = gray.at<uchar>(i, j);
            int neighbor = gray.at<uchar>(i, j + 1);
            glcm.at<float>(pixel, neighbor)++;
        }
    }

    normalize(glcm, glcm, 0, 1, NORM_MINMAX);

    double contrast = 0, energy = 0, homogeneity = 0, correlation = 0, meanI = 0, meanJ = 0, stdI = 0, stdJ = 0;

    // Compute statistical properties
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            contrast += glcm.at<float>(i, j) * (i - j) * (i - j);
            energy += glcm.at<float>(i, j) * glcm.at<float>(i, j);
            homogeneity += glcm.at<float>(i, j) / (1 + abs(i - j));
        }
    }

    return {contrast, energy, homogeneity};
}

// Main function to process image and display results
int main() {
    Mat image = imread("/Users/aditshah/Desktop/target.jpg");
    if (image.empty()) {
        cout << "Could not open image!" << endl;
        return -1;
    }

    Mat fftFeatures = computeFFTFeatures(image);
    Mat gaborFiltered = applyGaborFilter(image, CV_PI / 4); // Example with 45-degree angle
    Mat lbpHistogram = computeLBP(image);
    vector<double> glcmFeatures = computeGLCM(image);

    cout << "GLCM Contrast: " << glcmFeatures[0] << endl;
    cout << "GLCM Energy: " << glcmFeatures[1] << endl;
    cout << "GLCM Homogeneity: " << glcmFeatures[2] << endl;

    imshow("Original Image", image);
    imshow("FFT Features (16x16)", fftFeatures);
    imshow("Gabor Filtered", gaborFiltered);
    imshow("LBP Histogram", lbpHistogram);
    waitKey(0);
    
    return 0;
}
