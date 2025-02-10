/* CS5330 PRCV 
ADIT SHAH
JHEEL KAMDAR
DATE: 01/31/2025 */


#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "dirent.h"

using namespace std;
using namespace cv;

// Convert filename to lowercase
string toLowercase(const string& str) {
    string lowerStr = str;
    transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

// Extract color histogram features
vector<float> extractColorHistogramFeatures(const Mat& image, int bins = 8) {
    Mat img_color;
    if (image.channels() == 1) {
        cvtColor(image, img_color, COLOR_GRAY2BGR);
    } else {
        img_color = image.clone();
    }

    const int total_bins = bins * bins * bins;
    vector<float> histogram(total_bins, 0.0);
    int totalPixels = img_color.rows * img_color.cols;

    for (int i = 0; i < img_color.rows; ++i) {
        for (int j = 0; j < img_color.cols; ++j) {
            Vec3b pixel = img_color.at<Vec3b>(i, j);
            
            int bBin = static_cast<int>(pixel[0] / (256.0 / bins));
            int gBin = static_cast<int>(pixel[1] / (256.0 / bins));
            int rBin = static_cast<int>(pixel[2] / (256.0 / bins));
            
            bBin = min(max(bBin, 0), bins-1);
            gBin = min(max(gBin, 0), bins-1);
            rBin = min(max(rBin, 0), bins-1);

            int index = rBin * bins * bins + gBin * bins + bBin;
            histogram[index] += 1.0;
        }
    }

    for (float &val : histogram) {
        val /= totalPixels;
    }
    return histogram;
}

// Extract texture histogram features using Sobel magnitude
vector<float> extractTextureHistogramFeatures(const Mat& image, int bins = 8) {
    Mat img_gray;
    if (image.channels() == 3) {
        cvtColor(image, img_gray, COLOR_BGR2GRAY);
    } else {
        img_gray = image.clone();
    }

    // Compute Sobel gradients
    Mat grad_x, grad_y;
    Sobel(img_gray, grad_x, CV_32F, 1, 0);
    Sobel(img_gray, grad_y, CV_32F, 0, 1);

    // Compute gradient magnitude
    Mat grad_mag;
    magnitude(grad_x, grad_y, grad_mag);

    // Normalize gradient magnitude to [0, 255]
    normalize(grad_mag, grad_mag, 0, 255, NORM_MINMAX, CV_8U);

    // Compute histogram of gradient magnitudes
    vector<float> histogram(bins, 0.0);
    int totalPixels = grad_mag.rows * grad_mag.cols;

    for (int i = 0; i < grad_mag.rows; ++i) {
        for (int j = 0; j < grad_mag.cols; ++j) {
            int bin = static_cast<int>(grad_mag.at<uchar>(i, j) / (256.0 / bins));
            bin = min(max(bin, 0), bins-1);
            histogram[bin] += 1.0;
        }
    }

    for (float &val : histogram) {
        val /= totalPixels;
    }
    return histogram;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <image_directory> <output_csv> <bins>" << endl;
        return -1;
    }

    string imageDir = argv[1];
    string outputCSV = argv[2];
    int bins = stoi(argv[3]);

    ofstream csvFile(outputCSV);
    if (!csvFile.is_open()) {
        cerr << "Error: Could not open CSV file for writing: " << outputCSV << endl;
        return -1;
    }

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(imageDir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string filename = ent->d_name;
            string lowerFilename = toLowercase(filename);
            if (lowerFilename.find(".jpg") != string::npos || 
                lowerFilename.find(".png") != string::npos ||
                lowerFilename.find(".jpeg") != string::npos) {
                string imagePath = imageDir + "/" + filename;
                Mat image = imread(imagePath, IMREAD_COLOR);
                if (image.empty()) {
                    cerr << "Warning: Could not read image: " << imagePath << ". Skipping." << endl;
                    continue;
                }

                // Extract color histogram features
                vector<float> colorHist = extractColorHistogramFeatures(image, bins);

                // Extract texture histogram features
                vector<float> textureHist = extractTextureHistogramFeatures(image, bins);

                // Combine features
                vector<float> combinedFeatures;
                combinedFeatures.reserve(colorHist.size() + textureHist.size());
                combinedFeatures.insert(combinedFeatures.end(), colorHist.begin(), colorHist.end());
                combinedFeatures.insert(combinedFeatures.end(), textureHist.begin(), textureHist.end());

                // Write to CSV
                csvFile << imagePath;
                for (float f : combinedFeatures) {
                    csvFile << "," << f;
                }
                csvFile << endl;

                if (csvFile.fail()) {
                    cerr << "Error: Failed to write to CSV file: " << outputCSV << endl;
                    return -1;
                }
            }
        }
        closedir(dir);
    } else {
        cerr << "Error: Could not open directory: " << imageDir << endl;
        return -1;
    }

    csvFile.close();
    return 0;
}