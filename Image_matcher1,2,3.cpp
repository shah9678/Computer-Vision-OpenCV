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


// Task 1 Baseline 7 X 7 patch //


vector<float> extractPatchFeatures(const Mat& image) {
    Mat img_color;
    if (image.channels() == 1) {
        cvtColor(image, img_color, COLOR_GRAY2BGR);
    } else {
        img_color = image.clone();
    }

    int target_width = 7;
    int target_height = 7;

    if (img_color.cols < target_width || img_color.rows < target_height) {
        int top = (target_height - img_color.rows) / 2;
        int bottom = target_height - img_color.rows - top;
        int left = (target_width - img_color.cols) / 2;
        int right = (target_width - img_color.cols - left);
        copyMakeBorder(img_color, img_color, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
    }

    int x = (img_color.cols - target_width) / 2;
    int y = (img_color.rows - target_height) / 2;
    Rect roi(x, y, target_width, target_height);
    Mat patch = img_color(roi);

    vector<float> features;
    for (int i = 0; i < patch.rows; ++i) {
        for (int j = 0; j < patch.cols; ++j) {
            Vec3b pixel = patch.at<Vec3b>(i, j);
            features.push_back(static_cast<float>(pixel[0]));
            features.push_back(static_cast<float>(pixel[1]));
            features.push_back(static_cast<float>(pixel[2]));
        }
    }
    return features;
}


//Task 2 Histogram Matching //

vector<float> extractHistogramFeatures(const Mat& image, int bins = 8) {
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


// Task 3 Multi Histogram Matching //


vector<float> extractSpatialHistograms(const Mat& image, int bins = 8) {
    Mat img_color;
    if (image.channels() == 1) {
        cvtColor(image, img_color, COLOR_GRAY2BGR);
    } else {
        img_color = image.clone();
    }

    // Split image into top and bottom halves
    int midY = img_color.rows / 2;
    Rect topHalf(0, 0, img_color.cols, midY);
    Rect bottomHalf(0, midY, img_color.cols, img_color.rows - midY);
    
    Mat topPart = img_color(topHalf);
    Mat bottomPart = img_color(bottomHalf);

    // Extract histograms for both regions
    vector<float> topHist = extractHistogramFeatures(topPart, bins);
    vector<float> bottomHist = extractHistogramFeatures(bottomPart, bins);

    // Combine features
    vector<float> combined;
    combined.reserve(topHist.size() + bottomHist.size());
    combined.insert(combined.end(), topHist.begin(), topHist.end());
    combined.insert(combined.end(), bottomHist.begin(), bottomHist.end());
    
    return combined;
}


float histogramIntersection(const vector<float>& hist1, const vector<float>& hist2) {
    float intersection = 0.0f;
    for (size_t i = 0; i < hist1.size(); ++i) {
        intersection += min(hist1[i], hist2[i]);
    }
    return intersection;
}

//MAIN FUNCTION //

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <target_image> <feature_csv> <N> <feature_type>" << endl;
        return -1;
    }

    string targetImagePath = argv[1];
    string featureCSV = argv[2];
    int N = stoi(argv[3]);
    string featureType = argv[4];

    Mat targetImage = imread(targetImagePath, IMREAD_COLOR);
    if (targetImage.empty()) {
        cerr << "Could not read target image: " << targetImagePath << endl;
        return -1;
    }

    vector<float> targetFeatures;
    if (featureType == "patch") {
        targetFeatures = extractPatchFeatures(targetImage);
    } else if (featureType == "histogram") {
        targetFeatures = extractHistogramFeatures(targetImage);
    } else if (featureType == "spatial_histogram") {
        targetFeatures = extractSpatialHistograms(targetImage);
    } else {
        cerr << "Invalid feature type: " << featureType << " (use 'patch', 'histogram' or 'spatial_histogram')" << endl;
        return -1;
    }

    vector<pair<float, string>> distances;
    ifstream csvFile(featureCSV);
    if (!csvFile.is_open()) {
        cerr << "Could not open feature CSV: " << featureCSV << endl;
        return -1;
    }

    string line;
    while (getline(csvFile, line)) {
        stringstream ss(line);
        string filename;
        getline(ss, filename, ',');

        vector<float> features;
        string token;
        while (getline(ss, token, ',')) {
            features.push_back(stof(token));
        }

        if (features.size() != targetFeatures.size()) {
            cerr << "Feature size mismatch for " << filename << ". Skipping." << endl;
            continue;
        }

        float distance;
        if (featureType == "patch") {
            distance = norm(targetFeatures, features, NORM_L2);
        } else if (featureType == "histogram") {
            distance = 1.0f - histogramIntersection(targetFeatures, features);
        } else if (featureType == "spatial_histogram") {
            distance = 1.0f - histogramIntersection(targetFeatures, features);
        }
        distances.emplace_back(distance, filename);
    }

    sort(distances.begin(), distances.end());

    int num = min(N, (int)distances.size());
    // Create display parameters
    const int thumbnail_width = 200;
    const int thumbnail_height = 200;
    const int spacing = 10;
    
    // Create composite image
    int composite_width = thumbnail_width * (num + 1) + spacing * num;
    int composite_height = thumbnail_height + 40;  // Extra space for text
    Mat composite = Mat::zeros(composite_height, composite_width, CV_8UC3);
    
    // Process query image
    Mat queryDisplay;
    resize(targetImage, queryDisplay, Size(thumbnail_width, thumbnail_height));
    Rect queryROI(0, 0, thumbnail_width, thumbnail_height);
    queryDisplay.copyTo(composite(queryROI));
    putText(composite, "Query", Point(10, thumbnail_height + 20), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

    // Process matches
    int x_offset = thumbnail_width + spacing;
    for(int i = 0; i < num; i++) {
        Mat matchImage = imread(distances[i].second);
        if(matchImage.empty()) continue;

        // Resize and copy to composite
        Mat resizedMatch;
        resize(matchImage, resizedMatch, Size(thumbnail_width, thumbnail_height));
        Rect matchROI(x_offset, 0, thumbnail_width, thumbnail_height);
        resizedMatch.copyTo(composite(matchROI));

        // Add text label
        string label = "Match " + to_string(i+1);
        putText(composite, label, Point(x_offset + 10, thumbnail_height + 20), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);

        x_offset += thumbnail_width + spacing;
    }
    
    // Display composite image
    namedWindow("Results", WINDOW_AUTOSIZE);
    imshow("Results", composite);
    waitKey(0);

    return 0;
} 