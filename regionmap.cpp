#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;

// Function to generate random colors
Vec3b randomColor(RNG& rng) {
    return Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

// Function to compute features for a region
void computeRegionFeatures(const Mat& regionMap, int regionID, const Mat& stats, const Mat& centroids, Mat& output, vector<double>& featureVector) {
    Mat regionMask = (regionMap == regionID);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(regionMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        RotatedRect orientedBoundingBox = minAreaRect(contours[0]);
        double regionArea = contourArea(contours[0]);
        double boundingBoxArea = orientedBoundingBox.size.area();
        double percentFilled = (boundingBoxArea > 0) ? (regionArea / boundingBoxArea) * 100 : 0;
        double heightWidthRatio = orientedBoundingBox.size.height / orientedBoundingBox.size.width;
        featureVector = {regionArea, percentFilled, heightWidthRatio};

        Point2f vertices[4];
        orientedBoundingBox.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(output, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
        }
    }
}

// Function to save feature vectors to a CSV file
void saveFeatureVectors(const vector<double>& featureVector, const string& label) {
    ofstream file("/Users/aditshah/Desktop/object_db.csv", ios::app);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file for writing" << endl;
        return;
    }
    file << label;
    for (double feature : featureVector) {
        file << "," << feature;
    }
    file << endl;
    file.close();
}

// Function to classify a new object based on nearest-neighbor recognition
string classifyObject(const vector<double>& newFeatureVector) {
    ifstream file("/Users/aditshah/Desktop/object_db.csv");
    if (!file.is_open()) {
        cerr << "Error: Unable to open object database" << endl;
        return "Unknown";
    }

    string line, label, bestLabel = "Unknown";
    vector<vector<double>> database;
    vector<string> labels;
    double minDistance = DBL_MAX;
    vector<double> means(3, 0.0), stdevs(3, 0.0);
    int count = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        getline(ss, label, ',');
        vector<double> features;
        double val;
        while (ss >> val) {
            features.push_back(val);
            if (ss.peek() == ',') ss.ignore();
        }
        if (features.size() == 3) {
            database.push_back(features);
            labels.push_back(label);
            for (int i = 0; i < 3; i++) {
                means[i] += features[i];
            }
            count++;
        }
    }
    file.close();

    if (count == 0) return "Unknown";

    for (int i = 0; i < 3; i++) means[i] /= count;
    for (const auto& features : database) {
        for (int i = 0; i < 3; i++) {
            stdevs[i] += pow(features[i] - means[i], 2);
        }
    }
    for (int i = 0; i < 3; i++) stdevs[i] = sqrt(stdevs[i] / count);

    for (size_t i = 0; i < database.size(); i++) {
        double distance = 0.0;
        for (int j = 0; j < 3; j++) {
            if (stdevs[j] > 0) {
                distance += pow((newFeatureVector[j] - database[i][j]) / stdevs[j], 2);
            }
        }
        distance = sqrt(distance);

        if (distance < minDistance) {
            minDistance = distance;
            bestLabel = labels[i];
        }
    }
    return (minDistance < 2.0) ? bestLabel : "Unknown";
}

// Function to load test images and their true labels
vector<pair<Mat, string>> loadTestImages(const string& basePath) {
    vector<pair<Mat, string>> testImages;
    vector<string> labels = {"1", "2", "Phone", "Cup", "Pattern","5"};
    for (const string& label : labels) {
        for (int i = 1; i <= 3; i++) {
            string path = basePath + "/" + label + "_" + to_string(i) + ".jpg";
            Mat img = imread(path, IMREAD_COLOR);
            if (!img.empty()) {
                testImages.push_back({img, label});
            } else {
                cerr << "Error: Unable to load image " << path << endl;
            }
        }
    }
    return testImages;
}

// Function to evaluate the system and generate the confusion matrix
void evaluateSystem(const vector<pair<Mat, string>>& testImages, Mat& confusionMatrix) {
    vector<string> labels = {"1", "2", "Phone", "Cup", "Pattern","5"};
    for (const auto& testPair : testImages) {
        Mat frame = testPair.first;
        string trueLabel = testPair.second;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(15, 15), 0);

        Mat thresholded;
        threshold(gray, thresholded, 128, 255, THRESH_BINARY);

        Mat labelsMat, stats, centroids;
        int numLabels = connectedComponentsWithStats(thresholded, labelsMat, stats, centroids);

        for (int i = 1; i < numLabels; i++) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area > 1000) {
                vector<double> featureVector;
                computeRegionFeatures(labelsMat, i, stats, centroids, frame, featureVector);
                string predictedLabel = classifyObject(featureVector);

                // Update confusion matrix
                int trueIndex = distance(labels.begin(), find(labels.begin(), labels.end(), trueLabel));
                int predictedIndex = distance(labels.begin(), find(labels.begin(), labels.end(), predictedLabel));
                confusionMatrix.at<int>(trueIndex, predictedIndex)++;
            }
        }
    }
}

int main() {
    // Load test images
    string basePath = "test"; // Update this path to your test images directory
    vector<pair<Mat, string>> testImages = loadTestImages(basePath);

    // Initialize confusion matrix
    Mat confusionMatrix = Mat::zeros(5, 5, CV_32S);

    // Evaluate system
    evaluateSystem(testImages, confusionMatrix);

    // Print confusion matrix
    cout << "Confusion Matrix:" << endl;
    cout << confusionMatrix << endl;

    // Continue with the original video capture and processing
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open the camera" << endl;
        return -1;
    }

    namedWindow("Original Video", WINDOW_AUTOSIZE);
    namedWindow("Region Map", WINDOW_AUTOSIZE);
    RNG rng(12345);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Unable to capture frame" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(15, 15), 0);

        Mat thresholded;
        threshold(gray, thresholded, 128, 255, THRESH_BINARY);

        Mat labels, stats, centroids;
        int numLabels = connectedComponentsWithStats(thresholded, labels, stats, centroids);

        Mat regionMap = Mat::zeros(labels.size(), CV_8UC3);

        for (int i = 1; i < numLabels; i++) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area > 1000) {
                Vec3b color = randomColor(rng);
                regionMap.setTo(color, labels == i);
                vector<double> featureVector;
                computeRegionFeatures(labels, i, stats, centroids, regionMap, featureVector);
                string label = classifyObject(featureVector);
                putText(regionMap, label, Point(stats.at<int>(i, CC_STAT_LEFT), stats.at<int>(i, CC_STAT_TOP)), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            }
        }

        imshow("Original Video", frame);
        imshow("Region Map", regionMap);

        char key = waitKey(30);
        if (key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}