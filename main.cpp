#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>


using namespace cv;
using namespace std;


// Function to generate random colors
Vec3b randomColor(RNG& rng) {
   return Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}


// Function to compute features for a region
void computeRegionFeatures(const Mat& regionMap, int regionID, const Mat& stats, const Mat& centroids, Mat& output, vector<vector<double>>& featureVectors) {
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
       featureVectors.push_back({regionArea, percentFilled, heightWidthRatio});
      
       Point2f vertices[4];
       orientedBoundingBox.points(vertices);
       for (int i = 0; i < 4; i++) {
           line(output, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
       }
   }
}


void saveFeatureVectors(const vector<vector<double>>& featureVectors, const string& label) {
   ofstream file("object_db.csv", ios::app);
   if (!file.is_open()) {
       cerr << "Error: Unable to open file for writing" << endl;
       return;
   }
   for (const auto& vec : featureVectors) {
       file << label;
       for (double feature : vec) {
           file << "," << feature;
       }
       file << endl;
   }
   file.close();
}


int main() {
   VideoCapture cap(0);
   if (!cap.isOpened()) {
       cerr << "Error: Unable to open the camera" << endl;
       return -1;
   }


   namedWindow("Original Video", WINDOW_AUTOSIZE);
   namedWindow("Thresholded Video", WINDOW_AUTOSIZE);
   namedWindow("Cleaned Video", WINDOW_AUTOSIZE);
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
      
       Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
       Mat cleaned;
       morphologyEx(thresholded, cleaned, MORPH_OPEN, kernel);
       morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel);
      
       Mat labels, stats, centroids;
       int numLabels = connectedComponentsWithStats(cleaned, labels, stats, centroids);
      
       Mat regionMap = Mat::zeros(labels.size(), CV_8UC3);
       vector<Vec3b> colors(numLabels);
       vector<vector<double>> featureVectors;


       Point2f imageCenter(frame.cols / 2.0f, frame.rows / 2.0f);


       for (int i = 1; i < numLabels; i++) {
           int area = stats.at<int>(i, CC_STAT_AREA);
           int x = stats.at<int>(i, CC_STAT_LEFT);
           int y = stats.at<int>(i, CC_STAT_TOP);
           int width = stats.at<int>(i, CC_STAT_WIDTH);
           int height = stats.at<int>(i, CC_STAT_HEIGHT);


           bool touchesBoundary = (x <= 0 || y <= 0 || (x + width) >= frame.cols || (y + height) >= frame.rows);
           Point2f regionCentroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
           float distanceToCenter = norm(regionCentroid - imageCenter);


           if (area > 1000 && !touchesBoundary && distanceToCenter < 200) {
               colors[i] = randomColor(rng);
               regionMap.setTo(colors[i], labels == i);
               computeRegionFeatures(labels, i, stats, centroids, regionMap, featureVectors);
           }
       }


       imshow("Original Video", frame);
       imshow("Thresholded Video", thresholded);
       imshow("Cleaned Video", cleaned);
       imshow("Region Map", regionMap);


       char key = waitKey(30);
       if (key == 27) {
           break;
       } else if (key == 'N' || key == 'n') {
           string label;
           cout << "Enter object label: ";
           cin >> label;
           saveFeatureVectors(featureVectors, label);
           cout << "Feature vectors saved for label: " << label << endl;
       }
   }


   cap.release();
   destroyAllWindows();
   return 0;
}
