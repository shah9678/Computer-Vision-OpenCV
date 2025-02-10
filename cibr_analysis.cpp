/* CS5330 PRCV 
ADIT SHAH
JHEEL KAMDAR
DATE: 02/04/2025 */


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <vector>
#include <iostream>
#include <dirent.h>

namespace fs = std::filesystem;

// Load ResNet model
cv::dnn::Net loadResNetModel(const std::string& modelPath) {
    return cv::dnn::readNetFromONNX(modelPath);
}

// Extract features using ResNet
cv::Mat extractResNetFeatures(cv::dnn::Net& net, const cv::Mat& image) {
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    return net.forward();
}

// Compute color histogram for an image
cv::Mat computeHistogram(const cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    cv::Mat hist;
    for (int i = 0; i < 3; i++) {
        cv::Mat tempHist;
        cv::calcHist(&channels[i], 1, 0, cv::Mat(), tempHist, 1, &histSize, &histRange);
        tempHist /= cv::sum(tempHist)[0]; // Normalize
        if (i == 0)
            hist = tempHist;
        else
            cv::hconcat(hist, tempHist, hist);
    }
    return hist;
}

// Compute similarity using histogram intersection
float compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2) {
    return cv::compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
}

int main() {
    std::string train_path = "/Users/aditshah/Desktop/TestTask7";
    std::string query_image_path = "/Users/aditshah/Desktop/TestTask7/image (3).jpeg";
    std::string resnet_model_path = "/Users/aditshah/Desktop/DNN-Example/resnet18-v2-7.onnx";
    int top_n_results = 5; // Define how many results to display
    
    // Load ResNet model
    cv::dnn::Net resnet = loadResNetModel(resnet_model_path);
    
    // Load query image
    cv::Mat query = cv::imread(query_image_path);
    cv::Mat queryHist = computeHistogram(query);
    cv::Mat queryFeatures = extractResNetFeatures(resnet, query);
    
    std::vector<std::pair<std::string, float>> results;
    DIR* dir = opendir(train_path.c_str());
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fileName = entry->d_name;
        if (fileName != "." && fileName != "..") {
            std::string filePath = train_path + "/" + fileName;
            cv::Mat img = cv::imread(filePath);
            if (!img.empty()) {
                cv::Mat imgHist = computeHistogram(img);
                cv::Mat imgFeatures = extractResNetFeatures(resnet, img);
                
                // Compute similarity
                float histSim = compareHistograms(queryHist, imgHist);
                float featureSim = cv::norm(queryFeatures, imgFeatures, cv::NORM_L2);
                float combinedSim = histSim - featureSim; // Adjust weight as needed
                
                results.emplace_back(filePath, combinedSim);
            }
        }
    }
    closedir(dir);
    
    // Sort results by similarity score
    std::sort(results.begin(), results.end(), 
        [](const std::pair<std::string, float>& a, 
           const std::pair<std::string, float>& b) { 
            return a.second > b.second; 
        });
    
    // Display top N matches
    std::cout << "Top " << top_n_results << " Matches:\n";
    for (size_t i = 0; i < std::min(results.size(), static_cast<size_t>(top_n_results)); i++) {
        std::cout << results[i].first << " : " << results[i].second << "\n";
        cv::Mat matchedImg = cv::imread(results[i].first);
        cv::imshow("Match " + std::to_string(i + 1), matchedImg);
    }
    cv::waitKey(0);
    return 0;
}
