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
#include <map>
#include <cmath>
#include "dirent.h"
#include <string>
#include <unistd.h>
#include <limits.h> 

using namespace std;
using namespace cv;

// Function to split a string by a delimiter
vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Load the CSV file into a map
map<string, vector<float>> loadCSV(const string& filename) {
    map<string, vector<float>> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file: " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        if (tokens.size() != 513) { 
            cerr << "Warning: Invalid line in CSV file: " << line << endl;
            continue;
        }

        string filename = tokens[0];
        vector<float> features;
        for (size_t i = 1; i < tokens.size(); ++i) {
            features.push_back(stof(tokens[i]));
        }
        data[filename] = features;
    }

    file.close();
    return data;
}

// Histogram intersection for color and texture features
float histogramIntersection(const vector<float>& hist1, const vector<float>& hist2) {
    float intersection = 0.0f;
    for (size_t i = 0; i < hist1.size(); ++i) {
        intersection += min(hist1[i], hist2[i]);
    }
    return intersection;
}

// TASK 4 TEXTURE AND SOBEL FEATURES IMPLEMENTATION //


vector<float> extractColorHistogram(const Mat& image, int bins = 8) {
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

// Extract texture histogram features using Sobel gradient magnitude
vector<float> extractTextureHistogram(const Mat& image, int bins = 8) {
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
    normalize(grad_mag, grad_mag, 0, 255, NORM_MINMAX, CV_32F);

    // Compute histogram of gradient magnitudes
    vector<float> histogram(bins, 0.0);
    int totalPixels = grad_mag.rows * grad_mag.cols;

    for (int i = 0; i < grad_mag.rows; ++i) {
        for (int j = 0; j < grad_mag.cols; ++j) {
            float magnitude = grad_mag.at<float>(i, j);
            int bin = static_cast<int>(magnitude / (256.0 / bins));
            bin = min(max(bin, 0), bins-1);
            histogram[bin] += 1.0;
        }
    }

    for (float &val : histogram) {
        val /= totalPixels;
    }
    return histogram;
}

// TASK 5 COSINE AND EUCLEDIAN DISTANCE AND DNN RESNET FEATURES IMPLEMENTATION //

float euclideanDistance(const vector<float>& v1, const vector<float>& v2) {
    float sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Function to compute cosine distance
float cosineDistance(const vector<float>& v1, const vector<float>& v2) {
    float dotProduct = 0.0;
    float norm1 = 0.0;
    float norm2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }

    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);

    if (norm1 == 0 || norm2 == 0) {
        cerr << "Error: One of the vectors has zero magnitude." << endl;
        return 1.0; // Maximum distance
    }

    float cosineSimilarity = dotProduct / (norm1 * norm2);
    return 1.0 - cosineSimilarity; // Cosine distance
}


// TASK 7 CIBR  DEPTH V2 IMPLEMENTATION //
int get_depth(cv::Mat &src, cv::Mat &dst){
    // make a DANetwork object, if you use a different network, you have
  // to include the input and output layer names
  DA2Network da_net( "./src/model_fp16.onnx" );

  // scale the network input so it's not larger than 512 on the small side
  float scale_factor = 512.0 / (src.rows > src.cols ? src.cols : src.rows);
  scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

  // set up the network input
  da_net.set_input( src, scale_factor );

  // run the network
  da_net.run_network( dst, src.size() );

  return 0;
}


int get_depth_rgb_histogram(cv::Mat &src, cv::MatND &hist, const int histsize) {
    // Compute the depth map
    cv::Mat depth;
    get_depth(src, depth);

    // Normalize depth to 0-255
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::threshold(depth, depth, 150, 255, cv::THRESH_BINARY);

    // Define histogram size (3D)
    int histSize[] = {histsize, histsize, histsize};

    // Initialize a true 3D histogram using cv::MatND
    hist = cv::MatND(3, histSize, CV_32FC1, cv::Scalar(0));

    // Loop over all pixels manually
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);    // Pointer to row i (RGB)
        uchar *depth_ptr = depth.ptr<uchar>(i);   // Pointer to row i (Depth)

        for (int j = 0; j < src.cols; j++) {
            if (depth_ptr[j] > 150){
                // Extract R, G, B values
                int B = ptr[j][0];
                int G = ptr[j][1];
                int R = ptr[j][2];

                // Compute normalized RGB values
                float divisor = R + G + B;
                divisor = divisor > 0.0 ? divisor : 1.0; // Avoid division by zero

                float r = R / divisor;
                float g = G / divisor;
                float b = B / divisor;

                // Compute bin indices
                int rindex = static_cast<int>(r * (histsize - 1) + 0.5);
                int gindex = static_cast<int>(g * (histsize - 1) + 0.5);
                int bindex = static_cast<int>(b * (histsize - 1) + 0.5);

                // Increment histogram directly at (r, g, b)
                hist.at<float>(rindex, gindex, bindex)++;
            }
        }
    }

    // Normalize the histogram so the sum is 1
    hist /= (src.rows * src.cols);

    return 0;
}
// Extract edge histogram descriptor (EHD) features
vector<float> extractEdgeHistogram(const Mat& image, int bins = 5) {
    Mat img_gray;
    if (image.channels() == 3) {
        cvtColor(image, img_gray, COLOR_BGR2GRAY);
    } else {
        img_gray = image.clone();
    }

    // Compute gradients using Sobel
    Mat grad_x, grad_y;
    Sobel(img_gray, grad_x, CV_32F, 1, 0);
    Sobel(img_gray, grad_y, CV_32F, 0, 1);

    // Compute gradient magnitude and direction
    Mat grad_mag, grad_dir;
    cartToPolar(grad_x, grad_y, grad_mag, grad_dir, true); // grad_dir in degrees [0, 360)

    // Quantize edge directions into bins
    vector<float> histogram(bins, 0.0);
    int totalPixels = grad_dir.rows * grad_dir.cols;

    for (int i = 0; i < grad_dir.rows; ++i) {
        for (int j = 0; j < grad_dir.cols; ++j) {
            float angle = grad_dir.at<float>(i, j);
            int bin = static_cast<int>(angle / (360.0 / bins));
            bin = min(max(bin, 0), bins - 1);
            histogram[bin] += 1.0;
        }
    }

    // Normalize histogram
    for (float &val : histogram) {
        val /= totalPixels;
    }

    return histogram;
}

// Combine color, texture, and edge histogram features
vector<float> extractCombinedFeatures(const Mat& image, int bins = 8) {
    vector<float> colorHist = extractColorHistogram(image, bins);
    vector<float> textureHist = extractTextureHistogram(image, bins);
    vector<float> edgeHist = extractEdgeHistogram(image, 5); // 5 bins for EHD

    // Combine features
    vector<float> combined;
    combined.reserve(colorHist.size() + textureHist.size() + edgeHist.size());
    combined.insert(combined.end(), colorHist.begin(), colorHist.end());
    combined.insert(combined.end(), textureHist.begin(), textureHist.end());
    combined.insert(combined.end(), edgeHist.begin(), edgeHist.end());

    return combined;
}

// Distance metric combining color, texture, and edge histogram features
float combinedDistance(const vector<float>& hist1, const vector<float>& hist2) {
    // Split histograms into color, texture, and edge parts
    size_t colorSize = 512; 
    size_t textureSize = 512; 
    size_t edgeSize = 5; 

    vector<float> colorHist1(hist1.begin(), hist1.begin() + colorSize);
    vector<float> colorHist2(hist2.begin(), hist2.begin() + colorSize);
    vector<float> textureHist1(hist1.begin() + colorSize, hist1.begin() + colorSize + textureSize);
    vector<float> textureHist2(hist2.begin() + colorSize, hist2.begin() + colorSize + textureSize);
    vector<float> edgeHist1(hist1.begin() + colorSize + textureSize, hist1.end());
    vector<float> edgeHist2(hist2.begin() + colorSize + textureSize, hist2.end());

    // Compute histogram intersection for color, texture, and edge
    float colorDistance = 1.0f - histogramIntersection(colorHist1, colorHist2);
    float textureDistance = 1.0f - histogramIntersection(textureHist1, textureHist2);
    float edgeDistance = 1.0f - histogramIntersection(edgeHist1, edgeHist2);

    // Combine distances with weights
    return 0.4f * colorDistance + 0.4f * textureDistance + 0.2f * edgeDistance;
}

// Function to detect faces in an image
bool detectFaces(const Mat& image) {
    // Load the pre-trained face detection model (Haar Cascade)
    CascadeClassifier face_cascade;
    if (!face_cascade.load("/Users/aditshah/Desktop/haarcascade_frontalface_alt2.xml")) {
        cerr << "Error: Could not load face cascade." << endl;
        return false;
    }

    // Convert image to grayscale for face detection
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Detect faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces);

    return !faces.empty(); // Return true if faces are detected
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <target_image> <feature_csv> <N> <feature_type>" << endl;
        return -1;
    }

    string targetImagePath = argv[1];
    string featureCSV = argv[2];
    int N = stoi(argv[3]);
    string featureType = argv[4];

    // Load the CSV file into a map
    map<string, vector<float>> featureData = loadCSV(featureCSV);

    // Check if the target image exists in the CSV file
    if (featureData.find(targetImagePath) == featureData.end()) {
        cerr << "Error: Target image not found in CSV file." << endl;
        return -1;
    }
    std::string image_path = "../" + targetImagePath;
    Mat targetImage = imread(image_path);
    // Retrieve the target image's feature vector from the CSV file
    vector<float> targetFeatures = featureData[targetImagePath];

    // Check for faces in the target image
    if (detectFaces(targetImage)) {
        cout << "Faces detected in the target image!" << endl;
    } else {
        cout << "No faces detected in the target image." << endl;
    }
    // Debug: Print target feature size
    cout << "Target feature size: " << targetFeatures.size() << endl;

    vector<pair<float, string>> distances;
    for (const auto& entry : featureData) {
        const string& filename = entry.first;
        const vector<float>& features = entry.second;

        // Skip the target image
        if (filename == targetImagePath) {
            continue;
        }

        // Ensure feature sizes match
        if (features.size() != targetFeatures.size()) {
            cerr << "Warning: Feature size mismatch for " << filename << ". Skipping." << endl;
            continue;
        }

        float distance;
        if (featureType == "combined") {
            distance = combinedDistance(targetFeatures, features);
        } else if (featureType == "resnet") {
            // Use cosine distance for ResNet features
            distance = cosineDistance(targetFeatures, features);
        } else {
            cerr << "Error: Invalid feature type: " << featureType << endl;
            return -1;
        }

        distances.emplace_back(distance, filename);
    }

    // Sort distances
    sort(distances.begin(), distances.end());

    // Display the top N matches
    int num = min(N, (int)distances.size());
    for (int i = 0; i < num; ++i) {
        cout << "Match " << i + 1 << ": " << distances[i].second << " (Distance: " << distances[i].first << ")" << endl;
    }

    // Load and display the target image
    cout<< "Target Image: " << targetImagePath << endl;
    //std::string image_path = "../" + targetImagePath;

    // Print current working directory for debugging
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        cout << "Current working directory: " << cwd << endl;
    } else {
        cerr << "Error: Could not get current working directory." << endl;
    }
    
    if (targetImage.empty()) {
        cerr << "Error: Could not load target image: " << targetImagePath << endl;
        return -1;
    }
    namedWindow("Target Image", WINDOW_AUTOSIZE);
    imshow("Target Image", targetImage);

    
    // Load and display the top N matching images
    for (int i = 0; i < num; ++i) {
        string matchImagePath = "olympus/" + distances[i].second;  // Adjusting path

        Mat matchImage = imread(matchImagePath);
        if (matchImage.empty()) {
            cerr << "Error: Could not load match image: " << matchImagePath << endl;
            continue;
        }

        string windowName = "Match " + to_string(i + 1);
        namedWindow(windowName, WINDOW_AUTOSIZE);
        imshow(windowName, matchImage);
    }

    // Wait for a key press and close the windows
    waitKey(0);
    destroyAllWindows();

    return 0;
}