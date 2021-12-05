#include <opencv2/opencv.hpp> 
#include <iostream>
using namespace cv; 
using namespace std;

int main() {

    Mat image; 
    image = imread("star.png",0);
    Mat laplacian, abs_laplacian, sharpening;
    Laplacian(image, laplacian, CV_16S, 1, 1, 0); 
    convertScaleAbs(laplacian, abs_laplacian); 
    sharpening = abs_laplacian + image;

    imshow("star", image);
    imshow("star_filtered", sharpening);


    waitKey(0); 
    return 0; 
}