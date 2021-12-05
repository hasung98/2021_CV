#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

Mat gamma_transformation(Mat input) { // intensity transformation
    Mat gamma_img;
    Mat hsv_channels[3];

    cvtColor(input, gamma_img, CV_BGR2HSV);
    split(gamma_img, hsv_channels);
    hsv_channels[2].convertTo(hsv_channels[2], CV_32F);
    normalize(hsv_channels[2], hsv_channels[2], 0, 1, NORM_MINMAX);

    for(int i=0; i<gamma_img.rows; i++) {
        for(int j=0; j<gamma_img.cols; j++) {
            hsv_channels[2].at<float>(i, j) = 2.5 * pow(hsv_channels[2].at<float>(i, j), 2.5); // value of gamma as 2.5
        }
    }
    normalize(hsv_channels[2], hsv_channels[2], 0, 255, NORM_MINMAX);
    hsv_channels[2].convertTo(hsv_channels[2], CV_8U);

    merge(hsv_channels, 3, gamma_img);
    cvtColor(gamma_img, gamma_img, CV_HSV2BGR);
    return gamma_img;
}

int main(){

    Mat QR = imread("mid.png", 1);
    Mat current_QR = QR.clone(); 
    imshow("QR", QR);
    
    while(1){
        int mode;
        mode = waitKey(0);
        //Exit
        if(mode == 27){ // input = esc
            break;
        }
        
        if(mode == 103){ // input = g
            current_QR = gamma_transformation(current_QR);
            imshow("QR", current_QR);
        }
        
    }
    return 0;
}