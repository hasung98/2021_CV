#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

Mat negative_transformation(Mat input) { // intensity transformation
    Mat negetive_img; 
    Mat hsv_channels[3];

    cvtColor(input, negetive_img, CV_BGR2HSV);
    split(negetive_img, hsv_channels);

    for(int i=0; i<negetive_img.rows; i++) {
        for(int j=0; j<negetive_img.cols; j++) {
            hsv_channels[2].at<uchar>(i, j) = 255 - hsv_channels[2].at<uchar>(i, j);
        }
    }
    merge(hsv_channels, 3, negetive_img);
    cvtColor(negetive_img, negetive_img, CV_HSV2BGR);
    return negetive_img;
}

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

Mat histogram_equalization(Mat input) {
    Mat hist_equalized_image;
    Mat hsv_channels[3];

    cvtColor(input, hist_equalized_image, CV_BGR2HSV);
    split(hist_equalized_image, hsv_channels);
    equalizeHist(hsv_channels[2], hsv_channels[2]);
    merge(hsv_channels, 3, hist_equalized_image);
    cvtColor(hist_equalized_image, hist_equalized_image, CV_HSV2BGR);
    return hist_equalized_image;
}

Mat color_slicing(Mat input) { // Hue channel
    Mat slice_img;
    Mat hsv_channels[3];

    cvtColor(input, slice_img, CV_BGR2HSV);

    split(slice_img, hsv_channels);
    for(int i=0; i<slice_img.rows; i++) {
        for(int j=0; j<slice_img.cols; j++) {
            int h = hsv_channels[0].at<uchar>(i, j);
    
            if(h <= 9 || h >= 23) {//Hue value: 9<hue<23 ¾Æ´Ò¶§ 
                hsv_channels[1].at<uchar>(i, j) = 0; 
            }
        }
    }
    merge(hsv_channels, 3, slice_img);
    cvtColor(slice_img, slice_img, CV_HSV2BGR);
    return slice_img;
}

Mat color_conversion(Mat input) {
    Mat conversion_img;
    Mat hsv_channels[3];

    cvtColor(input, conversion_img, CV_BGR2HSV);
    split(conversion_img, hsv_channels); 
    for(int i = 0; i < conversion_img.rows; i++) {
        for(int j = 0; j < conversion_img.cols; j++) {
            int h = hsv_channels[0].at<uchar>(i, j);
            if(h <= 129)
                hsv_channels[0].at<uchar>(i, j) += 50; 
            else 
                hsv_channels[0].at<uchar>(i, j) -= 129;
        }
    }

    merge(hsv_channels, 3, conversion_img);
    cvtColor(conversion_img, conversion_img, CV_HSV2BGR);
    return conversion_img;
}

Mat average_filtering(Mat input) {
    Mat avg_img;
    Mat hsv_channels[3];

    cvtColor(input, avg_img, CV_BGR2HSV);
    split(avg_img, hsv_channels);
    blur(hsv_channels[2], hsv_channels[2], Size(9, 9));

    merge(hsv_channels, 3, avg_img);
    cvtColor(avg_img, avg_img, CV_HSV2BGR);
    return avg_img;
}

Mat white_balancing(Mat input) {
    Mat white_img = input.clone();
    Mat bgr_channels[3];
    split(input, bgr_channels);

    double avg;
    int sum,temp,i, j, c;

    for(c = 0; c < white_img.channels(); c++){
        sum = 0;
        avg = 0.0f;
        for(i = 0; i < white_img.rows; i++){
            for(j = 0; j < white_img.cols; j++){
                sum += bgr_channels[c].at<uchar>(i,j);
            }
        }
        avg = sum / (white_img.rows * white_img.cols);

        for(i = 0; i < white_img.rows; i++){
            for(j = 0; j < white_img.cols; j++){
                temp = (128/avg) * bgr_channels[c].at<uchar>(i,j);
                if(temp > 255)
                    bgr_channels[c].at<uchar>(i, j) = 255;
                else 
                    bgr_channels[c].at<uchar>(i, j) = temp;
            }
        }
    }
    merge(bgr_channels, 3, white_img); 
    return white_img;
} 

int main(){

    Mat lena = imread("lena.png", 1);
    Mat current_lena = lena.clone(); 
    imshow("lena", lena);

    Mat colorful = imread("colorful.jpg", 1); 
    Mat current_colorful = colorful.clone();
    imshow("colorful", colorful);
  
    Mat balancing = imread("balancing.jpg", 1);
    Mat current_balancing = balancing.clone();
    imshow("balancing", balancing);
    
    while(1){
        int mode;
        mode = waitKey(0);
        //Exit
        if(mode == 27){ // input = esc
            break;
        }
        
        // lena.png mode
        if(mode == 110){  // input = n
            current_lena = negative_transformation(current_lena);
            imshow("lena", current_lena);
        }
        if(mode == 103){ // input = g
            current_lena = gamma_transformation(current_lena);
            imshow("lena", current_lena);
        }
        if(mode == 104){ // input = h
            current_lena = histogram_equalization(current_lena);
            imshow("lena", current_lena);
        }
        
        // colorful.jpg mode
       if(mode == 115){ // input = s
            current_colorful = color_slicing(current_colorful);
            imshow("colorful", current_colorful);
        }
        if(mode == 99){ // input = c
            current_colorful = color_conversion(current_colorful);
            imshow("colorful", current_colorful);
        }
        
        // balancing.jpg mode
        if(mode == 97){ // input = a
            current_balancing = average_filtering(current_balancing);
            imshow("balancing", current_balancing);
        }
        if(mode == 119){ // input = w
            current_balancing = white_balancing(current_balancing);
            imshow("balancing", current_balancing);
        }
        //Reset 
        if(mode == 114){ // input = r 
            current_lena = lena; 
            imshow("lena", current_lena);

            current_colorful = colorful;
            imshow("colorful", current_colorful); 
            
            current_balancing = balancing;
            imshow("balancing", current_balancing);
        }
        
    }
    return 0;
}