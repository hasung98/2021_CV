#include <opencv2/opencv.hpp> 
#include <iostream>
using namespace cv; 
using namespace std;

Mat drawHistogram(Mat src){ 
    Mat hist, histImage;

    // establish the number of bins 
    int i, hist_w, hist_h, bin_w, histSize; 
    float range[] = { 0, 256 };
    const float* histRange = { range };
    hist_w = 512;
    hist_h = 512;
    histSize = 16;
    bin_w = cvRound((double)hist_w / histSize);

    //draw the histogram
    histImage = Mat(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
    // compute the histograms
    // &src: input image, 1: #of src image, 0: #of channels numerated from 0 ~ channels()-1, Mat(): optional mask
    // hist: output histogram, 1: histogram dimension, &histSize: array of histogram size, &histRange: array of histogram¡¯s boundaries 
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    // Fit the histogram to [0, histImage.rows]
    // hist: input Mat, hist: output Mat, 0: lower range boundary of range normalization, histImage.rows: upper range boundary // NORM_MINMAX: normalization type, -1: when negative, the ouput array has the same type as src, Mat(): optional mask 
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (i = 0; i < histSize; i++){
        rectangle(histImage, Point(bin_w * i, hist_h), Point(bin_w * i+hist_w/histSize, hist_h - cvRound(hist.at<float>(i))), Scalar(0, 0, 0), -1);
        //cout<<Point(bin_w * i+hist_w/histSize, hist_h - cvRound(hist.at<float>(i)))<<endl;
    }
    return histImage; 
}

int main() {
    Mat image;
    Mat hist_equalized_image; 
    Mat hist_graph;
    Mat hist_equalized_graph;
    double bin[8] = {0.012, 0, 0, 0, 0.6144, 0.1236, 0.1284, 0.126};
    image = imread("moon.png", 0);
    
    equalizeHist(image, hist_equalized_image);

    hist_graph = drawHistogram(image);
    hist_equalized_graph = drawHistogram(hist_equalized_image);
    for(int i = 0; i < 8; i++){
        putText(hist_equalized_image,format("bin%d: %.4f",i+1,bin[i]),Point(0,30*(i+1)), FONT_HERSHEY_SIMPLEX, 1,Scalar(0,255,0), 4);
    }
    imshow("before", image);
    imshow("after", hist_equalized_image); 
    imshow("h1", hist_graph);
    imshow("h2", hist_equalized_graph);

    waitKey(0); 
    return 0; 
}

