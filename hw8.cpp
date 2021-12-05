#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    VideoCapture capture("background.mp4");
    
    Mat image, avg; 
    Mat background;
    Mat foregroundMask; 

    int cnt = 2;
    int fps = capture.get(CAP_PROP_FPS);

    vector<vector<Point> > contours;
    vector<Vec4i>hierarchy;
    int count = 0;

    capture >> avg;
    cvtColor(avg, avg, CV_BGR2GRAY); 
    
    while(cnt <= 10){
        if(!capture.read(image)) break;
        cvtColor(image, image, CV_BGR2GRAY);
        add(image / cnt, avg*(cnt - 1) / cnt, avg);
        cnt++;
    }
    
    background = avg.clone(); 

    while(true){
        if(!capture.read(image)) break;

        cvtColor(image, image, CV_BGR2GRAY);
        absdiff(background, image, foregroundMask);
        threshold(foregroundMask, foregroundMask, 20, 255, CV_THRESH_BINARY);
        findContours(foregroundMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        //defining bounding rectangle
        vector<Rect> boundRect(contours.size()); 
        for (int i = 0 ; i < contours.size() ; i++)
            boundRect[i] = boundingRect(Mat(contours[i]));

        //draw rectangles on the contours
        for (int i = 0; i < contours.size(); i++){
            if(boundRect[i].area() >= 400){
                rectangle(image, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 255), 2, 8, 0);
                count++;
            }
        }
        putText(image, format("number of moving objects: %d", count), Point(10, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(256), 4); 
        count = 0;

        imshow("background image", background);
        imshow("result(x,y)",foregroundMask);
        imshow("final result", image);
        
        waitKey(33);
    }
    return 0;
}