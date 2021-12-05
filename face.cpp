#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    CascadeClassifier face_classifier; 
    Mat frame, grayframe; 
    vector<Rect> faces;
    int i;
    // open the webcam 
    VideoCapture cap(0);
    // check if we succeeded 
    if (!cap.isOpened()) {
        cout << "Could not open camera" << endl;
        return -1; 
    }
    // face detection configuration
    face_classifier.load("haarcascade_frontalface_alt.xml");
    while (true) {
        // get a new frame from webcam  
        cap >> frame;
        //convert captured frame to gray scale 
        cvtColor(frame, grayframe, COLOR_BGR2GRAY);
    face_classifier.detectMultiScale( grayframe,faces, 1.1, 3, 0, Size(30, 30)); 

    for (i = 0; i < faces.size(); i++) {
        Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height); 
        Point tr(faces[i].x, faces[i].y);
        rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
    }
    imshow("Face Detection", frame); 
    imshow("grayFace Detection", grayframe); 
    if (waitKey(33) == 27) break;
    }
    return 0;
}