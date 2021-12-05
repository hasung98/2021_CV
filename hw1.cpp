#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv; using namespace std;
int main() {
    Mat frame; 
    VideoCapture cap;
    // check if file exists. if none program ends 
    if (cap.open("background.mp4") == 0) {
        cout << "no such file!" << endl;
        waitKey(0); 
    }
    double fps = cap.get(CAP_PROP_FPS);
    double time_in_msec = cap.get(CAP_PROP_POS_MSEC);
    int total_frames = cap.get(CAP_PROP_FRAME_COUNT); 
    int curr_frames = 0;

    while (time_in_msec < 3000){
        cap >> frame;
        if (frame.empty()) {
            cout << "end of video" << endl;
            break; 
        }
        time_in_msec = cap.get(CAP_PROP_POS_MSEC);
        curr_frames = cap.get(CAP_PROP_POS_FRAMES);
        cout << "frames: " << curr_frames << " / " << total_frames << endl;

        imshow("video", frame);
        waitKey(1000/fps); 
    }
}