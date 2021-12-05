#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    CascadeClassifier face_classifier;
    Mat frame, grayframe;
    Mat storm_frame;
    vector<Rect> faces;
    int i;

    // open the webcam
    VideoCapture cap("Faces.mp4");
    VideoCapture storm_cap("background.mp4");

    // valid check
    if (!cap.isOpened() || !storm_cap.isOpened()) 
    {
        cout << "Could not open video" << endl;
        return -1;
    }

    // face detection configuration
    face_classifier.load("haarcascade_frontalface_alt.xml");

    while (true)
    {
        // get a new frame from webcam
        cap >> frame;
        storm_cap >> storm_frame; // gif라서 프레임수는 같은데 바뀌는게 5배 차이남
        storm_cap >> storm_frame;
        storm_cap >> storm_frame;
        storm_cap >> storm_frame;
        storm_cap >> storm_frame;
        storm_cap >> storm_frame;

        // resize(frame, frame, Size(640, 360));
        

        if (frame.empty())
            break;

        // convert captured frame to gray scale
        cvtColor(frame, grayframe, COLOR_BGR2GRAY);

        face_classifier.detectMultiScale(
            grayframe,
            faces,
            1.1,         // increase search scale by 10%  each pass
            8,           // merge groups of 3 detections
            0,           // not used for a new cascade
            Size(10, 10) // min size
        );

        // ************* f mode *******************
        // calc nearest, farthest
        int nearest_size = 0, farthest_size = 0;
        int nearest_idx = 0, farthest_idx = 0;
        if (faces.size())
        {
            nearest_size = farthest_size = faces[0].width * faces[0].height;
            for (int iter = 1; iter < faces.size(); iter++)
            {
                if (nearest_size < faces[iter].width * faces[iter].height)
                {
                    nearest_size = faces[iter].width * faces[iter].height;
                    nearest_idx = iter;
                }

                if (farthest_size > faces[iter].width * faces[iter].height)
                {
                    farthest_size = faces[iter].width * faces[iter].height;
                    farthest_idx = iter;
                }
            }
        }

        // Mat fgMask, fgImg;
        Mat fgMask, fgImg, faceDetection;
        Mat result, bgModel, fgModel, foreground;
        // foreground = Mat(image.size(), CV_8UC3, Scalar(255, 255, 255));
        fgMask = Mat(frame.size(), CV_8UC1, Scalar(0, 0, 0));
        frame.copyTo(faceDetection);
        // draw the results
        for (i = 0; i < faces.size(); i++)
        {
            Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point tr(faces[i].x, faces[i].y);

            Point zoom_lb(faces[i].x + (int)(faces[i].width*1.1), faces[i].y + (int)(faces[i].height*1.1));
            Point zoom_tr(faces[i].x - (int)(faces[i].width*0.1), faces[i].y - (int)(faces[i].height*0.1));

            if (i == nearest_idx)
                rectangle(faceDetection, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
            else if (i == farthest_idx)
                rectangle(faceDetection, lb, tr, Scalar(255, 0, 0), 3, 4, 0);
            else
            {
                rectangle(faceDetection, lb, tr, Scalar(0, 235, 255), 3, 4, 0);
            }
            // 그랩컷 느림; 
            grabCut(frame, result, Rect(zoom_lb, zoom_tr), bgModel, fgModel, 2, GC_INIT_WITH_RECT);
            compare(result, GC_PR_FGD, result, CMP_EQ);
            fgMask += result;
            
            // 사각형 마스크 따기 
            // rectangle(fgMask, Rect(zoom_lb, zoom_tr), Scalar(255), -1, 8 , 0);
        }
        // ************* f mode *******************


        // 그랩 컷 사용 코드 
        // cvtColor(fgMask, fgMask, CV_GRAY2BGR); 
        frame.copyTo(fgImg,fgMask);
        resize(storm_frame, storm_frame, Size(frame.cols, frame.rows));
        fgMask = 255 - fgMask;
        Mat for_bg;
        storm_frame.copyTo(for_bg, fgMask);
        for_bg += fgImg;
        imshow("in grapcut img", for_bg);
        // imshow("result", fgImg);

        // inRange를 써봅시다
        // 1) 마스크에 의해서 이미지를 먼저 추출 
        // frame.copyTo(fgImg, fgMask);

        // // 2) 추출한 이미지 YcrCb로 변환하고, 피부색 필터 적용해서 inRanged_mask 겟
        // Mat inRanged_mask;
        // cvtColor(fgImg, fgImg, CV_BGR2YCrCb);
        // inRange(fgImg, Scalar(20, 133, 77), Scalar(220, 173, 127), inRanged_mask);

        // // 3) 얻어낸 살색 마스크로 인물 따내고 YCrCb -> BGR
        // Mat inRanged_img;
        // frame.copyTo(inRanged_img, inRanged_mask);

        // 새로운 배경
        // resize(storm_frame, storm_frame, Size(frame.cols, frame.rows));
        // inRanged_mask = 255 - inRanged_mask;

        // Mat for_bg;
        // storm_frame.copyTo(for_bg,inRanged_mask);
        // for_bg += inRanged_img;
        // imshow("in ranged img", for_bg); // origin video size 720 x 1280
        

        // compare(result, GC_PR_FGD, result, CMP_EQ);
        imshow("Face Detection", faceDetection);
        // waitKey();
        if (waitKey(33) == 27)
            break;
    }
    return 0;
}