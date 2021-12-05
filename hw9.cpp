#include <opencv2/opencv.hpp> 
#include <iostream>


using namespace cv;
using namespace std;

int main()
{   
    int count = 0; // store drawn rectangles
    int area; // 찾은 boundrect의 넓이
    vector<vector<Point> > contours; // store count info
    vector<Vec4i>hierarchy;

    VideoCapture cap; // 영상을 불러오는 객체
    Mat frame, gray_frame; // 비디오 파일을 구성하는 프레임, 특정 프레임에 대한 gray-scale
    
    Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
    Mat image, foregroundMask, backgroundImg, foregroundImg;

    CascadeClassifier face_classifier; // face detection을 위한 classifier
    vector<Rect> faces; // 얼굴 정보 식별에 대한 정보를 저장
    int i; // 반복문 변수
    
    int validate_open; // 영상파일 불러오기 성공여부
    int key_input; // 키보드 입력값
    int lastKey = 0; // 가장 최근에 입력한 Key의 값을 업데이트, 초기값은 null을 의미하는 0으로 초기화
    int f_count = 0; 
    face_classifier.load("haarcascade_frontalface_alt.xml");
    validate_open = cap.open("Faces.mp4"); // Faces.mp4 파일을 읽음
    double fps = cap.get(CAP_PROP_FPS);   // 영상의 출력을 위해 프레임 값을 읽어들임
    
    // 예외처리 - 해당 파일이 없거나 파일 로딩을 실패한 경우
    // -> 프로그램의 실행을 중지시킴
    if(validate_open == 0) {
        cout << "Error! : No such file!" << endl;
        return -1;
    }
    
    // 읽어온 비디오 파일을 출력함
    while(1){
        key_input = waitKey(fps); // 사용자로부터 키 입력을 대기

        if(key_input != -1) {
            lastKey = key_input; 
        }
        cap >> frame; // 프레임을 가져옴
        if(frame.empty()) break; // loop를 빠져나옴
        
        cvtColor(frame, gray_frame, CV_BGR2GRAY); // 읽어온 frame을 gray-scale로 변경 - 내부적인 연산에 활용하기 위해서
        // f 입력하면 얼굴 인식 
        if(lastKey == 'f') {
            f_count++;
            putText(frame, format("Key input: %d", lastKey), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, 4);
            face_classifier.detectMultiScale(
                gray_frame,
                faces, 
                1.1, 
                5, 
                0, 
                Size(70, 70),
                Size(100, 100)
            ); 
            for (i = 0; i < faces.size(); i++) {
                Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point tr(faces[i].x, faces[i].y);
                rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 4, 0);
            }
            face_classifier.detectMultiScale(
                gray_frame,
                faces,
                1.1, // 각 frame의 scale을 10% 확대함
                4,   // 최소 3개 이상의 인식이 나타나야 얼굴로 인식함
                0,   // not used for a new cascade
                Size(55, 55), // 인식하는 최소 사이즈
                Size(60, 60)
            );
            // 인식한 얼굴을 식별할 수 있도록 bounding rect를 그려냄
            for (i = 0; i < faces.size(); i++) {
                Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point tr(faces[i].x, faces[i].y);
                rectangle(frame, lb, tr, Scalar(100, 255, 255), 3, 4, 0);
            }
        
            face_classifier.detectMultiScale(
                gray_frame,
                faces,
                1.1, // 각 frame의 scale을 10% 확대함
                4,   // 최소 3개 이상의 인식이 나타나야 얼굴로 인식함
                0,   // not used for a new cascade
                Size(35, 35), // 인식하는 최소 사이즈
                Size(41, 41)  // 인식하는 최대 사이즈
            );
            // 인식한 얼굴을 식별할 수 있도록 bounding rect를 그려냄
            for (i = 0; i < faces.size(); i++) {
                Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point tr(faces[i].x, faces[i].y);
                rectangle(frame, lb, tr, Scalar(255, 255, 100), 3, 4, 0);
            }
        }
        // b입력 배경제거 
        else if(lastKey == 66 || lastKey == 98) {
            while(true){
                cap >> frame;
                if (foregroundMask.empty()) 
                    foregroundMask.create(frame.size(), frame.type());
                    
                bg_model->apply(frame, foregroundMask);
                GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5); 
                threshold(foregroundMask, foregroundMask, 30, 255, THRESH_BINARY); 
                foregroundImg = Scalar::all(0);
                frame.copyTo(foregroundImg, foregroundMask);
                imshow("Faces", frame);
                waitKey(33);
            }
           
        }
        // g입력시 배경 삽입
        else if(lastKey == 71 || lastKey == 103) {
            
        }
        // 그 이외에 프로그램 내에서 정의되지 않은 입력이 들어온 경우
        else {
            putText(frame, format("Key input: %d", lastKey), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 4);
        }
        // 원본 frame을 window Faces에 출력
        imshow("Faces", frame);
    }
    return 0;
}