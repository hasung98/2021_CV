#include <opencv2/opencv.hpp> 
#include <iostream>


using namespace cv;
using namespace std;

int main()
{   
    int count = 0; // store drawn rectangles
    int area; // ã�� boundrect�� ����
    vector<vector<Point> > contours; // store count info
    vector<Vec4i>hierarchy;

    VideoCapture cap; // ������ �ҷ����� ��ü
    Mat frame, gray_frame; // ���� ������ �����ϴ� ������, Ư�� �����ӿ� ���� gray-scale
    
    Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
    Mat image, foregroundMask, backgroundImg, foregroundImg;

    CascadeClassifier face_classifier; // face detection�� ���� classifier
    vector<Rect> faces; // �� ���� �ĺ��� ���� ������ ����
    int i; // �ݺ��� ����
    
    int validate_open; // �������� �ҷ����� ��������
    int key_input; // Ű���� �Է°�
    int lastKey = 0; // ���� �ֱٿ� �Է��� Key�� ���� ������Ʈ, �ʱⰪ�� null�� �ǹ��ϴ� 0���� �ʱ�ȭ
    int f_count = 0; 
    face_classifier.load("haarcascade_frontalface_alt.xml");
    validate_open = cap.open("Faces.mp4"); // Faces.mp4 ������ ����
    double fps = cap.get(CAP_PROP_FPS);   // ������ ����� ���� ������ ���� �о����
    
    // ����ó�� - �ش� ������ ���ų� ���� �ε��� ������ ���
    // -> ���α׷��� ������ ������Ŵ
    if(validate_open == 0) {
        cout << "Error! : No such file!" << endl;
        return -1;
    }
    
    // �о�� ���� ������ �����
    while(1){
        key_input = waitKey(fps); // ����ڷκ��� Ű �Է��� ���

        if(key_input != -1) {
            lastKey = key_input; 
        }
        cap >> frame; // �������� ������
        if(frame.empty()) break; // loop�� ��������
        
        cvtColor(frame, gray_frame, CV_BGR2GRAY); // �о�� frame�� gray-scale�� ���� - �������� ���꿡 Ȱ���ϱ� ���ؼ�
        // f �Է��ϸ� �� �ν� 
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
                1.1, // �� frame�� scale�� 10% Ȯ����
                4,   // �ּ� 3�� �̻��� �ν��� ��Ÿ���� �󱼷� �ν���
                0,   // not used for a new cascade
                Size(55, 55), // �ν��ϴ� �ּ� ������
                Size(60, 60)
            );
            // �ν��� ���� �ĺ��� �� �ֵ��� bounding rect�� �׷���
            for (i = 0; i < faces.size(); i++) {
                Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point tr(faces[i].x, faces[i].y);
                rectangle(frame, lb, tr, Scalar(100, 255, 255), 3, 4, 0);
            }
        
            face_classifier.detectMultiScale(
                gray_frame,
                faces,
                1.1, // �� frame�� scale�� 10% Ȯ����
                4,   // �ּ� 3�� �̻��� �ν��� ��Ÿ���� �󱼷� �ν���
                0,   // not used for a new cascade
                Size(35, 35), // �ν��ϴ� �ּ� ������
                Size(41, 41)  // �ν��ϴ� �ִ� ������
            );
            // �ν��� ���� �ĺ��� �� �ֵ��� bounding rect�� �׷���
            for (i = 0; i < faces.size(); i++) {
                Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
                Point tr(faces[i].x, faces[i].y);
                rectangle(frame, lb, tr, Scalar(255, 255, 100), 3, 4, 0);
            }
        }
        // b�Է� ������� 
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
        // g�Է½� ��� ����
        else if(lastKey == 71 || lastKey == 103) {
            
        }
        // �� �̿ܿ� ���α׷� ������ ���ǵ��� ���� �Է��� ���� ���
        else {
            putText(frame, format("Key input: %d", lastKey), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 4);
        }
        // ���� frame�� window Faces�� ���
        imshow("Faces", frame);
    }
    return 0;
}