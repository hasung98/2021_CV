#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

/*
 프로그램에서 사용하는 함수에 대한 선언부
 */
void face_detection(Mat frame, vector<Rect>& faces, CascadeClassifier fc)
{
    Mat grayframe; // 입력에 대한 Gray-scale 이미지
    cvtColor(frame, grayframe, COLOR_BGR2GRAY);

    // 해당 frame에서 face detection 수행
    fc.detectMultiScale(
        grayframe,
        faces,
        1.1, // increase search scale by 10% each pass
        3,   // merge groups of three detections
        0,   // not used for a new cascade
        Size(30, 30) //min size
    );
}
Mat merge(Mat frame, Mat background, Rect focus)
{
    Mat result, bgModel, fgModel, foreground;
    
    grabCut(frame, result, focus, bgModel, fgModel, 5, GC_INIT_WITH_RECT);
    compare(result, GC_PR_FGD, result, CMP_EQ);
    foreground = background.clone();
    frame.copyTo(foreground, result);
    return foreground; // 결과값을 반환
}                              // 인자로 전달한 두 이미지를 merge하여 그 결과를 반환

/*
 전체적인 프로그램의 수행흐름을 기술하는 main function
 */
int main()
{
    VideoCapture cap(0);
    // 웹캠과의 정상적인 연결 여부를 확인
    if (!cap.isOpened()) {
        cout << "Could not open camera" << endl;
        return -1;
    }
    int fps = cap.get(CV_CAP_PROP_FPS); // webcam source의 fps를 구함
    CascadeClassifier face_classifier; // 얼굴인식을 위한 cascade classifier 정의
    Mat frame, grayframe;              // 얼굴인식에 사용하는 frame
    vector<Rect> faces;                // 인식한 얼굴영역을 저장하는 vector
    
    // Tracking에 필요한 변수들의 선언
    Mat m_backproj, hsv;
    Mat m_model3d;
    Rect m_rc;
    float hrange[] = {0, 180};
    float vrange[] = {0, 255};
    float srange[] = {0, 255};
    const float* ranges[] = {hrange, vrange, srange};
    int channels[] = {0, 1, 2};
    int hist_sizes[] = {16, 16, 16};
    int count = 0; // 업데이트 시점 기준으로 지나간 프레임 수를 저장
    
    // 프로그램의 흐름을 통제하기 위한 변수들
    Rect focus; // Face detection을 통해서 설정된 관심영역
    bool updated = false;  // 관심영역(roi)의 갱신 여부를 확인하는 Boolean 변수
    bool tracking = false; // tracking 수행 여부를 지정하는 Boolean 변수
    
    // 배경이미지 처리를 위한 변수
    Mat background = imread("dog.png"); // 합성에 사용할 배경이미지 로딩
    Mat result;
    
    
    /*
     Tracking을 위한 관심영역을 설정하는 부분
     -> 이 과정에서 관심영역을 제대로 설정하지 못하는 경우의 수도 고려해주어야 함.
     -> 아얘 얼굴 자체를 인식하지 못하는 상황이 발생하는 경우에 대한 처리
     */
    cap >> frame; // 첫번째 프레임을 읽어옴
    resize(frame, frame, Size(500, 300)); // 연산량을 줄이기 위한 목적
    face_classifier.load("haarcascade_frontalface_alt.xml"); // face detection을 위한 xml 파일로딩
    face_detection(frame, faces, face_classifier); // 얼굴이 인식되는 영역에 대한 정보를 수집
    focus = faces[0];
    tracking = false;
    
    
    /*
     loop문을 통해서 다음의 동작을 반복함.
     */
    while(true)
    {
        cvtColor(frame, hsv, COLOR_BGR2HSV); // frame의 color model을 HSV로 변경
        
        // 매 10프레임마다 관심영역을 업데이트
        // 갱신과정에서 얼굴 인식에 대한 영역 정보를 제대로 생성하지 못한 경우
        // -> 다시 갱신을 진행함.
        if(count > 10) {
            face_detection(frame, faces, face_classifier);
            // 인식한 얼굴이 존재하는 경우
            if(faces.size() > 0) {
                updated = true;
                focus = faces[0]; // 산출한 관심영역에 대한 값을 전달
                tracking = false;     // 새롭게 히스토그램을 계산해야하므로 Tracking 비활성화
                count = 0; // frame count값을 다시 0으로 초기화
            }
        }
        
        // 설정된 관심영역에 대한 히스토그램을 계산하는 부분
        if(updated) {
            Rect rc = focus;
            Mat mask = Mat::zeros(rc.height, rc.width, CV_8U);
            ellipse(mask, Point(rc.width / 2, rc.height / 2), Size(rc.width / 2, rc.height / 2), 0, 0, 360, 255, CV_FILLED);
            Mat roi(hsv, rc);
            calcHist(&roi, 1, channels, mask, m_model3d, 3, hist_sizes, ranges);
            m_rc = rc;
            
            updated = false; // update값을 false로 초기화, 설정한 기간동안은 업데이트하지 않도록
            tracking = true;       // tracking 값을 true로 초기화
        }
        
        // 다음 프레임을 읽어옴 - Tracking의 수행을 위함
        cap >> frame;
        resize(frame, frame, Size(500, 300)); // 연산량을 줄이기 위해서 이미지를 resize함.
        if(frame.empty()) break; // 더 이상 읽을 프레임이 존재하지 않는 경우
        
        /*
         MeanShfit Tracking 수행
         -> 어차피 매 10 프레임마다 관심영역을 갱신하기 때문에
         -> Camshift를 사용하지 않아도 객체의 크기를 반영할 수 있음.
         */
        if(tracking) {
            calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);
            meanShift(m_backproj, m_rc, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
            // rectangle(frame, m_rc, Scalar(255, 0, 0), 3);
            result = merge(frame, background, m_rc);
            imshow("Project3", result);
        }
        
        count++; // 하나의 프레임에 대한 처리가 끝났으므로 frame 값을 1증가
        
        // 수행결과를 출력하는 부분
        waitKey(1000 / fps);
    }
    return 0;
}






/*
 프로그램 상에서 사용하는 함수에 대한 정의부
 */


/*
 face detection을 수행하고 해당 결과를 main 함수 내에서 정의하여 사용하고 있는
 faces vector에 영역정보를 저장
 */


/*
 기존에 설정한 배경이미지와 Tracking한 결과를 처리하여서 처리된 영상결과 값을 반환함.
 */
