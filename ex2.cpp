#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

/*
 ���α׷����� ����ϴ� �Լ��� ���� �����
 */
void face_detection(Mat frame, vector<Rect>& faces, CascadeClassifier fc)
{
    Mat grayframe; // �Է¿� ���� Gray-scale �̹���
    cvtColor(frame, grayframe, COLOR_BGR2GRAY);

    // �ش� frame���� face detection ����
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
    return foreground; // ������� ��ȯ
}                              // ���ڷ� ������ �� �̹����� merge�Ͽ� �� ����� ��ȯ

/*
 ��ü���� ���α׷��� �����帧�� ����ϴ� main function
 */
int main()
{
    VideoCapture cap(0);
    // ��ķ���� �������� ���� ���θ� Ȯ��
    if (!cap.isOpened()) {
        cout << "Could not open camera" << endl;
        return -1;
    }
    int fps = cap.get(CV_CAP_PROP_FPS); // webcam source�� fps�� ����
    CascadeClassifier face_classifier; // ���ν��� ���� cascade classifier ����
    Mat frame, grayframe;              // ���νĿ� ����ϴ� frame
    vector<Rect> faces;                // �ν��� �󱼿����� �����ϴ� vector
    
    // Tracking�� �ʿ��� �������� ����
    Mat m_backproj, hsv;
    Mat m_model3d;
    Rect m_rc;
    float hrange[] = {0, 180};
    float vrange[] = {0, 255};
    float srange[] = {0, 255};
    const float* ranges[] = {hrange, vrange, srange};
    int channels[] = {0, 1, 2};
    int hist_sizes[] = {16, 16, 16};
    int count = 0; // ������Ʈ ���� �������� ������ ������ ���� ����
    
    // ���α׷��� �帧�� �����ϱ� ���� ������
    Rect focus; // Face detection�� ���ؼ� ������ ���ɿ���
    bool updated = false;  // ���ɿ���(roi)�� ���� ���θ� Ȯ���ϴ� Boolean ����
    bool tracking = false; // tracking ���� ���θ� �����ϴ� Boolean ����
    
    // ����̹��� ó���� ���� ����
    Mat background = imread("dog.png"); // �ռ��� ����� ����̹��� �ε�
    Mat result;
    
    
    /*
     Tracking�� ���� ���ɿ����� �����ϴ� �κ�
     -> �� �������� ���ɿ����� ����� �������� ���ϴ� ����� ���� ������־�� ��.
     -> �ƾ� �� ��ü�� �ν����� ���ϴ� ��Ȳ�� �߻��ϴ� ��쿡 ���� ó��
     */
    cap >> frame; // ù��° �������� �о��
    resize(frame, frame, Size(500, 300)); // ���귮�� ���̱� ���� ����
    face_classifier.load("haarcascade_frontalface_alt.xml"); // face detection�� ���� xml ���Ϸε�
    face_detection(frame, faces, face_classifier); // ���� �νĵǴ� ������ ���� ������ ����
    focus = faces[0];
    tracking = false;
    
    
    /*
     loop���� ���ؼ� ������ ������ �ݺ���.
     */
    while(true)
    {
        cvtColor(frame, hsv, COLOR_BGR2HSV); // frame�� color model�� HSV�� ����
        
        // �� 10�����Ӹ��� ���ɿ����� ������Ʈ
        // ���Ű������� �� �νĿ� ���� ���� ������ ����� �������� ���� ���
        // -> �ٽ� ������ ������.
        if(count > 10) {
            face_detection(frame, faces, face_classifier);
            // �ν��� ���� �����ϴ� ���
            if(faces.size() > 0) {
                updated = true;
                focus = faces[0]; // ������ ���ɿ����� ���� ���� ����
                tracking = false;     // ���Ӱ� ������׷��� ����ؾ��ϹǷ� Tracking ��Ȱ��ȭ
                count = 0; // frame count���� �ٽ� 0���� �ʱ�ȭ
            }
        }
        
        // ������ ���ɿ����� ���� ������׷��� ����ϴ� �κ�
        if(updated) {
            Rect rc = focus;
            Mat mask = Mat::zeros(rc.height, rc.width, CV_8U);
            ellipse(mask, Point(rc.width / 2, rc.height / 2), Size(rc.width / 2, rc.height / 2), 0, 0, 360, 255, CV_FILLED);
            Mat roi(hsv, rc);
            calcHist(&roi, 1, channels, mask, m_model3d, 3, hist_sizes, ranges);
            m_rc = rc;
            
            updated = false; // update���� false�� �ʱ�ȭ, ������ �Ⱓ������ ������Ʈ���� �ʵ���
            tracking = true;       // tracking ���� true�� �ʱ�ȭ
        }
        
        // ���� �������� �о�� - Tracking�� ������ ����
        cap >> frame;
        resize(frame, frame, Size(500, 300)); // ���귮�� ���̱� ���ؼ� �̹����� resize��.
        if(frame.empty()) break; // �� �̻� ���� �������� �������� �ʴ� ���
        
        /*
         MeanShfit Tracking ����
         -> ������ �� 10 �����Ӹ��� ���ɿ����� �����ϱ� ������
         -> Camshift�� ������� �ʾƵ� ��ü�� ũ�⸦ �ݿ��� �� ����.
         */
        if(tracking) {
            calcBackProject(&hsv, 1, channels, m_model3d, m_backproj, ranges);
            meanShift(m_backproj, m_rc, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
            // rectangle(frame, m_rc, Scalar(255, 0, 0), 3);
            result = merge(frame, background, m_rc);
            imshow("Project3", result);
        }
        
        count++; // �ϳ��� �����ӿ� ���� ó���� �������Ƿ� frame ���� 1����
        
        // �������� ����ϴ� �κ�
        waitKey(1000 / fps);
    }
    return 0;
}






/*
 ���α׷� �󿡼� ����ϴ� �Լ��� ���� ���Ǻ�
 */


/*
 face detection�� �����ϰ� �ش� ����� main �Լ� ������ �����Ͽ� ����ϰ� �ִ�
 faces vector�� ���������� ����
 */


/*
 ������ ������ ����̹����� Tracking�� ����� ó���Ͽ��� ó���� ������ ���� ��ȯ��.
 */
