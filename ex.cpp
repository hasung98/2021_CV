
#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

// 정의한 함수에 대한 선언
int basic_method(Mat, int, int); // basic method를 구현한 함수


int main() {
    Mat image1 = imread("finger_print.png", 0);
    Mat image2 = imread("adaptive_1.jpg", 0);
    Mat image3 = imread("adaptive.png", 0);
    
    int thresh_image1 = basic_method(image1, 120, 10); // finger_print.png를 위한 임계값을 Basic method를 사용하여 설정
    Mat dst1, dst2, dst3;
    
    // 2. 이미지에 대한 처리를 수행하는 부분
    // 2-1. Finger.png에 대한 처리
    threshold(image1, dst1, thresh_image1, 255, THRESH_BINARY);
    
    // 2-2. adaptive_1.jpg에 대한 처리
    // blur(image2, image2, Size(3, 3)); // smoothing - 1
    // 조명에 의한 영향이 어느정도 있을 것으로 판단되어 smoothing을 사용하는 것이 좋을 것이라 판단하였으나
    // smoothing을 수행하면 mask의 사이즈를 작게 설정하더라도 작은 글씨의 자세함을 저하시키는 경향이 커서 코드에서 제외시킴
    adaptiveThreshold(image2, dst2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 85, 15);
    
    // 2-3. adaptive.png에 대한 처리
    adaptiveThreshold(image3, dst3, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 10);

    // 처리한 결과를 출력
    imshow("finger_print", dst1);
    imshow("adaptive_1", dst2);
    imshow("adaptive", dst3);
    
    waitKey(0);
    return 0;
}


// 임계값을 산정하는 Basic method를 구현한 함수
// 초기에 설정한 T 값을 첫번째 인자로 전달
// T에 대한 차이의 임계값을 두번쨰 인자로 전달
int basic_method(Mat img, int init_T, int init_dif)
{
    int high_cnt=0, low_cnt=0, high_sum=0, low_sum=0;
    int thresh_T = init_T; // 초기 T을 가져옴
    int thresh_dif = init_dif; // T값의 차이에 대한 임계값을 가져옴
    Mat image = img.clone(); // 사용할 이미지를 deepcopy하여 사용
    
    // 이미지의 각 픽셀에 대하여 접근
    // 이미지의 픽셀들을 임계값을 기준으로 두 그룹으로 나눔
    while(1) {
        for(int j=0 ; j<image.rows ; j++) {
            for(int i=0 ; i<image.cols ; i++) {
                if(image.at<uchar>(j, i) < init_T) {
                    low_sum += image.at<uchar>(j, i);
                    low_cnt++;
                }
                else {
                    high_sum += image.at<uchar>(j, i);
                    high_cnt++;
                }
            }
        }
        // 이전의 T갑과 새롭게 산출한 T값의 차이를 비교
        if(abs(thresh_T - (((low_sum / low_cnt) + (high_sum / high_cnt)) / 2.0f)) < thresh_dif) {
            break;
        }
        else {
            thresh_T = (((low_sum / low_cnt) + (high_sum / high_cnt)) / 2.0f);
            low_cnt = high_cnt = low_sum = high_sum = 0; // 각 변수들의 값을 초기화
        }
    }
    return thresh_T; // 산출한 T값을 반환
}