#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

// 정의한 함수에 대한 선언
int basic_method(Mat, int, int); // basic method를 구현한 함수


int main() {
    Mat image1 = imread("family.jpeg", 0);
    Mat image2 = imread("adaptive_1.jpg", 0);
    Mat image3 = imread("adaptive.png", 0);
    Mat dst1, dst2, dst3;
    
    threshold(image1, dst1, 0, 255, THRESH_BINARY | THRESH_OTSU);
  
    adaptiveThreshold(image2, dst2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 85, 15);
    
    adaptiveThreshold(image3, dst3, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 10);

    imshow("finger_print", dst1);
    imshow("adaptive_1", dst2);
    imshow("adaptive", dst3);
    
    waitKey(0);
    return 0;
}


    // finger_print.png에 Otsu’s algorithm 적용
  // 2-2. adaptive_1.jpg에 대한 처리
    // blur(image2, image2, Size(3, 3)); // smoothing - 1
    // 조명에 의한 영향이 어느정도 있을 것으로 판단되어 smoothing을 사용하는 것이 좋을 것이라 판단하였으나
    // smoothing을 수행하면 mask의 사이즈를 작게 설정하더라도 작은 글씨의 자세함을 저하시키는 경향이 커서 코드에서 제외시킴    // 2-3. adaptive.png에 대한 처리