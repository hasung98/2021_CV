#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

// ������ �Լ��� ���� ����
int basic_method(Mat, int, int); // basic method�� ������ �Լ�


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


    // finger_print.png�� Otsu��s algorithm ����
  // 2-2. adaptive_1.jpg�� ���� ó��
    // blur(image2, image2, Size(3, 3)); // smoothing - 1
    // ���� ���� ������ ������� ���� ������ �ǴܵǾ� smoothing�� ����ϴ� ���� ���� ���̶� �Ǵ��Ͽ�����
    // smoothing�� �����ϸ� mask�� ����� �۰� �����ϴ��� ���� �۾��� �ڼ����� ���Ͻ�Ű�� ������ Ŀ�� �ڵ忡�� ���ܽ�Ŵ    // 2-3. adaptive.png�� ���� ó��