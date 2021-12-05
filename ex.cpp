
#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

// ������ �Լ��� ���� ����
int basic_method(Mat, int, int); // basic method�� ������ �Լ�


int main() {
    Mat image1 = imread("finger_print.png", 0);
    Mat image2 = imread("adaptive_1.jpg", 0);
    Mat image3 = imread("adaptive.png", 0);
    
    int thresh_image1 = basic_method(image1, 120, 10); // finger_print.png�� ���� �Ӱ谪�� Basic method�� ����Ͽ� ����
    Mat dst1, dst2, dst3;
    
    // 2. �̹����� ���� ó���� �����ϴ� �κ�
    // 2-1. Finger.png�� ���� ó��
    threshold(image1, dst1, thresh_image1, 255, THRESH_BINARY);
    
    // 2-2. adaptive_1.jpg�� ���� ó��
    // blur(image2, image2, Size(3, 3)); // smoothing - 1
    // ���� ���� ������ ������� ���� ������ �ǴܵǾ� smoothing�� ����ϴ� ���� ���� ���̶� �Ǵ��Ͽ�����
    // smoothing�� �����ϸ� mask�� ����� �۰� �����ϴ��� ���� �۾��� �ڼ����� ���Ͻ�Ű�� ������ Ŀ�� �ڵ忡�� ���ܽ�Ŵ
    adaptiveThreshold(image2, dst2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 85, 15);
    
    // 2-3. adaptive.png�� ���� ó��
    adaptiveThreshold(image3, dst3, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 10);

    // ó���� ����� ���
    imshow("finger_print", dst1);
    imshow("adaptive_1", dst2);
    imshow("adaptive", dst3);
    
    waitKey(0);
    return 0;
}


// �Ӱ谪�� �����ϴ� Basic method�� ������ �Լ�
// �ʱ⿡ ������ T ���� ù��° ���ڷ� ����
// T�� ���� ������ �Ӱ谪�� �ι��� ���ڷ� ����
int basic_method(Mat img, int init_T, int init_dif)
{
    int high_cnt=0, low_cnt=0, high_sum=0, low_sum=0;
    int thresh_T = init_T; // �ʱ� T�� ������
    int thresh_dif = init_dif; // T���� ���̿� ���� �Ӱ谪�� ������
    Mat image = img.clone(); // ����� �̹����� deepcopy�Ͽ� ���
    
    // �̹����� �� �ȼ��� ���Ͽ� ����
    // �̹����� �ȼ����� �Ӱ谪�� �������� �� �׷����� ����
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
        // ������ T���� ���Ӱ� ������ T���� ���̸� ��
        if(abs(thresh_T - (((low_sum / low_cnt) + (high_sum / high_cnt)) / 2.0f)) < thresh_dif) {
            break;
        }
        else {
            thresh_T = (((low_sum / low_cnt) + (high_sum / high_cnt)) / 2.0f);
            low_cnt = high_cnt = low_sum = high_sum = 0; // �� �������� ���� �ʱ�ȭ
        }
    }
    return thresh_T; // ������ T���� ��ȯ
}