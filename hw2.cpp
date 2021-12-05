#include <opencv2/opencv.hpp> 
#include <iostream>
using namespace cv; 
using namespace std;

int main() {
    Mat image;
    image = imread("lena.png");

    Mat negative_img = image.clone();
    Mat f_img, log_img;
    Mat gamma_img;
    double c = 1.5;

    //negative 변환
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            negative_img.at<uchar>(i, j) = 255 - image.at<uchar>(i, j);

    //log 변환
    image.convertTo(f_img, CV_32F);
	f_img = abs(f_img);
	log(f_img, f_img);
	normalize(f_img, f_img, 0, 255, NORM_MINMAX); 
	convertScaleAbs(f_img, log_img, c); 

    //gamma 변환
    MatIterator_<uchar> it, end;
    float gamma = 1.0;
	unsigned char pix[256];
    for (int i = 0; i < 256; i++)
		pix[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    gamma_img = image.clone();
    
    for (it = gamma_img.begin<uchar>(), end = gamma_img.end<uchar>(); it != end; it++) 
		*it = pix[(*it)]; 
	
	
    //이미지가 없을때 출력
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return 0; 
    }

    imshow("lena", image);
    imshow("Negative transformation", negative_img);
    imshow("Log transformation", log_img);
    imshow("Gamma transformation", gamma_img);

    waitKey(0); 
    return 0; 
}