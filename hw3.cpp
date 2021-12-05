#include <opencv2/opencv.hpp> 
#include <iostream>
using namespace cv; 
using namespace std;

int main() {
    Mat image;
    Mat left_image;
    Mat right_image;
    image = imread("lena.png", 0);
    blur(image, left_image, Size(7, 7));
    Rect rect(0, 0, (image.size().width/2), (image.size().height));
    left_image = left_image(rect);
    Rect rect2((image.size().width/2), 0, (image.size().width/2), (image.size().height));
    right_image = image(rect2);
    Mat dst;
    hconcat(left_image, right_image, dst); // 이미지 합쳐주는 명령어

    Mat image2; 
    image2 = imread("moon.png",0);
    Mat laplacian, abs_laplacian, sharpening;
    Rect rect3(0, 0, (image2.size().width/2), (image2.size().height));
    Rect rect4((image2.size().width/2), 0, (image2.size().width/2), (image2.size().height));
    Laplacian(image2, laplacian, CV_16S, 1, 1, 0); 
    convertScaleAbs(laplacian, abs_laplacian); 
    sharpening = abs_laplacian + image2;
    Mat dst2;
    hconcat(image2(rect3), sharpening(rect4), dst2);

    Mat image3;
    Mat median;
    image3 = imread("saltnpepper.png",0);
    medianBlur(image3, median, 9);

    imshow("lena", image);
    imshow("lena_filtered", dst);
    imshow("moon", image2);
    imshow("moon_filtered", dst2);
    imshow("SaltAndPepper", image3);
    imshow("SaltAndPepper_filtered", median);

    waitKey(0); 
    return 0; 
}