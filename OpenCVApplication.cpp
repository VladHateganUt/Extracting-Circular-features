#include "stdafx.h"
#include "common.h"
#include "OpenCV/include/opencv2/highgui/highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <queue>
#include <iostream>
#include <conio.h>
#include <string.h>
#include <algorithm>    
#include <vector>

struct greater
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};



#define MODE_BRIGHT 1
#define MODE_DARK 2
#define MODE_BOTH 3

using namespace std;
using namespace cv;


Mat fastRadialSymmetry(Mat src,  double alpha, const int mode, double stdFactor, int range){

	Mat Sn = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);
	for (int e = range; e < 13; e+=2){
		int height = src.rows;
		int width = src.cols;

		bool darkCheck = false, brightCheck = false;

		if (mode == MODE_BRIGHT){
			brightCheck = true;
		}
		else if (mode == MODE_DARK){
			darkCheck = true;
		}
		else if (mode == MODE_BOTH){
			brightCheck = true;
			darkCheck = true;
		}

		

		Mat M_n = cv::Mat::zeros(src.size(), CV_64FC1);

		Mat O_n = cv::Mat::zeros(src.size(), CV_64FC1);
		cv::Mat S = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);



		Mat gradientx;
		Mat gradienty;
		Sobel(src, gradientx, CV_64FC1, 0, 1);
		Sobel(src, gradienty, CV_64FC1, 1, 0);
		for (int i = e; i < height - e; i++){
			for (int j = e; j < width - e; j++){
				Point p(i, j);

				Vec2d grad = Vec2d(gradientx.at<double>(i, j), gradienty.at<double>(i, j));
				double gradNorm = sqrt(grad.val[0] * grad.val[0] + grad.val[1] * grad.val[1]);
				if (gradNorm > 0){

					Vec2i gradPoint;
					gradPoint.val[0] = (int)round((grad[0] / gradNorm) * e);
					gradPoint.val[1] = (int)round((grad[1] / gradNorm) * e);

					if (brightCheck){
						Point pPositive;
						pPositive.x = p.x + gradPoint[0];
						pPositive.y = p.y + gradPoint[1];
						O_n.at<double>(pPositive.x, pPositive.y) = O_n.at<double>(pPositive.x, pPositive.y) + 1;
						M_n.at<double>(pPositive.x, pPositive.y) = M_n.at<double>(pPositive.x, pPositive.y) + gradNorm;
					}
					if (darkCheck){
						Point pNegative;
						pNegative.x = p.x - gradPoint[0];
						pNegative.y = p.y - gradPoint[1];
						O_n.at<double>(pNegative.x, pNegative.y) = O_n.at<double>(pNegative.x, pNegative.y) - 1;
						M_n.at<double>(pNegative.x, pNegative.y) = M_n.at<double>(pNegative.x, pNegative.y) - gradNorm;
					}
				}
			}
		}

		Mat M_nTemp = M_n.clone();
		Mat O_nTemp = O_n.clone();

		double locMinimum, locMaximum;
		O_nTemp = abs(O_n);
		minMaxLoc(O_nTemp, &locMinimum, &locMaximum);
		O_nTemp = O_n / locMaximum;

		M_nTemp = abs(M_n);
		minMaxLoc(M_nTemp, &locMinimum, &locMaximum);
		M_nTemp = M_n / locMaximum;


		cv::pow(O_nTemp, alpha, S);
		S.mul(M_nTemp);
		int kernelSize = (int)std::ceil(e / 2);
		if (kernelSize % 2 == 0){ kernelSize++; } //kernelSize trebuie sa fie impar si pozitiv

		GaussianBlur(S, S, cv::Size(kernelSize, kernelSize), e*stdFactor, e*stdFactor);
		Sn += S;
		
	}
	Mat dst = cv::Mat::zeros(src.size(), CV_64FC1);
	dst = Sn;

	return dst;
	
}

vector<Point> celule(Mat src){
	vector<Point> celMax;
	int celx = 20;
	int cely = 20;
	for (int i = celx; i <= src.rows-celx; i+=celx){
		for (int j = cely; j <= src.cols-cely; j+=cely){
			double max=-1;
			int maxX, maxY;
			for (int dx = i - celx / 2; dx < i + celx / 2; dx++){
				for (int dy = j - cely / 2; dy < j + cely / 2; dy++){
					if (src.at<double>(dx, dy) > max) {
						max = src.at<double>(dx, dy);
						maxX = dx;
						maxY = dy;
					}
				}
			}
			celMax.push_back(Point(maxY, maxX));
		}
	}

	return celMax;
}

int main(){

	cv::Mat image;

	image = cv::imread("Images/img3.jpg");

	if (!image.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
		
	if (image.channels() == 4) {
		cv::cvtColor(image, image, CV_BGRA2BGR);
	}

	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, CV_BGR2GRAY);

	cv::Mat frstImage = grayImg.clone();
	frstImage=fastRadialSymmetry(grayImg,3.1, MODE_DARK, 0.1,7);

	vector<Point> p = celule(frstImage);
	for (int i = 0; i< p.size()-1; i++){
		for (int j = i + 1; j<p.size(); j++){
			if (frstImage.at<double>(p[j]) > frstImage.at<double>(p[i])){
				Point aux = p[i];
				p[i] = p[j];
				p[j] = aux;
			}
		}
	}


	for (int i = 0; i < 5; i++){
		printf("%lf %d %d\n", frstImage.at<double>(p[i]) ,p[i].x, p[i].y);
		circle(image, Point(p[i]), 5, Scalar(255, 0, 0),-1);
	}

	cv::normalize(frstImage, frstImage, 0, 1, NORM_MINMAX); //afisarea punctelor negative
	frstImage.convertTo(frstImage, CV_8U, 255); // call the FRST
	imshow("original", image);
	imshow("Display window", frstImage);

	waitKey();
	return 0;
}