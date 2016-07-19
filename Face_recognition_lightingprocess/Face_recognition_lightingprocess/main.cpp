
#include "stdafx.h"
#pragma once
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include"Face_recognition_lightingprocess.h"
#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
using namespace cv;
using namespace std;

double cases[11]={
	//0,1,0,0,0,0,0,0,0.0,0,0 % no normalization
	1,0.2,1,-2,0,0,0,10,0.09,1,6//default setting

};
int main()
{
	
	double gamma=cases[1];//gamma parameter
	double sigma0=cases[2];//inner Gaussian size
	double sigma1=cases[3];//outer Gaussian size
	double sx=cases[4];// x offset of centres of inner and outer filter
	double sy=cases[5];// y offset of centres of inner and outer filter
	double mask=cases[6];//mask
	double do_norm=cases[7];//Normalize the spread of output values
	double shift[2]={sx,sy};
	Mat im1=imread("02463d254.pgm",CV_LOAD_IMAGE_GRAYSCALE);
	Mat im1n=Mat(im1.rows,im1.cols,CV_32F);
	im1.convertTo(im1n,CV_32F);
	TickMeter tm;     
	tm.start();  

	im1n = preproc2(im1n);
	

	tm.stop();
	cout<<"count="<<tm.getCounter()<<",process time="<<tm.getTimeMilli()<<endl; //13-14ms×óÓÒ
	imshow("im1",im1);
	imshow("im1n",im1n);

	waitKey(0);
	return 0;
}