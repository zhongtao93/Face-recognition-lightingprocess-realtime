#include "stdafx.h"
#pragma once
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//找出矩阵中的最大值或最小值，输入MAX，或MIN
double MatMaxMin(Mat im,String flag="MAX");
//高斯滤波
Mat gaussianfilter(Mat img,double sigma0,double sigma1,double shift1=0,double shift2=0);
//行列卷积
Mat gaussianfilter(Mat img,double sigma0,double sigma1,double shift1,double shift2);
/*不同参数设置
	0  1    0    0  0  0  0 0  0.0  0 0    % no normalization
    1  0.2  1   -2  0  0  0 10 0.09 1 6;  % default setting
    2  1    1   -2  0  0  0 10 0.09 1 6;   % no gamma
    3  0.2  0    0  0  0  0 10 0.09 1 6;   % no DoG
    4  0.2  1   -2  0  0  0 0  0.09 1 6;   % no equalization
    5  0.2  1   -2  0  0  0 -10  0.09 1 6; % no tanh compression*/
Mat preproc2( const Mat &img,double gamma=0.2,double sigma0=1,double sigma1=-2,double mask=0,double do_norm=10);