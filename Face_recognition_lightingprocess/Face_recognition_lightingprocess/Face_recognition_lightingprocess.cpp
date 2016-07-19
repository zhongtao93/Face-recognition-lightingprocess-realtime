// Face_recognition_lightingprocess.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#pragma once
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
using namespace cv;
using namespace std;

int gauss(float x[],float y[],int length,float sigma);
Mat gaussianfilter(Mat img,double sigma0,double sigma1,double shift1,double shift2);
Mat preproc2( const Mat &img,double gamma,double sigma0,double sigma1,double mask,double do_norm);

//找出矩阵中的最大值或最小值，输入MAX，或MIN
double MatMaxMin(Mat im,String flag="MAX")
{
	double value=im.ptr<float>(0)[0];
	if(flag=="MAX")
	{
		for(int i=0;i<im.rows;i++)
			for(int j=0;j<im.cols;j++)
				if(im.ptr<float>(i)[j]>value)
					value=im.ptr<float>(i)[j];
		return value;
	}
	else if(flag=="MIN")
	{
		for(int i=0;i<im.rows;i++)
			for(int j=0;j<im.cols;j++)
				if(im.ptr<float>(i)[j]<value)
					value=im.ptr<float>(i)[j];
		return value;
	}
	return -1;
}
//高斯滤波
Mat gaussianfilter(Mat img,double sigma0,double sigma1,double shift1=0,double shift2=0)
{
	int i,j;
	sigma0=(float)sigma0;
	sigma1=(float)sigma1;
	shift1=(float) shift1;
	shift2=(float)shift2;
	Mat img2=img;
	Mat img3=img;
	Mat  imgResult;

	//将数据存入横向高斯模板中
	int rowLength = (int) (floor(3.0*sigma0+0.5-shift1)-ceil(-3.0*sigma0-0.5-shift1)+1); 
	int rowBegin = (int) ceil(-3.0*sigma0-0.5-shift1);
	float rowArray[30],Gx[30]; 
	for(i = 0;i < rowLength; i++) 
	{
		rowArray[i] = rowBegin + i;
	}
	gauss(rowArray,Gx,rowLength, sigma0);	
	Mat kx = Mat( 1,rowLength , CV_32F); //转换成mat类型
	float *pData1=kx.ptr<float>(0);
	for ( i = 0; i < rowLength; i++)
    {
		pData1[i]=Gx[i];
    }
	//将数据存入纵向高斯模板中
	int colLength = (int) (floor(3.0*sigma1+0.5-shift2)-ceil(-3.0*sigma1-0.5-shift2)+1);
	int colBegin = (int) ceil(-3.0*sigma1-0.5-shift2);
	float colArray[30],Gy[30]; 
	for( i=0;i<colLength;i++) 
	{
		colArray[i] = colBegin + i;
	}
	gauss(colArray,Gy,colLength, sigma1);
	Mat ky = Mat( colLength,1,CV_32F);
	float *pData2;
	for ( i = 0; i < colLength; i++)
    {
		pData2=ky.ptr<float>(i);
		pData2[0]=Gy[i];
    }
	filter2D(img, img2, img.depth(), kx,Point(-1,-1));
	filter2D(img2, imgResult, img2.depth(), ky,Point(-1,-1));
	return imgResult;

}
//行列卷积
int gauss(float x[],float y[],int length,float sigma)
{
	int i;
	float sum=0.0;
	for(i=0;i<length;i++)
	{
		x[i]=exp(-pow(x[i],2)/(2*pow(sigma,2)));
		sum += x[i] ;
	}
	for(i=0;i<length;i++)
	{
		y[i]=x[i]/sum;
	}
	return 1;
}


Mat preproc2( const Mat &img,double gamma=0.2,double sigma0=1,double sigma1=-2,double mask=0,double do_norm=10)
{	Mat imT1,imT2;
	int rows=img.rows;
	int cols=img.cols;
	Mat im=img;
	int b=floor(3*abs(sigma1));//左右扩充边缘的距离
	Mat imtemp(Size(cols+2*b,rows+2*b),CV_32F,Scalar(0));//保存扩充的图形
	Mat imtemp2(Size(cols,rows),CV_32F,Scalar(0));
	float s=0.0;
	//Gamma correct input image to increase local contrast in shadowed regions.
	if(gamma==0)
	{
		double impixeltemp=0;
		double Max=MatMaxMin(im,"MAX");//等价于max(1,max(max(im)))
		for(int i=0;i<rows;i++)
			for(int j=0;j<cols;j++)
			{
				impixeltemp=log(im.ptr<float>(i)[j]+Max/256);
				im.ptr<float>(i)[j]=impixeltemp;
			}
	}
	else
	{
		for(int i=0;i<rows;i++)
			for(int j=0;j<cols;j++)
				im.ptr<float>(i)[j]=pow(im.ptr<float>(i)[j],gamma);
	}
	float *pData1;
	//run prefilter, if any
	if(sigma1)
	{
		double border=1;
		if(border)	//add extend-as-constant image border to reduce 
					//boundary effects
		{
			for(int i=0;i<rows+2*b-1;i++)
			{
				pData1=imtemp.ptr<float>(i);
				for(int j=0;j<cols+2*b-1;j++){
					//中间
					if(i>=b&&i<im.rows+b&&j>=b&&j<im.cols+b)
						pData1[j]=im.ptr<float>(i-b)[j-b];
					//左上
					else if(i<b&&j<b)
						pData1[j]=im.ptr<float>(0)[0];
					//右上
					else if(i<b&&j>=im.cols+b&&j<cols+2*b)
						pData1[j]=im.ptr<float>(0)[cols-1];
					//左下
					else if(i>=im.rows+b&&i<rows+2*b&&j<b)
						pData1[j]=im.ptr<float>(rows-1)[0];
					//右下
					else if(i>=im.rows+b&&j>=im.cols+b)
						pData1[j]=im.ptr<float>(im.rows-1)[im.cols-1];
					//上方
					else if(i<b&&j>=b&&j<im.cols+b)
						pData1[j]=im.ptr<float>(0)[j-b];
					//下方
					else if(i>=im.rows+b&&j>=b&&j<im.cols+b)
						pData1[j]=im.ptr<float>(im.rows-1)[j-b];
					//左方
					else if(j<b&&i>=b&&i<im.rows+b)
						pData1[j]=im.ptr<float>(i-b)[0];
					//右方
					else if(j>=im.cols+b&&i>=b&&i<im.rows+b)
						pData1[j]=im.ptr<float>(i-b)[im.cols-1];/**/
						}			
				}
		}

		if(sigma1>0)
		//Build difference of gaussian filter and apply it to image.
		{
			;//70行，先跳过
		}
		
		else
		{
			if(sigma0>0)
			{   imT1=gaussianfilter(imtemp,sigma0,sigma0);
				imT2=gaussianfilter(imtemp,-sigma1,-sigma1);
				imtemp=imT1-imT2;
				//imtemp=gaussianfilter(imtemp,sigma0,sigma0)-gaussianfilter(imtemp,-sigma1,-sigma1);
			}
			else 
				imtemp = imtemp-gaussianfilter(imtemp,-sigma1,-sigma1);
		}

		if(border)
		{
			//再取回中间部分
				for(int i=0;i<rows;i++)
				{
						pData1=im.ptr<float>(i);
						for(int j=0;j<cols;j++)
							pData1[j]=imtemp.ptr<float>(i+b)[j+b];
				}
		}
	//	test=im.ptr<float>(19)[19];
	}

	/*
	% Global contrast normalization. Normalizes the spread of output
    % values. The mean is near 0 so we don't bother to subtract
    % it. We use a trimmed robust scatter measure for resistance to
    % outliers such as specularities and image borders that have
    % different values from the main image.  Usually trim is about
    % 10.
	*/

	if(do_norm)
	{
		double a=0.1;
		double trim=abs(do_norm);

		//im = im./mean(mean(abs(im).^a))^(1/a);
		imtemp2=abs(im);

		//cvPow(&im,&im,a)//为每个元素求pow
		for(int i=0;i<rows;i++)
		{
				pData1=imtemp2.ptr<float>(i);//imtemp2为零矩阵
				for(int j=0;j<cols;j++)
					pData1[j]=pow(imtemp2.ptr<float>(i)[j],a);
		}	

		//求平均值s
		s=0.0;
		for(int i=0;i<rows;i++)
			{
				pData1=imtemp2.ptr<float>(i);
					for(int j=0;j<cols;j++)
						s+=imtemp2.ptr<float>(i)[j];
			}
		s/=(im.rows*im.cols);
		double temp=pow(s,1/a);
		for(int i=0;i<rows;i++)
				{
						pData1=im.ptr<float>(i);
						for(int j=0;j<cols;j++)
							pData1[j]=pData1[j]/temp;//点除
				}	

		//im = im./mean(mean(min(trim,abs(im)).^a))^(1/a);
		imtemp2=abs(im);
		for(int i=0;i<rows;i++)
				{
						pData1=imtemp2.ptr<float>(i);
						for(int j=0;j<cols;j++)
							if(pData1[j]>trim)
							pData1[j]=trim;//min(trim,abs(im))
				}
		//cvPow(&im,&im,a);////为每个元素求pow
		for(int i=0;i<rows;i++)
		{
				pData1=imtemp2.ptr<float>(i);
				for(int j=0;j<cols;j++)
					pData1[j]=pow(pData1[j],a);
		}
		//求平均值
		s=0.0;
		for(int i=0;i<rows;i++)
			{
				pData1=imtemp2.ptr<float>(i);
					for(int j=0;j<cols;j++)
						s+=pData1[j];
			}
		s/=(im.rows*im.cols);
		temp=pow(s,1/a);//
		for(int i=0;i<rows;i++)
				{
						pData1=im.ptr<float>(i);
						for(int j=0;j<cols;j++)
							pData1[j]=pData1[j]/temp;//点除
				}

		if(do_norm>0)
		{//im = trim*tanh(im/trim);
			for(int i=0;i<rows;i++)
				{
					pData1=im.ptr<float>(i);
					for(int j=0;j<cols;j++)
						pData1[j]=trim*tanh(pData1[j]/trim);
				}
		}

	}
	//归一化处理
	double Min;
	Min=MatMaxMin(im,"MIN");//找到矩阵的最小值
	for(int i=0;i<rows;i++)
	{
		pData1=im.ptr<float>(i);
		for(int j=0;j<cols;j++)
			pData1[j]+=Min;
	}
	//im.convertTo(im, CV_32F, 1.0/255.0);
	
	normalize(im,im,0,255,NORM_MINMAX);
	cout<<"0-1:"<<im.ptr<float>(0)[1]<<endl;
	//normalize(im,im,0,255,NORM_MINMAX);
	/*	for(int i=0;i<rows;i++)
	{
		pData1=im.ptr<float>(i);
		for(int j=0;j<cols;j++)
			pData1[j]*=255;
	}*/
	im.convertTo(im,CV_8UC1);
	cout<<"0-255:"<<im.ptr<unsigned>(0)[1]<<endl;
	return im;
}



