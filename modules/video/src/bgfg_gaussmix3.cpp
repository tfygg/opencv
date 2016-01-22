/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <float.h>
#include <iostream>

// to make sure we can use these short names
#undef K
#undef L
#undef T

// This is based on the "An Improved Adaptive Background Mixture Model for
// Real-time Tracking with Shadow Detection" by P. KaewTraKulPong and R. Bowden
// http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf
//
// The windowing method is used, but not the shadow detection. I make some of my
// own modifications which make more sense. There are some errors in some of their
// equations.
//

namespace cv
{

static const int defaultNMixtures3 = 5;
static const int defaultHistory3 = 200;
static const double defaultBackgroundRatio3 = 0.7;
static const double defaultVarThreshold3 = 2.5*2.5;
static const double defaultNoiseSigma3 = 30*0.5;
static const double defaultInitialWeight3 = 0.05;

BackgroundSubtractorMOG3::BackgroundSubtractorMOG3()
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    nmixtures = defaultNMixtures3;
    history = defaultHistory3;
    varThreshold = defaultVarThreshold3;
    backgroundRatio = defaultBackgroundRatio3;
    noiseSigma = defaultNoiseSigma3;
}

BackgroundSubtractorMOG3::BackgroundSubtractorMOG3(int _history, int _nmixtures,
                                                 double _backgroundRatio,
                                                 double _noiseSigma)
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    nmixtures = min(_nmixtures > 0 ? _nmixtures : defaultNMixtures3, 8);
    history = _history > 0 ? _history : defaultHistory3;
    varThreshold = defaultVarThreshold3;
    backgroundRatio = min(_backgroundRatio > 0 ? _backgroundRatio : 0.95, 1.);
    noiseSigma = _noiseSigma <= 0 ? defaultNoiseSigma3 : _noiseSigma;
}

BackgroundSubtractorMOG3::~BackgroundSubtractorMOG3()
{
}


void BackgroundSubtractorMOG3::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( CV_MAT_DEPTH(frameType) == CV_8U );

    // for each gaussian mixture of each pixel bg model we store ...
    // the mixture sort key (w/sum_of_variances), the mixture weight (w),
    // the mean (nchannels values) and
    // the diagonal covariance matrix (another nchannels values)
    bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + 2*nchannels), CV_32F );//初始化一个1行*XX列的矩阵  
   //该矩阵的大小是这样计算的：图像的行*列*混合模型的个数*（1（优先级）+ 1（权值）+ 2（均值 + 方差）*通道数）  参考template<typename VT> struct MixData{}
    bgmodel = Scalar::all(0);
}


template<typename VT> struct MixData
{
    float sortKey;
    float weight;
    VT mean;
    VT var;
};


static void process8uC1( const Mat& image, Mat& fgmask, double learningRate,
                         Mat& bgmodel, int nmixtures, double backgroundRatio,
                         double varThreshold, double noiseSigma )
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;
    MixData<float>* mptr = (MixData<float>*)bgmodel.data;

    const float w0 = (float)defaultInitialWeight3;
    const float sk0 = (float)(w0/(defaultNoiseSigma3*2));
    const float var0 = (float)(defaultNoiseSigma3*defaultNoiseSigma3*4);
    const float minVar = (float)(noiseSigma*noiseSigma);

    for( y = 0; y < rows; y++ )
    {
        const uchar* src = image.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        if( alpha > 0 )
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float wsum = 0;
                float pix = src[x];
                int kHit = -1, kForeground = -1;
				
			/*************************************************************
			S1: 判断像素是否有匹配的高斯函数，并进行参数更新，以及高斯函数的排序
			*************************************************************/
                for( k = 0; k < K; k++ )
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if( w < FLT_EPSILON )
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff*diff;

					/************************************************************
					S1-1: 如果该像素满足第k个高斯函数，则进行参数更新(EM但没有完全按照公式)并排序
					*************************************************************/
                    if( d2 < vT*var )		// 判断像素是否匹配的高斯函数的公式:  (pix - mu)^2 < 2.5*2.5*var (第一个高斯函数) 
                    {
                        wsum -= w;
                        float dw = alpha*(1.f - w);
                        mptr[k].weight = w + dw;			//更新第k个高斯函数的权值，公式 w=w+alfa*(1-w)
                        mptr[k].mean = mu + alpha*diff;			//更新高斯函数的均值，公式 u=u+alfa*(X-u)
                        var = max(var + alpha*(d2 - var), minVar);	//更新高斯函数的均值，公式 var1=var+alfa*(d2-var)，不能小于minVar
                        mptr[k].var = var;
                        mptr[k].sortKey = w/sqrt(var);			//因为需要按照 W/σ 的降序排列，所以将其赋值为sortkey

                        for( k1 = k-1; k1 >= 0; k1-- )    //将SortKey从大到小排序，只有第k个高斯函数的参数进行了迭代
                        {
                            if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                                break;
                            std::swap( mptr[k1], mptr[k1+1] );
                        }	

                        kHit = k1+1;		//在新列表中，像素与第 kHit 个高斯函数匹配
                        break;
                    }
                }

				/************************************************************
				S1-2: 如果该像素没有满足的高斯函数，则进行以下更新
				*************************************************************/
                if( kHit < 0 ) 
                {
                    kHit = k = min(k, K-1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = var0;
                    mptr[k].sortKey = sk0;
                }
                else
                    for( ; k < K; k++ )
                        wsum += mptr[k].weight;		//计算所有高斯函数的权值之和

			/************************************************************
			S2: 背景判断
			*************************************************************/
                float wscale = 1.f/wsum;					//权值归一化以后，前 K 个高斯函数的权值之和
                wsum = 0;
                for( k = 0; k < K; k++ )		//计算出有效像素模型个数 B, 公式 B = arg min(sum(w)>T)
                {
                    wsum += mptr[k].weight *= wscale;		
                    mptr[k].sortKey *= wscale;
                    if( wsum > T && kForeground < 0 )		//前 k 个权重之和大于阈值 T 的高斯模型，被认为是有效背景模型。
                        kForeground = k+1;					//kForeground表示: 第 k+1 以后的模型之后就判为前景了
                }

                dst[x] = (uchar)(-(kHit >= kForeground));	//匹配模型的序号< kForeground，则认为是背景。否则被认为是前景
            }
        }
        else//如果学习速率小于等于0，则没有背景更新过程
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float pix = src[x];
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )	//判断每一个像素是否有匹配的高斯模型
                {
                    if( mptr[k].weight < FLT_EPSILON )
                        break;
                    float mu = mptr[k].mean;
                    float var = mptr[k].var;
                    float diff = pix - mu;
                    float d2 = diff*diff;
                    if( d2 < vT*var )		//判断的公式
                    {
                        kHit = k;
                        break;
                    }
                }

                if( kHit >= 0 )
                {
                    float wsum = 0;
                    for( k = 0; k < K; k++ )
                    {
                        wsum += mptr[k].weight;
                        if( wsum > T )
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

                dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }
}


static void process8uC3( const Mat& image, Mat& fgmask, double learningRate,
                         Mat& bgmodel, int nmixtures, double backgroundRatio,
                         double varThreshold, double noiseSigma )
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = (float)learningRate, T = (float)backgroundRatio, vT = (float)varThreshold;
    int K = nmixtures;

    const float w0 = (float)defaultInitialWeight3;
    const float sk0 = (float)(w0/(defaultNoiseSigma3*2*sqrt(3.)));
    const float var0 = (float)(defaultNoiseSigma3*defaultNoiseSigma3*4);
    const float minVar = (float)(noiseSigma*noiseSigma);
    MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;

    for( y = 0; y < rows; y++ )
    {
        const uchar* src = image.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        if( alpha > 0 )
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                float wsum = 0;
                Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    float w = mptr[k].weight;
                    wsum += w;
                    if( w < FLT_EPSILON )
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
                    if( d2 < vT*(var[0] + var[1] + var[2]) )
                    {
                        wsum -= w;
                        float dw = alpha*(1.f - w);
                        mptr[k].weight = w + dw;
                        mptr[k].mean = mu + alpha*diff;
                        var = Vec3f(max(var[0] + alpha*(diff[0]*diff[0] - var[0]), minVar),
                                    max(var[1] + alpha*(diff[1]*diff[1] - var[1]), minVar),
                                    max(var[2] + alpha*(diff[2]*diff[2] - var[2]), minVar));
                        mptr[k].var = var;
                        mptr[k].sortKey = w/sqrt(var[0] + var[1] + var[2]);

                        for( k1 = k-1; k1 >= 0; k1-- )
                        {
                            if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                                break;
                            std::swap( mptr[k1], mptr[k1+1] );
                        }

                        kHit = k1+1;
                        break;
                    }
                }

                if( kHit < 0 ) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
                {
                    kHit = k = min(k, K-1);
                    wsum += w0 - mptr[k].weight;
                    mptr[k].weight = w0;
                    mptr[k].mean = pix;
                    mptr[k].var = Vec3f(var0, var0, var0);
                    mptr[k].sortKey = sk0;
                }
                else
                    for( ; k < K; k++ )
                        wsum += mptr[k].weight;

                float wscale = 1.f/wsum;
                wsum = 0;
                for( k = 0; k < K; k++ )
                {
                    wsum += mptr[k].weight *= wscale;
                    mptr[k].sortKey *= wscale;
                    if( wsum > T && kForeground < 0 )
                        kForeground = k+1;
                }

                dst[x] = (uchar)(-(kHit >= kForeground));
            }
        }
        else
        {
            for( x = 0; x < cols; x++, mptr += K )
            {
                Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
                int kHit = -1, kForeground = -1;

                for( k = 0; k < K; k++ )
                {
                    if( mptr[k].weight < FLT_EPSILON )
                        break;
                    Vec3f mu = mptr[k].mean;
                    Vec3f var = mptr[k].var;
                    Vec3f diff = pix - mu;
                    float d2 = diff.dot(diff);
                    if( d2 < vT*(var[0] + var[1] + var[2]) )
                    {
                        kHit = k;
                        break;
                    }
                }

                if( kHit >= 0 )
                {
                    float wsum = 0;
                    for( k = 0; k < K; k++ )
                    {
                        wsum += mptr[k].weight;
                        if( wsum > T )
                        {
                            kForeground = k+1;
                            break;
                        }
                    }
                }

                dst[x] = (uchar)(kHit < 0 || kHit >= kForeground ? 255 : 0);
            }
        }
    }

	
	//light change
	int num , total;
	double rate ;

	num = cv::countNonZero(fgmask);
	total = fgmask.total();
	rate = (double)num/(double)total;
	int first =0;
	
	MixData<Vec3f>* mptr2 = (MixData<Vec3f>*)bgmodel.data;
	if(rate>0.75)	
	{
		for( y = 0; y < rows; y++ )
		{
			const uchar* src = image.ptr<uchar>(y);
			for( x = 0; x < cols; x++ ,	mptr2 += K)
			{
				Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
				mptr2[0].mean = pix;
				mptr2[0].var = Vec3f(500, 500, 500);
			}
		}
	}
	
}

void BackgroundSubtractorMOG3::operator()(InputArray _image, OutputArray _fgmask, double learningRate)
{
    Mat image = _image.getMat();
    bool needToInitialize = nframes == 0 || learningRate >= 1 || image.size() != frameSize || image.type() != frameType;

    if( needToInitialize )
        initialize(image.size(), image.type());

    CV_Assert( image.depth() == CV_8U );
    _fgmask.create( image.size(), CV_8U );
    Mat fgmask = _fgmask.getMat();

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./min( nframes, history );
    CV_Assert(learningRate >= 0);

    if( image.type() == CV_8UC1 )
        process8uC1( image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma );
    else if( image.type() == CV_8UC3 )
        process8uC3( image, fgmask, learningRate, bgmodel, nmixtures, backgroundRatio, varThreshold, noiseSigma );
    else
        CV_Error( CV_StsUnsupportedFormat, "Only 1- and 3-channel 8-bit images are supported in BackgroundSubtractorMOG3" );
}

void BackgroundSubtractorMOG3::getBackgroundImage(OutputArray backgroundImage) const
{
	int nchannels = CV_MAT_CN(frameType);
	CV_Assert( nchannels == 3 );
	Mat meanBackground(frameSize, CV_8UC3, Scalar::all(0));		
		
	const MixData<Vec3f>* mptr = (MixData<Vec3f>*)bgmodel.data;	
	for(int row=0; row<meanBackground.rows; row++)
	{
		for(int col=0; col<meanBackground.cols; col++, mptr += nmixtures)
		{				
			Vec3f meanVal;
			float totalWeight = 0.f;
			for( int k = 0; k < nmixtures; k++ )	//计算出有效像素模型个数 B, 公式 B = arg min(sum(w)>T)
			{
				float w = mptr[k].weight;					
				if( w < FLT_EPSILON )
					break;
					
				totalWeight += w;
				meanVal += mptr[k].mean*w;			//均值(即，像素值) * 权值 (前B个高斯模型)
				if ( totalWeight> backgroundRatio)		//前 k 个权重之和大于阈值 T 的高斯模型，被认为是有效背景模型。
					break;
			}
			meanVal *= (1.f / totalWeight);			//背景的每个像素值等于  sum(均值*权值)*sum(权值),就是前B个数据的期望。
			meanBackground.at<Vec3b>(row, col) = Vec3b(meanVal);				
		}
	}

	switch(CV_MAT_CN(frameType))
	{
		case 1:
		{
			vector<Mat> channels;
			split(meanBackground, channels);
			channels[0].copyTo(backgroundImage);
				break;
		}

		case 3:
		{
			meanBackground.copyTo(backgroundImage);
				break;
		}

		default:
		CV_Error(CV_StsUnsupportedFormat, "");
	}		
}

}

/* End of file. */
