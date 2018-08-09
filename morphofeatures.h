#ifndef MORPHOFEATURES_H
#define MORPHOFEATURES_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Morphology
{

public:
    cv::Mat getCorners(const cv::Mat &image);  // 角点检测函数
    void drawOnImage(const cv::Mat &binary, cv::Mat &image);  // 在角点处标记圆圈

    // 以下构造四种不同的结构元素用来检测灰度图像的角点
    MorphoFeatures():threshold(-1), cross(5,5,CV_8U,cv::Scalar(0)), diamond(5,5,CV_8U,cv::Scalar(1)), square(5,5,CV_8U,cv::Scalar(1)), x(5,5,CV_8U,cv::Scalar(0))
    {
        // 5*5的十字形元素
        for (int i=0; i<5; i++) {
            cross.at<uchar>(2,i)= 1;
            cross.at<uchar>(i,2)= 1;
        }

        // 5*5的菱形元素
        diamond.at<uchar>(0,0)= 0;
        diamond.at<uchar>(0,1)= 0;
        diamond.at<uchar>(1,0)= 0;
        diamond.at<uchar>(4,4)= 0;
        diamond.at<uchar>(3,4)= 0;
        diamond.at<uchar>(4,3)= 0;
        diamond.at<uchar>(4,0)= 0;
        diamond.at<uchar>(4,1)= 0;
        diamond.at<uchar>(3,0)= 0;
        diamond.at<uchar>(0,4)= 0;
        diamond.at<uchar>(0,3)= 0;
        diamond.at<uchar>(1,4)= 0;

        // 5*5的x型元素
        for (int i=0; i<5; i++)
        {
            x.at<uchar>(i,i)= 1;
            x.at<uchar>(4-i,i)= 1;
        }
    }
}

#endif // MORPHOFEATURES_H
