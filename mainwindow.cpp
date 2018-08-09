#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "opencv2/imgproc/imgproc.hpp"

/*-----------------PictureFunction----------------*/

void MainWindow::myShowMat(cv::Mat img)    //show Mat in Label
{
    QImage img_1= QImage((const unsigned char*)(img.data),  // Qt image structure
                       img.cols, img.rows, QImage::Format_RGB888);
//    QImage img_1= QImage((const unsigned char*)(img.data),  // Qt image structure // int bytesPerLine ???
//                       img.cols, img.rows, img.cols, QImage::Format_RGB888);
    ui->label->setPixmap(QPixmap::fromImage(img_1));
    ui->label->resize(ui->label->pixmap()->size());
}

void MainWindow::myShowShreshold()  //show shreshold_1 and shreshold_2 in textBrowser
{
    QString text_1 = QString::number(shreshold_1, 10);
    QString text_2 = QString::number(shreshold_2, 10);
    ui->textBrowser->moveCursor(QTextCursor::End);  //接收框始终定为在末尾一行
    ui->textBrowser->setText("shreShold_1 "+text_1);
    ui->textBrowser_2->moveCursor(QTextCursor::End);
    ui->textBrowser_2->setText("shreShold_2 "+text_2);
}

cv::Mat MainWindow::QImage2cvMat(QImage image)
{
    cv::Mat mat;
    // qDebug() << image.format();
    switch(image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}

/*--------------------WindowFunction---------------------*/

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
//    int nMin = 1;
//    int nMax = 10;
//    int nSingleStep = 1;

//    ui->thresholdSlider_1->setMinimum(nMin);  // 最小值
//    ui->thresholdSlider_1->setMaximum(nMax);  // 最大值
//    ui->thresholdSlider_1->setSingleStep(nSingleStep);  // 步长
//    ui->thresholdSlider_2->setMinimum(nMin);  // 最小值
//    ui->thresholdSlider_2->setMaximum(nMax);  // 最大值
//    ui->thresholdSlider_2->setSingleStep(nSingleStep);  // 步长

    ui->setupUi(this);
    ui->processButton->setEnabled(false);
    ui->cannyButton->setEnabled(false);
    ui->openButton->setEnabled(false);
    ui->closeButton->setEnabled(false);
    ui->dilateButton->setEnabled(false);
    ui->erodeButton->setEnabled(false);
    ui->getConersButton->setEnabled(false);
    ui->houghLinesButton->setEnabled(false);
    ui->houghCirclesButton->setEnabled(false);
    ui->thresholdSlider_1->setEnabled(false);
    ui->thresholdSlider_2->setEnabled(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_openImageButton_clicked()    //Open
{
    QString fileName = QFileDialog::getOpenFileName(this,
     tr("Open Image"), ".", tr("Image Files (*.png *.jpg *.bmp)"));

    //image = cv::imread(fileName.toAscii().data());//Qt4
    image = cv::imread(fileName.toLatin1().data());//Qt5

    if (image.data) {
        cv::namedWindow("Original Image");
        cv::imshow("Original Image", image);
        ui->processButton->setEnabled(true);
        ui->cannyButton->setEnabled(true);
        ui->openButton->setEnabled(true);
        ui->closeButton->setEnabled(true);
        ui->dilateButton->setEnabled(true);
        ui->erodeButton->setEnabled(true);
        ui->getConersButton->setEnabled(true);
        ui->houghLinesButton->setEnabled(true);
        ui->houghCirclesButton->setEnabled(true);
        ui->thresholdSlider_1->setEnabled(true);
        ui->thresholdSlider_2->setEnabled(true);
    }
}

void MainWindow::on_processButton_clicked()  //Process
{
    cv::flip(image,image,1); // process the image

    cv::cvtColor(image,image,CV_BGR2RGB);  // change color channel ordering

//    // ATTENTION!
//    unsigned char *image_data;
//    unsigned int last_insert_count;
//    last_insert_count = 4 - image.cols%4;
//    image_data = new unsigned char[(image.cols+last_insert_count)*image.rows];
//    //计算列数是否为4的倍数，每行补上last_insert_count个0，使行字节数为4的倍数
//    for(int i=0;i<image.rows;i++){
//        for(int j=0;j<image.cols;j++){
//            image_data[i*(image.cols+last_insert_count)+j]=image.data[i*image.cols+j];
//        }
//        for(int k=0;k<last_insert_count;k++)
//            image_data[i*(image.cols+last_insert_count)+image.cols+k]=0;
//    }

    myShowMat(image);
}

void MainWindow::on_cannyButton_clicked()  //Canny
{
    using namespace cv;

    Mat image1 = image.clone();
    Mat dst,edge,gray;

    dst.create(image1.size(), image1.type());
    cvtColor(image1, gray, CV_BGR2GRAY);
    blur(gray, edge, cv::Size(3,3));
    Canny(edge, edge, shreshold_1, shreshold_2, 3);
    dst = cv::Scalar::all(0);

    image1.copyTo(dst, edge);
    //imshow("Canny", dst);

    myShowMat(dst);
    myShowShreshold();
}

void MainWindow::on_thresholdSlider_1_valueChanged(int value)
{
    using namespace cv;

    if(value != shreshold_1)
    {
        shreshold_1 = value;

        Mat image1 = image.clone();
        Mat dst,edge,gray;

        dst.create(image1.size(), image1.type());
        cvtColor(image1, gray, CV_BGR2GRAY);
        blur(gray, edge, cv::Size(3,3));
        Canny(edge, edge, shreshold_1, shreshold_2, 3);
        dst = cv::Scalar::all(0);

        image1.copyTo(dst, edge);

        myShowMat(dst);
        myShowShreshold();
    }
}

void MainWindow::on_thresholdSlider_2_valueChanged(int value)
{
    using namespace cv;

    if(value != shreshold_2)
    {
        shreshold_2 = value;

        Mat image1 = image.clone();
        Mat dst,edge,gray;

        dst.create(image1.size(), image1.type());
        cvtColor(image1, gray, CV_BGR2GRAY);
        blur(gray, edge, cv::Size(3,3));
        Canny(edge, edge, shreshold_1, shreshold_2, 3);
        dst = cv::Scalar::all(0);

        image1.copyTo(dst, edge);

        myShowMat(dst);
        myShowShreshold();
    }
}

void MainWindow::on_dilateButton_clicked()  //Dilate
{
    using namespace cv;

    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat out;

//    QImage image1 = ui->label->pixmap()->toImage(); //Pixel Loss ???
//    Mat in = QImage2cvMat(image1);

    Mat in = image.clone();
    dilate(in, out, element);

    myShowMat(out);
}


void MainWindow::on_erodeButton_clicked()  //Erode
{
    using namespace cv;

    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat out;

//    QImage image1 = ui->label->pixmap()->toImage(); //Pixel Loss ???
//    Mat in = QImage2cvMat(image1);

    Mat in = image.clone();
    erode(in, out, element);

    myShowMat(out);
}

void MainWindow::on_openButton_clicked()  //OPEN
{
    using namespace cv;

    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat out1,out;

    Mat in = image.clone();
    erode(in, out1, element);
    dilate(out1, out, element);

    myShowMat(out);
}

void MainWindow::on_closeButton_clicked()  //CLOSE
{
    using namespace cv;

    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat out1,out;

    Mat in = image.clone();
    dilate(in, out1, element);
    erode(out1, out, element);


    myShowMat(out);
}

void MainWindow::on_getConersButton_clicked()
{
    using namespace cv;

    // 将原图像进行灰度化
    Mat image1 = image.clone();
    Mat GraySrc;
    cvtColor(image1, GraySrc, CV_BGR2GRAY);

    // 定义结构元素
    Mat CrossMat(5, 5, CV_8U, Scalar(0));	// 十字型结构元素
    Mat DiamondMat(5, 5, CV_8U, Scalar(1));		// 菱形结构元素
    Mat squareMat(5, 5, CV_8U, Scalar(1));	// 方形结构元素
    Mat XMat(5, 5, CV_8U, Scalar(0));
    // 十字形形状的结构元素
    for (int i = 0; i < 5; i++)
    {
        CrossMat.at<uchar>(2, i) = 1;
        CrossMat.at<uchar>(i, 2) = 1;
    }
    // 定义菱形形状
    DiamondMat.at<uchar>(0, 0) = 0;
    DiamondMat.at<uchar>(0, 1) = 0;
    DiamondMat.at<uchar>(1, 0) = 0;
    DiamondMat.at<uchar>(4, 4) = 0;
    DiamondMat.at<uchar>(3, 4) = 0;
    DiamondMat.at<uchar>(4, 3) = 0;
    DiamondMat.at<uchar>(4, 0) = 0;
    DiamondMat.at<uchar>(4, 1) = 0;
    DiamondMat.at<uchar>(3, 0) = 0;
    DiamondMat.at<uchar>(0, 4) = 0;
    DiamondMat.at<uchar>(0, 3) = 0;
    DiamondMat.at<uchar>(1, 4) = 0;

    // 定义X型形状
    for (int i = 0; i < 5; i++)
    {
        XMat.at<uchar>(i, i) = 1;
        XMat.at<uchar>(4 - i, i) = 1;
    }

    // 十字形对原图进行膨胀
    Mat dstImage1;
    dilate(GraySrc, dstImage1, CrossMat);
    // 菱形对上步结果进行腐蚀
    erode(dstImage1, dstImage1, DiamondMat);
    Mat dstImage2;
    // 使用X型结构元素对原图进行腐蚀
    dilate(GraySrc, dstImage2, XMat);
    // 正方形对上一步的结果进行腐蚀
    erode(dstImage2, dstImage2, squareMat);
    // 计算差值
    absdiff(dstImage2, dstImage1, dstImage1);
    threshold(dstImage1, dstImage1, 40, 255, THRESH_BINARY);
    // 绘图
    for (int i = 0; i < dstImage1.rows; i++)
    {
        // 获取行指针
        const uchar *data = dstImage1.ptr<uchar>(i);
        for (int j = 0; j < dstImage1.cols; j++)
        {
            // 如果是角点，则进行绘制圆圈
            if (data[j])
            {
                circle(image1, Point(j, i), 8, Scalar(0, 255, 0));
            }
        }
    }
    myShowMat(image1);
}

void MainWindow::on_houghLinesButton_clicked()
{
    using namespace cv;
    using namespace std;

    Mat midImage,dstImage;//临时变量和目标图的定义

    // 进行边缘检测和转化为灰度图
    Canny(image, midImage, 50, 200, 3);//进行一此canny边缘检测
    cvtColor(midImage,dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图

    /* HoughLines */
    //    // 进行霍夫线变换
    //    vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
    //    HoughLines(midImage, lines, 1, CV_PI/180, 150, 0, 0 );

    //    // 依次在图中绘制出每条线段
    //    for( size_t i = 0; i < lines.size(); i++ )
    //    {
    //        float rho = lines[i][0], theta = lines[i][1];
    //        Point pt1, pt2;
    //        double a = cos(theta), b = sin(theta);
    //        double x0 = a*rho, y0 = b*rho;
    //        pt1.x = cvRound(x0 + 1000*(-b));
    //        pt1.y = cvRound(y0 + 1000*(a));
    //        pt2.x = cvRound(x0 - 1000*(-b));
    //        pt2.y = cvRound(y0 - 1000*(a));
    //        line( dstImage, pt1, pt2, Scalar(55,100,195), 1, CV_AA);
    //    }

    /* HoughLinesP */
    // 进行霍夫线变换
    vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
    HoughLinesP(midImage, lines, 1, CV_PI/180, 80, 50, 10 );

    // 依次在图中绘制出每条线段
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 1, CV_AA);
    }


    myShowMat(dstImage);
}

void MainWindow::on_houghCirclesButton_clicked() //HoughCircles
{
    using namespace cv;
    using namespace std;

    // 载入原始图和Mat变量定义
    Mat srcImage = image.clone();
    Mat midImage;//临时变量和目标图的定义

    // 转为灰度图，进行图像平滑
    cvtColor( srcImage, midImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图
    GaussianBlur( midImage, midImage, Size(9, 9), 1, 2 );

    // 进行霍夫圆变换
    vector<Vec3f> circles;

    /*ATTENTION!
     *HoughCircles方法对参数比较敏感，
     * 很小的改动就可能导致差别很大的检测效果，
     * 需要针对不同图像的不同检测用途进行调试。*/
    HoughCircles( midImage, circles, CV_HOUGH_GRADIENT, 1, 1, 10, 160, 0, 0 );

    // 依次在图中绘制出圆
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        //绘制圆心
        circle( srcImage, center, 3, Scalar(0,255,0), -1, 8, 0 );
        //绘制圆轮廓
        circle( srcImage, center, radius, Scalar(155,50,255), 3, 8, 0 );
    }

    // 显示效果图
    myShowMat(srcImage);
}
