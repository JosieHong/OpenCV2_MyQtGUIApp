#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QSlider>

namespace Ui
{
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void myShowMat(cv::Mat img);
    void myShowShreshold();
    cv::Mat QImage2cvMat(QImage image);

private:
    Ui::MainWindow *ui;
    cv::Mat image;  // the image variable

    int shreshold_1 = 3;
    int shreshold_2 = 9;

private slots:
    void on_openImageButton_clicked();
    void on_processButton_clicked();
    void on_cannyButton_clicked();
    void on_thresholdSlider_1_valueChanged(int value);
    void on_thresholdSlider_2_valueChanged(int value);
    void on_dilateButton_clicked();
    void on_erodeButton_clicked();
    void on_openButton_clicked();
    void on_closeButton_clicked();
    void on_getConersButton_clicked();
    void on_houghLinesButton_clicked();
    void on_houghCirclesButton_clicked();
    void on_floofFillButton_clicked();
};

#endif // MAINWINDOW_H
