// Created by kwanghoe on 19. 1. 7.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Config.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Mat stitchImg(Mat src, Mat dest);

int main(int argc, char **argv){
    cout << "run stitching_images..." << endl;

    // Read Images
    vector<Mat> img_vec;
    vector<string> img_str_vec;
    if(argc > 1){
        for(int idx = 0; idx < argc; ++idx)
            img_str_vec.push_back(argv[idx]);
    }else{
        img_str_vec.push_back(getCmakeCurrentSourceDir() + "/data/1.jpg");
        img_str_vec.push_back(getCmakeCurrentSourceDir() + "/data/2.jpg");
    }
    for(int idx = 0; idx <= argc; ++idx)
        img_vec.push_back(imread( img_str_vec.at(idx), IMREAD_COLOR ).clone());

    Mat result = stitchImg(img_vec.at(0), img_vec.at(1));
    imshow("result", result);
    waitKey(0);

    return 0;
}

Mat stitchImg(Mat src, Mat dest){
    //-- Detect the keypoints using SURF Detector, compute the descriptors
    Ptr<SIFT> detector = SIFT::create();
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    detector->detectAndCompute(src, Mat(), keypoints_1, descriptors_1);
    detector->detectAndCompute(dest, Mat(), keypoints_2, descriptors_2);

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_1.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.02))
        {good_matches.push_back( matches[i]);}
    }

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( src, keypoints_1, dest, keypoints_2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for(int idx = 0; idx < good_matches.size(); ++idx)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_1[ good_matches[idx].queryIdx ].pt);
        scene.push_back(keypoints_2[ good_matches[idx].trainIdx ].pt);
    }

    Mat H = findHomography(scene, obj, CV_RANSAC);

    //-- Get the result image
    Mat result;
    warpPerspective(dest, result, H, cv::Size(dest.cols + src.cols, dest.rows));
    Mat half(result, cv::Rect(0, 0, src.cols, src.rows) );
    src.copyTo(half);

    return result;
}
