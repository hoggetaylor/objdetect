#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

int main( int argc, char** argv )
{
    cv::ocl::setUseOpenCL(true);
    std::cout << "Loading target image" << std::endl;
    // Read the source images
    if( argc != 2 ){ return -1; }
    Mat target_img = imread( argv[1], IMREAD_GRAYSCALE );
    if( !target_img.data){ 
        std::cout<< " --(!) Error reading target image" << std::endl;
        return -1;
    }
    namedWindow("A", WINDOW_AUTOSIZE);
    cv::Ptr<SIFT> detector = SIFT::create();
    cv::Ptr<SIFT> descriptor = SIFT::create();
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

    // Precompute keypoints and descriptors for target image
    std::vector<cv::KeyPoint> target_keypoints;
    cv::Mat target_descriptors;
    detector->detect(target_img, target_keypoints);
    descriptor->compute(target_img, target_keypoints, target_descriptors);

    VideoCapture camera;
    if (!camera.open(0)) {
        return -1;
    } else {
        while (true) {
            cv::Mat scene_img;
            cv::Mat scene_img_gray;
            camera >> scene_img;
            if (scene_img.empty()) break;
            cvtColor(scene_img, scene_img_gray, COLOR_BGR2GRAY);

            std::vector<cv::KeyPoint> scene_keypoints;
            cv::Mat scene_descriptors;
            detector->detect(scene_img_gray, scene_keypoints);
            descriptor->compute(scene_img_gray, scene_keypoints, scene_descriptors);
            std::vector<cv::DMatch> matches;
            matcher->match(target_descriptors, scene_descriptors, matches);

            std::vector<cv::Point2f> target, scene;
            for (const auto &match : matches) {
                target.push_back(target_keypoints[match.queryIdx].pt);
                scene.push_back(scene_keypoints[match.trainIdx].pt);
            }

            cv::Mat H = findHomography(target, scene, RANSAC);

            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( target_img.cols, 0 );
            obj_corners[2] = cvPoint( target_img.cols, target_img.rows ); obj_corners[3] = cvPoint( 0, target_img.rows );
            std::vector<Point2f> scene_corners(4);
            perspectiveTransform(obj_corners, scene_corners, H);

            //-- Draw lines between the corners
            line(scene_img, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
            line(scene_img, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
            line(scene_img, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
            line(scene_img, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

            imshow("A", scene_img);
            if(waitKey(10) == 27) break; // stop capturing by pressing ESC
        }
    }
}

