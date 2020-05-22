#ifndef ENV_POSE_H
#define ENV_POSE_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <string>
#include "Frame.h"

namespace ORB_SLAM2
{

class Env_processor
{

private:
    cv::Ptr<cv::FeatureDetector> mpdetector;
    cv::Ptr<cv::DescriptorExtractor> mpextractor;
    cv::Ptr<cv::BFMatcher> mpmatcher;
    int miminHession=1000;
    int msetcount=0;
    cv::Mat menvmask;

public:
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mTcw;
    cv::Mat mK;
    float fx;
    float fy;
    float cx;
    float cy;
    float mnMinX;
    float mnMaxX;
    float mnMinY;
    float mnMaxY;
    int menvnum;
    cv::Mat mH21;
    bool mbinitOK;


public:
    Env_processor(cv::Mat K,float nMaxX,float nMaxY);
    bool set_Envpose(const cv::Mat img1, const cv::Mat img2, cv::Mat &R, cv::Mat &t, cv::Mat &H21, cv::Mat &H12, cv::Mat K1, cv::Mat K2,string outputdir);
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                const std::vector<cv::DMatch> &matches, std::vector<bool> &vbMatchesInliers, const cv::Mat &K1,
                const cv::Mat &K2, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);

    void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                              std::vector<cv::KeyPoint> &keypoints1,
                              std::vector<cv::KeyPoint> &keypoints2,
                              std::vector<cv::DMatch> &matches);

    void draw_point(const cv::Mat img_1, const cv::Mat img_2, const std::vector<cv::KeyPoint> vKeys1,
                    const std::vector<cv::KeyPoint> vKeys2,
                    const std::vector<cv::DMatch> matches, const cv::Scalar color, const std::string show_name1,
                    const std::string show_name2, std::vector<bool> vbmatches);

    bool pose_estimation_2d2d(std::vector<cv::KeyPoint> vKeys1,
                          std::vector<cv::KeyPoint> vKeys2,
                          std::vector<cv::DMatch> matches,
                          cv::Mat &H21, cv::Mat &H12,std::vector<bool> &vbMatchesInliers);

    cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K);

    void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames1, std::vector<std::string> &vstrImageFilenames2, std::vector<float> &vTimestamps);
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<cv::DMatch> matches, float sigma, std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, std::vector<bool> &vbmatches);
    float CheckFundamental(const cv::Mat &F21, std::vector<cv::DMatch> matches, float sigma, std::vector<cv::KeyPoint> vKeys1,
                           std::vector<cv::KeyPoint> vKeys2, std::vector<bool> &vbmatches);

    void get_mask(const std::vector<cv::DMatch> matches, std::vector<cv::DMatch> &matchesout, cv::Mat pFrImg, cv::Mat H21, const std::vector<cv::KeyPoint> vKeys1,
                  const std::vector<cv::KeyPoint> vKeys2, std::vector<bool> vbmatches);
    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K1, cv::Mat &K2,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated,
                      float minParallax, int minTriangulated,
                      const std::vector<cv::KeyPoint> mvKeys1, const std::vector<cv::KeyPoint> mvKeys2,
                      const std::vector<cv::DMatch> matches);

    bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated,
                      std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, std::vector<cv::DMatch> matches);

    void recoverH(const cv::Mat K, const cv::Mat R, const cv::Mat t, const std::vector<cv::DMatch> matches, const std::vector<bool> vbMatchesInliers,
                                 std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, const cv::Mat &img1, const cv::Mat &img2,const string outputdir);

    void triangulation(
        const std::vector<cv::KeyPoint> &vKeys1,
        const std::vector<cv::KeyPoint> &vKeys2,
        const std::vector<cv::DMatch> &matches,
        const cv::Mat &R, const cv::Mat &t,
        std::vector<cv::Mat> &points, const cv::Mat K1, const cv::Mat K2);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    bool isDynamic(MapPoint *pMP,float &u,float &v,float &init_u,float &init_v,cv::Mat curTcw,bool &bmask);
    bool get_envmask(string env_dir,string env_name,string move_name,const cv::Mat &im_bk);
    void set_pose(cv::Mat Tcw,cv::Mat H21);
};

} // namespace ORB_SLAM2

#endif
