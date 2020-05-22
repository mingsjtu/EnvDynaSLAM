// /home/gm/slam_current/ch7/small_env/1.jpg

// /home/gm/slam_current/ch7/small_env/IMG_20200419_074811.jpg
#include "Env_processor.h"

namespace ORB_SLAM2
{
    Env_processor::Env_processor(cv::Mat K, float nMaxX, float nMaxY) : mK(K),
                                mnMinX(0.0), mnMaxX(nMaxX), mnMinY(0.0), mnMaxY(nMaxY), menvnum(0), mbinitOK(false), msetcount(0)
    {
        miminHession = 1000;
        mpdetector = cv::xfeatures2d::SIFT::create(miminHession); //和surf的区别：只是SURF→SIFT
        // SiftDescriptorExtractor extractor;
        mpextractor = cv::xfeatures2d::SIFT::create();
        mpmatcher = new cv::BFMatcher(cv::NORM_L2, true); // 暴风匹配
        fx = mK.at<float>(0, 0);
        fy = mK.at<float>(1, 1);
        cx = mK.at<float>(0, 2);
        cy = mK.at<float>(1, 2);
        //printf("Env processor:\n");
        printf("nMaxX %f, nMaxY %f\n",nMaxX,nMaxY);
        menvmask = cv::Mat::zeros(cv::Size(nMaxX, nMaxY), CV_8UC1);
        mbinitOK = false;
    }

    cv::Point2f Env_processor::pixel2cam(const cv::Point2f &p, const cv::Mat &K)
    {
        return cv::Point2f(
            (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
            (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1));
    }

    void Env_processor::find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                                             std::vector<cv::KeyPoint> &vKeys1,
                                             std::vector<cv::KeyPoint> &vKeys2,
                                             std::vector<cv::DMatch> &matches)
    {
        // cv::imshow("img1",img1);
        // cv::imshow("img2",img2);
        // cv::waitKey(10);
        vector<cv::DMatch> match;
        mpdetector->detect(img1, vKeys1, cv::Mat()); //找出关键点
        mpdetector->detect(img2, vKeys2, cv::Mat()); //找出关键点
        // //printf("keypoints size %d\n",keypoints1.size());
        // //printf("keypoints size %d\n",keypoints2.size());

        cv::Mat descriptor1, descriptor2;
        mpextractor->compute(img1, vKeys1, descriptor1);
        mpextractor->compute(img2, vKeys2, descriptor2);
        mpmatcher->match(descriptor1, descriptor2, match);
        // cout << "# matches : " << match.size() << endl;

        //-- 第四步:匹配点对筛选
        double min_dist = 10000, max_dist = 0;

        //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
        for (size_t i = 0; i < match.size(); i++)
        {
            double dist = match[i].distance;
            if (dist < min_dist)
                min_dist = dist;
            if (dist > max_dist)
                max_dist = dist;
        }

        //printf ( "-- Max dist : %f \n", max_dist );
        //printf ( "-- Min dist : %f \n", min_dist );

        //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        for (int i = 0; i < match.size(); i++)
        {
            if (match[i].distance <= max(8 * min_dist, 100.0))
            {
                matches.push_back(match[i]);
            }
        }
    }

    bool Env_processor::pose_estimation_2d2d(std::vector<cv::KeyPoint> vKeys1,
                                             std::vector<cv::KeyPoint> vKeys2,
                                             std::vector<cv::DMatch> matches,
                                             cv::Mat &H21, cv::Mat &H12, std::vector<bool> &vbMatchesInliers)
    {
        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;

        for (int i = 0; i < (int)matches.size(); i++)
        {
            // //printf("queryIdx %d,trainIdx %d\n", matches[i].queryIdx, matches[i].trainIdx);
            points1.push_back(vKeys1[matches[i].queryIdx].pt);
            points2.push_back(vKeys2[matches[i].trainIdx].pt);
        }

        std::vector<cv::Point3f> vP3D;
        std::vector<bool> vbTriangulated;

        H21 = findHomography(points1, points2, cv::RANSAC, 3);

        H21.convertTo(H21, CV_32F);

        // std::cout << "H21 is " << std::endl
        //      << H21 << std::endl;

        H12 = H21.inv();
        mH21=H21.clone();
        float SH = CheckHomography(H21, H12, matches, 1.0, vKeys1, vKeys2, vbMatchesInliers);
        printf("SH %.4f\n", SH);

        return (SH > 280.0);
    }

    bool Env_processor::set_Envpose(const cv::Mat img1, const cv::Mat img2, cv::Mat &R, cv::Mat &t, cv::Mat &H21, cv::Mat &H12, cv::Mat K1, cv::Mat K2,string outputdir)
    {
        std::vector<cv::KeyPoint> vKeys1;
        std::vector<cv::KeyPoint> vKeys2;
        std::vector<cv::DMatch> matches;
        find_feature_matches(img1, img2, vKeys1, vKeys2, matches);
        std::vector<bool> vbMatchesInliers(matches.size());

        bool result = pose_estimation_2d2d(vKeys1, vKeys2, matches, H21, H12, vbMatchesInliers);
        msetcount++;
        if (result)
        {
            mbinitOK = true;
            recoverH(K1,R,t,matches,vbMatchesInliers,vKeys1,vKeys2,img1,img2,outputdir);
            return true;
        }
        return false;
    }

    bool Env_processor::ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                                     cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated,
                                     std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, std::vector<cv::DMatch> matches)
    {
        int N = 0;
        for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
            if (vbMatchesInliers[i])
                N++;

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewise planar environment.
        // International Journal of Pattern Recognition and Artificial Intelligence, 1988

        cv::Mat invK = K.inv();
        cv::Mat A = invK * H21 * K;

        cv::Mat U, w, Vt, V;
        cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
        V = Vt.t();

        float s = determinant(U) * determinant(Vt);

        float d1 = w.at<float>(0);
        float d2 = w.at<float>(1);
        float d3 = w.at<float>(2);

        if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
        {
            return false;
        }

        std::vector<cv::Mat> vR, vt, vn;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);

        //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
        float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
        float x1[] = {aux1, aux1, -aux1, -aux1};
        float x3[] = {aux3, -aux3, aux3, -aux3};

        //case d'=d2
        float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

        float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        for (int i = 0; i < 4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = ctheta;
            Rp.at<float>(0, 2) = -stheta[i];
            Rp.at<float>(2, 0) = stheta[i];
            Rp.at<float>(2, 2) = ctheta;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = -x3[i];
            tp *= d1 - d3;

            cv::Mat t = U * tp;
            vt.push_back(t / norm(t));

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;
            if (n.at<float>(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        //case d'=-d2
        float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

        float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        for (int i = 0; i < 4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = cphi;
            Rp.at<float>(0, 2) = sphi[i];
            Rp.at<float>(1, 1) = -1;
            Rp.at<float>(2, 0) = sphi[i];
            Rp.at<float>(2, 2) = -cphi;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = x3[i];
            tp *= d1 + d3;

            cv::Mat t = U * tp;
            vt.push_back(t / norm(t));

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;
            if (n.at<float>(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;
        float bestParallax = -1;
        std::vector<cv::Point3f> bestP3D;
        std::vector<bool> bestTriangulated;

        // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        for (size_t i = 0; i < 8; i++)
        {
            float parallaxi;
            std::vector<cv::Point3f> vP3Di;
            std::vector<bool> vbTriangulatedi;

            int nGood = CheckRT(vR[i], vt[i], vKeys1, vKeys2, matches, vbMatchesInliers, K, K, vP3Di, 1.0, vbTriangulatedi, parallaxi);
            // std::cout<<"nGood   "<<nGood<<std::endl;

            //std::cout<<"vR "<<vR[i]<<std::endl<<"vt "<<vt[i]<<std::endl;
            if (nGood > bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                bestParallax = parallaxi;
                bestP3D = vP3Di;
                bestTriangulated = vbTriangulatedi;
            }
            else
            {
                if (nGood > secondBestGood)
                {
                    secondBestGood = nGood;
                }
            }
            // std::cout<<"secondBestGood "<< secondBestGood<< "bestGood "<<bestGood<<std::endl;
        }

        // R21=vR[bestSolutionIdx];
        // t21=vt[bestSolutionIdx];

        //printf("secondBestGood %d,bestGood %d, bestParallax %f,minParallax %f, minTriangulated %d, N %d",secondBestGood,bestGood,bestParallax,minParallax,minTriangulated ,N);
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;
        if (secondBestGood < bestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.7 * N)
        {
            return true;
        }

        return false;
    }

    void Env_processor::recoverH(const cv::Mat K, const cv::Mat R, const cv::Mat t, const std::vector<cv::DMatch> matches, const std::vector<bool> vbMatchesInliers,
                                 std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, const cv::Mat &img1, const cv::Mat &img2,const string outputdir)
    {
        const float h11 = mH21.at<float>(0, 0);
        const float h12 = mH21.at<float>(0, 1);
        const float h13 = mH21.at<float>(0, 2);
        const float h21 = mH21.at<float>(1, 0);
        const float h22 = mH21.at<float>(1, 1);
        const float h23 = mH21.at<float>(1, 2);
        const float h31 = mH21.at<float>(2, 0);
        const float h32 = mH21.at<float>(2, 1);
        const float h33 = mH21.at<float>(2, 2);

        std::vector<cv::KeyPoint> img2_vKeys;
        std::vector<cv::DMatch> img2_vmatches;
	    int img2_count=0;
        for (size_t i = 0, iend = matches.size(); i < iend; i++)
        {
            if (!vbMatchesInliers[i])
                continue;

            const cv::KeyPoint &kp1 = vKeys1[matches[i].queryIdx];
            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
            const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
            const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;
	    if(u1in2>0&& u1in2<mnMaxX&&v1in2>0&& v1in2<mnMaxY)
	    {
	      img2_vKeys.push_back(cv::KeyPoint(u1in2, v1in2, 1.0));
	      img2_vmatches.push_back(cv::DMatch(matches[i].queryIdx, img2_count++, 0));
	    }  
        }
        cv::Mat output;
        cv::drawMatches(img1, vKeys1, img2, img2_vKeys, img2_vmatches, output, cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0));
        // cv::drawMatches(img1,vKeys1,img2,vKeys2,pure_matches,output,cv::Scalar(0, 255, 0),cv::Scalar(255, 0, 0));
        cv::imwrite(outputdir+"/setinit" + to_string(msetcount) + ".png", output);
        // cv::waitKey(0);

        //w 3*3 b 3*1 p2=w*p1+b
        // cv::Mat w=K*R*K.inv();
        // cv::Mat b=K*t;

        // for(int i=0;i<matches.size();i++)
        // {
        //     cv::Point2f x1=pixel2cam(vKeys1[matches[i].queryIdx].pt,K);
        //     cv::Point2f x2=pixel2cam(vKeys2[matches[i].trainIdx].pt,K);

        // }
    }
    float Env_processor::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<cv::DMatch> matches, float sigma,
                                         std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, std::vector<bool> &vbMatchesInliers)
    {
        const int N = matches.size();

        const float h11 = H21.at<float>(0, 0);
        const float h12 = H21.at<float>(0, 1);
        const float h13 = H21.at<float>(0, 2);
        const float h21 = H21.at<float>(1, 0);
        const float h22 = H21.at<float>(1, 1);
        const float h23 = H21.at<float>(1, 2);
        const float h31 = H21.at<float>(2, 0);
        const float h32 = H21.at<float>(2, 1);
        const float h33 = H21.at<float>(2, 2);

        const float h11inv = H12.at<float>(0, 0);
        const float h12inv = H12.at<float>(0, 1);
        const float h13inv = H12.at<float>(0, 2);
        const float h21inv = H12.at<float>(1, 0);
        const float h22inv = H12.at<float>(1, 1);
        const float h23inv = H12.at<float>(1, 2);
        const float h31inv = H12.at<float>(2, 0);
        const float h32inv = H12.at<float>(2, 1);
        const float h33inv = H12.at<float>(2, 2);

        //检验H21 的精度

        float score = 0;

        const float th = 10;

        const float invSigmaSquare = 1.0 / (sigma * sigma);
        int out_point = 0;
        for (int i = 0; i < N; i++)
        {
            bool bIn = true;
            const cv::KeyPoint &kp1 = vKeys1[matches[i].queryIdx];
            const cv::KeyPoint &kp2 = vKeys2[matches[i].trainIdx];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
            const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
            const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

            const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

            const float chiSquare1 = squareDist1 * invSigmaSquare;

            if (chiSquare1 > th)
                bIn = false;
            else
                score += th - chiSquare1;

            const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
            const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
            const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

            const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

            const float chiSquare2 = squareDist2 * invSigmaSquare;

            if (chiSquare2 > th)
                bIn = false;
            else
                score += th - chiSquare2;
            ////printf("chiSquare1 %lf,chiSquare %lf\n",chiSquare1,chiSquare2);
            if (!bIn)
            {
                vbMatchesInliers[i] = false;
                out_point++;
            }
            else
            {
                vbMatchesInliers[i] = true;
            }
        }
        //printf("H: all point %d, out point %d\n", N, out_point);
        return score;
    }

    float Env_processor::CheckFundamental(const cv::Mat &F21, std::vector<cv::DMatch> matches, float sigma,
                                          std::vector<cv::KeyPoint> vKeys1, std::vector<cv::KeyPoint> vKeys2, std::vector<bool> &vbMatchesInliers)
    {
        std::cout << "F21  " << F21 << std::endl;
        const int N = matches.size();

        const float f11 = F21.at<float>(0, 0);
        const float f12 = F21.at<float>(0, 1);
        const float f13 = F21.at<float>(0, 2);
        const float f21 = F21.at<float>(1, 0);
        const float f22 = F21.at<float>(1, 1);
        const float f23 = F21.at<float>(1, 2);
        const float f31 = F21.at<float>(2, 0);
        const float f32 = F21.at<float>(2, 1);
        const float f33 = F21.at<float>(2, 2);

        float score = 0;

        const float th = 10;
        const float thScore = 5.991;
        const float invSigmaSquare = 1.0 / (sigma * sigma);
        int out_point = 0;

        for (int i = 0; i < N; i++)
        {

            bool bIn = true;

            const cv::KeyPoint &kp1 = vKeys1[matches[i].queryIdx];
            const cv::KeyPoint &kp2 = vKeys2[matches[i].trainIdx];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            const float a2 = f11 * u1 + f12 * v1 + f13;
            const float b2 = f21 * u1 + f22 * v1 + f23;
            const float c2 = f31 * u1 + f32 * v1 + f33;

            const float num2 = a2 * u2 + b2 * v2 + c2;
            const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);
            const float chiSquare1 = squareDist1 * invSigmaSquare;
            if (chiSquare1 > th)
                bIn = false;
            else
                score += thScore - chiSquare1;
            const float a1 = f11 * u2 + f21 * v2 + f31;
            const float b1 = f12 * u2 + f22 * v2 + f32;
            const float c1 = f13 * u2 + f23 * v2 + f33;
            const float num1 = a1 * u1 + b1 * v1 + c1;

            const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);
            const float chiSquare2 = squareDist2 * invSigmaSquare;
            ////printf("chi1 %f, chi2 %f\n",chiSquare1,chiSquare2);
            if (chiSquare2 > th)
                bIn = false;
            else
                score += thScore - chiSquare2;

            if (!bIn)
            {
                vbMatchesInliers[i] = false;
                out_point++;
            }
            else
            {
                vbMatchesInliers[i] = true;
            }
        }
        //printf("F: all point %d, out point %d\n", N, out_point);

        return score;
    }

    void Env_processor::draw_point(const cv::Mat img_1, const cv::Mat img_2, const std::vector<cv::KeyPoint> vKeys1,
                                   const std::vector<cv::KeyPoint> vKeys2,
                                   const std::vector<cv::DMatch> matches, const cv::Scalar color, const std::string show_name1, const std::string show_name2, std::vector<bool> vbMatchesInliers)
    {
        cv::Mat img_1_copy, img_2_copy;
        img_1.copyTo(img_1_copy);
        img_2.copyTo(img_2_copy);
        int i = 0;
        for (cv::DMatch m : matches)
        {
            if (!vbMatchesInliers[i])
                continue;

            circle(img_1_copy, vKeys1[m.queryIdx].pt, 5, color, -1);

            circle(img_2_copy, vKeys2[m.queryIdx].pt, 5, color, -1);

            i++;
        }
        imshow(show_name1, img_1_copy);
        imshow(show_name2, img_2_copy);
    }

    void Env_processor::triangulation(
        const std::vector<cv::KeyPoint> &vKeys1,
        const std::vector<cv::KeyPoint> &keypoint_2,
        const std::vector<cv::DMatch> &matches,
        const cv::Mat &R, const cv::Mat &t,
        std::vector<cv::Mat> &points, const cv::Mat K1, const cv::Mat K2)
    {
        cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0);
        cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2), t.at<float>(0, 0),
                      R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2), t.at<float>(1, 0),
                      R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2), t.at<float>(2, 0));

        std::vector<cv::Point2f> pts_1, pts_2;
        for (cv::DMatch m : matches)
        {
            // 将像素坐标转换至相机坐标
            pts_1.push_back(pixel2cam(vKeys1[m.queryIdx].pt, K1));
            pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K2));
        }

        cv::Mat pts_4d;
        triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

        // 转换成非齐次坐标
        for (int i = 0; i < pts_4d.cols; i++)
        {
            cv::Mat x = pts_4d.col(i);
            x /= x.at<float>(3, 0); // 归一化
                                    // Point3d p (
                                    //     x.at<float>(0,0),
                                    //     x.at<float>(1,0),
                                    //     x.at<float>(2,0)
                                    // );
            x = x.rowRange(0, 3).clone();
            //cv::Mat p = (cv::Mat_<float>(3, 1)<<x.at<float>(0,0),x.at<float>(1,0),x.at<float>(2,0)) ;
            //std::cout<<x.size()<<std::endl;
            points.push_back(x);
        }
    }

    int Env_processor::CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                               const std::vector<cv::DMatch> &matches, std::vector<bool> &vbMatchesInliersInliers, const cv::Mat &K1,
                               const cv::Mat &K2, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax)
    {
        // Calibration parameters
        const float fx1 = K1.at<float>(0, 0);
        const float fy1 = K1.at<float>(1, 1);
        const float cx1 = K1.at<float>(0, 2);
        const float cy1 = K1.at<float>(1, 2);

        // Calibration parameters
        const float fx2 = K2.at<float>(0, 0);
        const float fy2 = K2.at<float>(1, 1);
        const float cx2 = K2.at<float>(0, 2);
        const float cy2 = K2.at<float>(1, 2);

        vbGood = std::vector<bool>(vKeys1.size(), false);
        vP3D.resize(vKeys1.size());

        std::vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // Camera 1 Projection Matrix K[I|0]
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        //K1.copyTo(P1.rowRange(0,3).colRange(0,3));

        cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

        // Camera 2 Projection Matrix K[R|t]
        cv::Mat P2(3, 4, CV_32F);
        R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t.copyTo(P2.rowRange(0, 3).col(3));
        P2 = P2;

        cv::Mat O2 = -R.t() * t;

        int nGood = 0;
        std::vector<cv::Mat> tri_result;
        triangulation(vKeys1, vKeys2, matches, R, t, tri_result, K1, K2);

        for (size_t i = 0, iend = matches.size(); i < iend; i++)
        {
            if (!vbMatchesInliersInliers[i])
                continue;

            const cv::KeyPoint &kp1 = vKeys1[matches[i].queryIdx];
            const cv::KeyPoint &kp2 = vKeys2[matches[i].trainIdx];
            cv::Mat p3dC1;

            p3dC1 = tri_result[i];
            // std::cout<<"p3dC1  "<<p3dC1<<std::endl;

            if (!std::isfinite(p3dC1.at<float>(0)) || !std::isfinite(p3dC1.at<float>(1)) || !std::isfinite(p3dC1.at<float>(2)))
            {
                vbGood[matches[i].queryIdx] = false;
                continue;
            }

            // Check parallax
            cv::Mat normal1 = p3dC1 - O1;
            float dist1 = norm(normal1);

            cv::Mat normal2 = p3dC1 - O2;
            float dist2 = norm(normal2);

            float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

            // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            cv::Mat p3dC2 = R * p3dC1 + t;

            if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            // Check reprojection error in first image
            float im1x, im1y;
            float invZ1 = 1.0 / p3dC1.at<float>(2);
            im1x = fx1 * p3dC1.at<float>(0) * invZ1 + cx1;
            im1y = fy1 * p3dC1.at<float>(1) * invZ1 + cy1;
            // cout<<"env init:   "<<p3dC1<<endl;

            float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);
            if (squareError1 > th2)
                continue;

            // Check reprojection error in second image
            float im2x, im2y;
            float invZ2 = 1.0 / p3dC2.at<float>(2);
            im2x = fx2 * p3dC2.at<float>(0) * invZ2 + cx2;
            im2y = fy2 * p3dC2.at<float>(1) * invZ2 + cy2;

            float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);
            ////printf("err1 %f, err2 %f\n",squareError1,squareError2);

            if (squareError2 > th2)
                continue;

            vCosParallax.push_back(cosParallax);
            vP3D[matches[i].queryIdx] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
            nGood++;

            if (cosParallax < 0.99998)
                vbGood[matches[i].queryIdx] = true;
        }

        if (nGood > 0)
        {
            sort(vCosParallax.begin(), vCosParallax.end());

            size_t idx = std::min(50, int(vCosParallax.size() - 1));
            parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
        }
        else
            parallax = 0;

        return nGood;
    }

    void Env_processor::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
    {
        cv::Mat u, w, vt;
        cv::SVD::compute(E, w, u, vt);

        u.col(2).copyTo(t);
        t = t / norm(t);

        cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
        W.at<float>(0, 1) = -1;
        W.at<float>(1, 0) = 1;
        W.at<float>(2, 2) = 1;

        R1 = u * W * vt;
        if (determinant(R1) < 0)
            R1 = -R1;

        R2 = u * W.t() * vt;
        if (determinant(R2) < 0)
            R2 = -R2;
    }

    bool Env_processor::ReconstructF(std::vector<bool> &vbMatchesInliersInliers, cv::Mat &F21, cv::Mat &K1, cv::Mat &K2,
                                     cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated,
                                     float minParallax, int minTriangulated,
                                     const std::vector<cv::KeyPoint> mvKeys1, const std::vector<cv::KeyPoint> mvKeys2,
                                     const std::vector<cv::DMatch> matches)
    {
        int N = 0;
        for (size_t i = 0, iend = vbMatchesInliersInliers.size(); i < iend; i++)
            if (vbMatchesInliersInliers[i])
                N++;

        // Compute Essential Matrix from Fundamental Matrix
        cv::Mat E21 = K2.t() * F21 * K1;

        cv::Mat R1, R2, t;

        // Recover the 4 motion hypotheses
        DecomposeE(E21, R1, R2, t);

        cv::Mat t1 = t;
        cv::Mat t2 = -t;

        // Reconstruct with the 4 hyphoteses and check
        std::vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
        std::vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
        float parallax1, parallax2, parallax3, parallax4;

        float mSigma2 = 1.0;
        int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, matches, vbMatchesInliersInliers, K1, K2, vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
        int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, matches, vbMatchesInliersInliers, K1, K2, vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
        int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, matches, vbMatchesInliersInliers, K1, K2, vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
        int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, matches, vbMatchesInliersInliers, K1, K2, vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

        int maxGood = std::max(nGood1, std::max(nGood2, std::max(nGood3, nGood4)));

        R21 = cv::Mat();
        t21 = cv::Mat();

        int nMinGood = std::max(static_cast<int>(0.9 * N), minTriangulated);

        int nsimilar = 0;
        if (nGood1 > 0.7 * maxGood)
            nsimilar++;
        if (nGood2 > 0.7 * maxGood)
            nsimilar++;
        if (nGood3 > 0.7 * maxGood)
            nsimilar++;
        if (nGood4 > 0.7 * maxGood)
            nsimilar++;
        //printf("nGood1 %d, nGood2 %d, nGood3 %d, nGood4 %d\n",nGood1,nGood2,nGood3,nGood4);
        //printf("maxGood %d,minnGood %d,nsimilar%d\n",maxGood,nMinGood,nsimilar);

        // If there is not a clear winner or not enough triangulated points reject initialization
        if (maxGood < nMinGood || nsimilar > 1)
        {
            return false;
        }

        // If best reconstruction has enough parallax initialize
        if (maxGood == nGood1)
        {
            if (parallax1 > minParallax)
            {
                vP3D = vP3D1;
                vbTriangulated = vbTriangulated1;

                R1.copyTo(R21);
                t1.copyTo(t21);
                return true;
            }
        }
        else if (maxGood == nGood2)
        {
            if (parallax2 > minParallax)
            {
                vP3D = vP3D2;
                vbTriangulated = vbTriangulated2;

                R2.copyTo(R21);
                t1.copyTo(t21);
                return true;
            }
        }
        else if (maxGood == nGood3)
        {
            if (parallax3 > minParallax)
            {
                vP3D = vP3D3;
                vbTriangulated = vbTriangulated3;

                R1.copyTo(R21);
                t2.copyTo(t21);
                return true;
            }
        }
        else if (maxGood == nGood4)
        {
            if (parallax4 > minParallax)
            {
                vP3D = vP3D4;
                vbTriangulated = vbTriangulated4;

                R2.copyTo(R21);
                t2.copyTo(t21);
                return true;
            }
        }

        return false;
    }

    // bool Env_processor::isDynamic(MapPoint *pMP,float &u,float &v,float &init_u,float &init_v,cv::Mat curTcw,bool &bmask)
    // {
    //     bmask=false;
    //     if(!mbinitOK)
    //         return false;
    //     // 3D in absolute coordinates
    //     cv::Mat P = pMP->GetWorldPos();
    //     cv::Mat curRcw,curtcw,Pc;

    //     curRcw=curTcw.rowRange(0, 3).colRange(0, 3).clone();
    //     curtcw=curTcw.rowRange(0, 3).col(3).clone();
    //     Pc=curRcw*P+curtcw;
    //     const float &PX = Pc.at<float>(0);
    //     const float &PY= Pc.at<float>(1);
    //     const float &PZ = Pc.at<float>(2);
    //     const float invPZ = 1.0f/PZ;

    //     init_u=fx*PX*invPZ+cx;
    //     init_v=fy*PY*invPZ+cy;
    //     // cout<<"init u"<<init_u<<endl;
    //     // cout<<"init v"<<init_v<<endl;

    //     // 3D in camera coordinates
    //     const cv::Mat Penv = mR_envw*P+mt_envw;
    //     const float &PenvX = Penv.at<float>(0);
    //     const float &PenvY= Penv.at<float>(1);
    //     const float &PenvZ = Penv.at<float>(2);

    //     // Check positive depth
    //     if(PenvZ<0.0f)
    //         return false;

    //     // Project in image and check it is not outside
    //     const float invz = 1.0f/PenvZ;
    //     u=fx*PenvX*invz+cx;
    //     v=fy*PenvY*invz+cy;
    //     // cout<<"Env u"<<u<<endl;
    //     // cout<<"Env v"<<v<<endl;

    //     if(u<mnMinX || u>mnMaxX)
    //         return false;
    //     if(v<mnMinY || v>mnMaxY)
    //         return false;

    //     cout<<"menvmask size"<<menvmask.size()<<endl;
    //     cout<<"menvmask at"<<int(u)<<' '<<int(v)<<endl;
    //     printf("%d\n",*(uchar*)(menvmask.ptr<uchar>(int(u))+int(v)));
    //     if(menvmask.at<uchar>(int(u),int(v))==255)
    //     {
    //         bmask=true;
    //     }
    //     return true;

    // }

    bool Env_processor::isDynamic(MapPoint *pMP, float &u, float &v, float &init_u, float &init_v, cv::Mat curTcw, bool &bmask)
    {
      if (!mbinitOK)
            return false;
        const float h11 = mH21.at<float>(0, 0);
        const float h12 = mH21.at<float>(0, 1);
        const float h13 = mH21.at<float>(0, 2);
        const float h21 = mH21.at<float>(1, 0);
        const float h22 = mH21.at<float>(1, 1);
        const float h23 = mH21.at<float>(1, 2);
        const float h31 = mH21.at<float>(2, 0);
        const float h32 = mH21.at<float>(2, 1);
        const float h33 = mH21.at<float>(2, 2);
        bmask = false;
        
        // 3D in absolute coordinates
        cv::Mat Pc = mRcw * pMP->GetWorldPos() + mtcw;
        // cv::Mat curRcw,curtcw,Pc;

        const float &PX = Pc.at<float>(0);
        const float &PY = Pc.at<float>(1);
        const float &PZ = Pc.at<float>(2);
        const float invPZ = 1.0f / PZ;

        init_u = fx * PX * invPZ + cx;
        init_v = fy * PY * invPZ + cy;
        // cout<<"init u"<<init_u<<endl;
        // cout<<"init v"<<init_v<<endl;

        // Check positive depth
        if (PZ < 0.0f)
            return false;

        // cout<<"Env u"<<u<<endl;
        // cout<<"Env v"<<v<<endl;
        const float w1in2inv = 1.0 / (h31 * init_u + h32 * init_v + h33);
        u = (h11 * init_u + h12 * init_v + h13) * w1in2inv;
        v = (h21 * init_u + h22 * init_v + h23) * w1in2inv;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        if(int(menvmask.at<uchar>(int(u), int(v))))
            cout << "menvmask_after " << int(menvmask.at<uchar>(int(u), int(v))) << endl;
        if (menvmask.at<uchar>(int(u), int(v)) == 255)
        {
            bmask = true;
        }
        return true;
    }
    void imfill(cv::Mat srcimage, cv::Mat &dstimage)
    {
        cv::Size m_Size = srcimage.size();  
        cv::Mat temimage = cv::Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcimage.type());

        srcimage.copyTo(temimage(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));  
        cv::floodFill(temimage, cv::Point(0,0), cv::Scalar(255)); 
        cv::Mat cutImg;
        temimage(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cutImg);  
        dstimage = srcimage | (~cutImg);  
    }

    bool Env_processor::get_envmask(string env_dir, string env_name, string move_name, const cv::Mat &im_bk)
    {
        if (!mbinitOK)
            return false;
        cv::Mat pFrame = cv::imread(env_dir + '/' + env_name, 0);
        cout << "pFrame::" << env_dir + '/' + env_name << endl;
        cv::absdiff(pFrame, im_bk, menvmask);
        // cv::addWeighted(pFrame,0.003,im_bk,1-0.003,0,im_bk);
        cv::threshold(menvmask, menvmask, 40, 255.0, CV_THRESH_BINARY);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        erode(menvmask, menvmask, element);

        int dilation_size = 15;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                   cv::Point(dilation_size, dilation_size));
        ////printf("pFr TYPE %d\n", menvmask.type()); type 0 CV8U

        cv::dilate(menvmask, menvmask, kernel);

        cv::Mat pFrame_copy;
        pFrame.copyTo(pFrame_copy);

        ////最大联通区域
            // // 查找轮廓，对应连通域 
            // vector<vector<cv::Point>> contours;  
            // cv::findContours(menvmask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
              
            // // 寻找最大连通域  
            // double maxArea = 0;  
            // vector<cv::Point> maxContour;
            // int index=0;
            // for(size_t i = 0; i < contours.size(); i++)  
            // {  
            //     double area = cv::contourArea(contours[i]);  
            //     if (area > maxArea)  
            //     {  
            //         maxArea = area;  
            //         maxContour = contours[i];
            //         index=i;
            //     }  
            // }
            // cout<<"maxArea"<<maxArea<<endl;
            // cv::Mat dstImage =  cv::Mat::zeros(menvmask.rows, menvmask.cols, menvmask.type());
            // if(maxArea>6000)
            // {
            //     drawContours(dstImage, contours, index, cv::Scalar(255));
            //     imfill(dstImage, dstImage);
            //     menvmask=dstImage.clone();
            // }
            /////

            for (int i = 0; i < menvmask.rows; i++)
            {
                for (int j = 0; j < menvmask.cols; j++)
                {
                    
                    // if (dstImage.at<uchar>(i, j) == 255)
                    if (menvmask.at<uchar>(i, j) == 255)
                    {
                        pFrame_copy.at<uchar>(i, j) = 0;
                    }
                    
                }
            }
        cv::imshow("filter", pFrame_copy);
        // cv::imwrite("/media/gm/Data/SLAM/self_video/4.21moon/xiaomi_20200421_131142/output/mask"+move_name.substr(4),pFrame_copy);
        double mT = 1e3 / 30;
        // cv::waitKey(mT);
    }

    void Env_processor::set_pose(cv::Mat Tcw, cv::Mat H21)
    {
        mTcw = Tcw.clone();
        mtcw = (Tcw.rowRange(0, 3).col(3)).clone();
        mRcw = Tcw.rowRange(0, 3).colRange(0, 3).clone();
        mH21 = H21.clone();
    }
} // namespace ORB_SLAM2
