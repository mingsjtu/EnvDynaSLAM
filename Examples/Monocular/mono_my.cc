/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<map>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames1, vector<string> &vstrImageFilenames2, 
vector<double> &vTimestamps1,vector<double> &vTimestamps2);
int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence path_to_envsequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames1;
    vector<string> vstrImageFilenames2;
    vector<double> vTimestamps1;
    vector<double> vTimestamps2;
    string move_dir=argv[3];
    string env_dir=argv[4];
    string strFile = string(argv[3])+"/join_rgb.txt";
    LoadImages(strFile, vstrImageFilenames1,vstrImageFilenames2, vTimestamps1,vTimestamps2);
    
    cv::Mat im_bk;
    cv::Mat im_env;

    int nImages = vstrImageFilenames1.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,move_dir,env_dir,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames1[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps1[ni];
        double env_tframe = vTimestamps2[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames1[ni] << endl;
            return 1;
        }

        //get env mask
        
        if (ni < 30)
        {
            im_env = cv::imread(string(env_dir)+"/"+vstrImageFilenames2[ni],0);
            if (ni == 0)
            {
                // im_bk = cv::Mat::zeros(im_env.rows, im_env.cols, CV_8UC3);
                im_bk = cv::Mat::zeros(im_env.rows, im_env.cols, CV_8UC1);
            }
            // cout<<"im_bk"<<im_bk.type()<<endl;
	        // cout<<"im_env"<<im_env.type()<<endl;

            im_bk = im_bk*0.9 + 0.1 * im_env.clone();
            SLAM.Set_envbk(im_bk);
            // im_bk=cv::Mat::ones(im_env.rows, im_env.cols, CV_32FC1);
        }        

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,env_tframe,vstrImageFilenames1[ni],vstrImageFilenames2[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps1[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps1[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames1, vector<string> &vstrImageFilenames2, 
    vector<double> &vTimestamps1,vector<double> &vTimestamps2)
    //1, move_dir;2,env_dir
{
    ifstream f;
    f.open(strFile.c_str());
    std::cout << "Load start \n";
    
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);
    cout<<s0<<endl;
    int num = 0;
    while (!f.eof())
    {
        num++;
        string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t1,t2=0;
            string sRGB1;
            string sRGB2="";
            ss >> t1>>sRGB1>>t2>>sRGB2;
            //ss >> t1>>sRGB1;
            vTimestamps1.push_back(t1);
            vTimestamps2.push_back(t2);
            vstrImageFilenames1.push_back(sRGB1);
            vstrImageFilenames2.push_back(sRGB2);
        }
    }
    std::cout << num << "Load end \n";
}