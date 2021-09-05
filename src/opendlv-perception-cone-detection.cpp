/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/imgcodecs/imgcodecs.hpp> //Remove
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

struct Coord {
   int x;
   int y;
   
   Coord(int ax, int ay) : x(ax), y(ay) {}
};


int32_t main(int32_t argc, char **argv) {
  int32_t retCode{1};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if ( (0 == commandlineArguments.count("cid")) ||
       (0 == commandlineArguments.count("name")) ||
       (0 == commandlineArguments.count("width")) ||
       (0 == commandlineArguments.count("height")) ) {
    std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> --width=<width of frame> --height=<height of frame> [--verbose]" << std::endl;
    std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
    std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
    std::cerr << "         --width:  width of the frame" << std::endl;
    std::cerr << "         --height: height of the frame" << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=1280 --height=720 --verbose" << std::endl;
  } else {
    const std::string NAME{commandlineArguments["name"]};
    const uint32_t WIDTH{static_cast<uint32_t>(
      std::stoi(commandlineArguments["width"]))};
    const uint32_t HEIGHT{static_cast<uint32_t>(
      std::stoi(commandlineArguments["height"]))};
    const bool VERBOSE{commandlineArguments.count("verbose") != 0};

    // Attach to the shared memory.
    std::unique_ptr<cluon::SharedMemory> 
      sharedMemory{new cluon::SharedMemory{NAME}};
    if (sharedMemory && sharedMemory->valid()) {
      std::clog << argv[0] << ": Attached to shared memory '" 
        << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." 
        << std::endl;

      // Interface to a running OpenDaVINCI session; here, 
      // you can send and receive messages.
      cluon::OD4Session od4{static_cast<uint16_t>(
        std::stoi(commandlineArguments["cid"]))};
      
      while (od4.isRunning()) {
        cv::Mat img;

        // Wait for a notification of a new frame.
        sharedMemory->wait();

        // Lock the shared memory.
        sharedMemory->lock();
        {
          // Copy image into cvMat structure.
          // Be aware of that any code between lock/unlock is blocking
          // the camera to provide the next frame. Thus, any
          // computationally heavy algorithms should be placed outside
          // lock/unlock
          cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
          img = wrapped.clone();
        }
        sharedMemory->unlock();
        
        //Part I: Find masks for each color of cones using hsv and preprocessing
        std::vector<cv::Mat> conesList; //Blue, Yellow, Red
        uint32_t heightOfCrop = HEIGHT*2/6; // For plotting
        {
  	      img = img(cv::Rect(0,HEIGHT/2,WIDTH,heightOfCrop));
          
          cv::Scalar const hsvLowBlue(78, 138, 31);
          cv::Scalar const hsvHiBlue(157, 255, 200);
          cv::Scalar const hsvLowYellow(15, 50, 140);
          cv::Scalar const hsvHiYellow(25, 210, 255);
          cv::Scalar const hsvRedLow(170, 120, 100);
          cv::Scalar const hsvRedHigh(180, 180, 160);
        
          cv::Mat hsv, blueCones, yellowCones, redCones;
          cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
          cv::inRange(hsv, hsvLowBlue, hsvHiBlue, blueCones);
          cv::inRange(hsv, hsvLowYellow, hsvHiYellow, yellowCones);
          cv::inRange(hsv, hsvRedLow, hsvRedHigh, redCones);
          conesList = std::vector<cv::Mat>{blueCones, yellowCones, redCones};
          
          
          uint32_t const roundsOfPreprocessing = 3;
          cv::Mat element = cv::getStructuringElement(
            cv::MORPH_RECT,cv::Size(3,3),cv::Point(-1,1));
          for (auto cones : conesList) {
            for (uint32_t i = 0; i < roundsOfPreprocessing; i++) {
              // removes noise in the background
              cv::morphologyEx(cones, cones, cv::MORPH_OPEN, element,
               cv::Point(-1,-1), 1, 1, 1); 
              // closes holes in the cones mask
              cv::morphologyEx(cones, cones, cv::MORPH_CLOSE, element,
                 cv::Point(-1,-1), 3, 1, 1); 
              // enlarges the segmented area with cones
              cv::morphologyEx(cones, cones, cv::MORPH_DILATE, element,
                 cv::Point(-1, -1), 2, 1, 1); 
            }
          }
        }        
        
        //Part II: Find boudning boxes
        // Bounding boxes for each cone:
        std::vector<std::vector<cv::Rect>> allBoundingRects(3, 
          std::vector<cv::Rect>()); //Blue, Yellow, Red
        {
          int32_t minAreaThres = 50;
          int32_t maxAreaThres = 5500;
          double widthToHeigthRatio = 0.9;
          for ( uint32_t i=0; i < 3; i++ ) {
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours( conesList[i], contours, hierarchy, 
              cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
            
            for ( uint32_t j=0; j < contours.size(); j++) {
              std::vector<cv::Point> contours_poly;
              
              cv::RotatedRect rectRotated = cv::minAreaRect(contours[j]);
              cv::Rect rect = rectRotated.boundingRect();
              
              if ( rect.width < widthToHeigthRatio*rect.height &&
                   (rect.area() > minAreaThres && rect.area() < maxAreaThres) ) 
              {
                allBoundingRects[i].push_back(rect);
              } 
            }
          }
        }
        
        //Part III: Find coords of boxes
        std::vector<std::vector<cv::Point>> allPositionsCones(3,
          std::vector<cv::Point>());
        std::vector<std::string> allXCoordsCones(3,
          std::string());
        std::vector<std::string> allYCoordsCones(3,
          std::string());
        {
          for ( uint32_t i=0; i < 3; i++ ) {
            for ( uint32_t j=0; j < allBoundingRects[i].size(); j++ ) {
              cv::Rect rect = allBoundingRects[i][j];
              cv::Point bottomCentre = cv::Point(
                static_cast<int>(rect.x + rect.width/2),
                static_cast<int>(rect.y + rect.height)
              );
              allPositionsCones[i].push_back(bottomCentre);
              allXCoordsCones[i] += std::to_string(
                bottomCentre.x - static_cast<int>(WIDTH/2))+",";
              allYCoordsCones[i] += std::to_string(
                static_cast<int>(heightOfCrop) - bottomCentre.y) + ",";
            }
          }
        }
        
        //Part IV: send messages
        {
          cluon::data::TimeStamp sampleTime;
          for ( uint32_t i=0; i < 3; i++ ) {
            opendlv::logic::perception::ConePosition conesMsg;
            conesMsg.color(i);
            conesMsg.x(allXCoordsCones[i]);
            conesMsg.y(allYCoordsCones[i]);
            
            if (VERBOSE) {
              std::cout<<"msg: i=" << conesMsg.color() <<", x=" << 
                conesMsg.x()<<", y=" <<conesMsg.y()<<std::endl;
            }
            od4.send(conesMsg, sampleTime, i);
          }
        }
        
        // Display image.
        if (VERBOSE) {
            // Original image:
            cv::Mat dispImage = cv::Mat::zeros(cv::Size(WIDTH, 3*heightOfCrop),
              CV_8UC4);
            cv::Rect ROI_1(0, 0, WIDTH ,heightOfCrop);
            img.copyTo(dispImage(ROI_1));
            
            // Masked image:
            // Apply cone masks to original image
            cv::Mat fullMask = conesList[0] + conesList[1] + conesList[2];
            cv::Mat output;
            bitwise_and(img,img,output,fullMask);
            
            cv::Rect ROI_2(0, heightOfCrop, WIDTH, heightOfCrop);
            output.copyTo(dispImage(ROI_2));
            
            // Cones with bounding boxes:
            std::vector<cv::Scalar> colors{cv::Scalar(255,0,0), 
              cv::Scalar(0,255,255), cv::Scalar(0,0,255)};
            for (uint32_t i=0; i<3; i++) {
              for (size_t j = 0; j < allBoundingRects[i].size(); j++) {
                rectangle(img, allBoundingRects[i][j], colors[i], 2);
                circle(img, allPositionsCones[i][j], 4, colors[i], -1, 8, 0 );
              }
            }
            
            cv::Rect ROI_3(0,2*heightOfCrop,WIDTH,heightOfCrop);
            img.copyTo(dispImage(ROI_3));
            imshow("Cone detection", dispImage);
            cv::waitKey(1);
        }
      }
    }
    
    retCode = 0;
  }
  return retCode;
}

