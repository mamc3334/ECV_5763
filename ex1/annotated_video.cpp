/*
 *
 *  Author: Mason McGaffin
 *
 *  Derived from example created by Sam Siewert 
 *
 *  Verify your hardware and OS configuration with:
 *  1) lsusb
 *  2) ls -l /dev/video*
 *  3) dmesg | grep UVC
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define ESCAPE_KEY (27)
#define FRAME_WIDTH (320)
#define FRAME_HEIGHT (240)

#define BORDER_WIDTH (4)
Scalar BORDER_COLOR(0,255,255);

int main()
{
   VideoCapture cam0(0);
   namedWindow("annotated_video");
   char winInput;

   if (!cam0.isOpened())
   {
       exit(-1);
   }

   cam0.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
   cam0.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

   while (1)
   {
      Mat frame, frame_border;
      cam0.read(frame);

      //TODO: modify frame
      line(frame, Point(0,FRAME_HEIGHT/2), Point(FRAME_WIDTH,FRAME_HEIGHT/2), BORDER_COLOR, 1);
      line(frame, Point(FRAME_WIDTH/2,0), Point(FRAME_WIDTH/2,FRAME_HEIGHT), BORDER_COLOR, 1);
      copyMakeBorder(frame, frame_border, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, BORDER_CONSTANT, BORDER_COLOR);

      imshow("annotated_video", frame_border);

      if ((winInput = waitKey(10)) == ESCAPE_KEY) break;
   }

   destroyWindow("annotated_video"); 
};
