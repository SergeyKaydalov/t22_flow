#include <unistd.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include "external/PX4-OpticalFlow/include/flow_opencv.hpp"

using namespace cv;
using namespace std;

static constexpr float lens_f = 2.1f; // Focal distance
static constexpr float lens_x = 5.6f; // Widht of the sensor
static constexpr float lens_y = 3.1f; // height of the sensor
static constexpr int    img_x  = 1280/4; // Image resolution along X
static constexpr int    img_y  = 720/4;  // Image resolution along Y
static constexpr int    fps    = 60; // Frame rate
static constexpr float  f_length_x =  lens_f * (float)img_x / lens_x;
static constexpr float  f_length_y =  lens_f * (float)img_x / lens_y;

int main(int argc, char** argv)
{
    const string about = "T22 optical flow detector.\n";
    const string keys =
        "{ h help |      | print this help message }"
        "{ @image | vtest.avi | path to image file }";
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string filename = samples::findFile(parser.get<string>("@image"));
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    VideoCapture capture(filename);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }


    OpticalFlowOpenCV oflow(f_length_x, f_length_y, 30, img_x, img_y);

    float xmax = -1, xmin = 1;
    float ymax = -1, ymin = 1;

    while(true)
    {
        Mat frame, frame_ds, frame_grey;

        capture >> frame;
        if (frame.empty())
            break;
        uint64_t ts = capture.get(CAP_PROP_POS_MSEC)*1000;

        cvtColor(frame, frame_grey, COLOR_BGR2GRAY);
        resize(frame_grey, frame_ds, Size(img_x, img_y), INTER_NEAREST);

        int dt;
        float oflow_x = 0, oflow_y = 0;
        int oprob = oflow.calcFlow(frame_ds.data, ts, dt, oflow_x, oflow_y);

        if (oprob >= 0)
        {
            if (oflow_x > xmax)
                xmax = oflow_x;
            if (oflow_x < xmin)
                xmin = oflow_x;

            if (oflow_y > ymax)
                ymax = oflow_y;
            if (oflow_y < ymin)
                ymin = oflow_y;

            printf("Prob: %d, flow_x: %.4f/%.4f/%.4f, flow_y: %.4f/%.4f/%.4f\n", oprob, oflow_x,xmin,xmax, oflow_y,ymin,ymax);
        }
	usleep(10000);

    }
}
