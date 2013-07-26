#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

extern "C" {

cv::Point2f computeIntersect(cv::Vec4i a,
                             cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
	float denom;

	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
		pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
	Mat& mGr = *(Mat*) addrGray;
	Mat& mRgb = *(Mat*) addrRgba;

	/*
	vector<KeyPoint> v;

	FastFeatureDetector detector(50);
	detector.detect(mGr, v);
	for (unsigned int i = 0; i < v.size(); i++) {
		const KeyPoint& kp = v[i];
		circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255, 0, 0, 255));
	}
	*/

	// Perform edge detect on the black and white image buffer.
	cv::Mat mInt = mGr.clone();
	//cv::blur(mGr, mGr, cv::Size(3, 3));
	cv::Canny(mGr, mInt, 100, 100, 3);

	// Convert and output back to the color image buffer for display.
	cv::cvtColor(mInt, mRgb, CV_GRAY2RGBA, 4);

	// Crop off the edges because we detect a line along the screen edge if we don't.
	//cv::Mat dst = src.clone();
	int padding = 10;
	cv::Rect myROI(padding, padding,
			mGr.size().width - (padding * 2),
			mGr.size().height - (padding * 2));
	IplImage img = (IplImage)mInt;
	//Mat image_roi = img(myROI);
	cv::Mat* croppedImage = new Mat(mInt, myROI);

	// Detect lines.
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(*croppedImage, lines,
			1, // rho – Distance resolution of the accumulator in pixels.
			CV_PI/180, // theta – Angle resolution of the accumulator in radians
			140, // threshold – Accumulator threshold parameter. Only those lines are returned that get enough votes ( >\texttt{threshold} ).
			60, // minLineLength – Minimum line length. Line segments shorter than that are rejected.
			10 // maxLineGap – Maximum allowed gap between points on the same line to link them.
			);

	// Draw detected lines in green.
	for (int i = 0; i < lines.size(); i++)
	{
		cv::Vec4i v = lines[i];
		// Add the padding back to make up for the cropping.
		cv::Point start = cv::Point(v[0] + padding, v[1] + padding);
		cv::Point end = cv::Point(v[2] + padding, v[3] + padding);
		cv::line(mRgb, start, end, CV_RGB(0,255,0),
				4); // thickness
	}

	// Detect intersections.
	std::vector<cv::Point2f> corners;
	for (int i = 0; i < lines.size(); i++)
	{
		for (int j = i+1; j < lines.size(); j++)
		{
			cv::Point2f pt = computeIntersect(lines[i], lines[j]);
			if (pt.x >= 0 && pt.y >= 0)
				corners.push_back(pt);
		}
	}

	// Circle intersections.
		for (int i = 0; i < corners.size(); i++)
		{
			cv::Point2f p = corners[i];
			cv::Point2f adjustedP = cv::Point(p.x + padding, p.y + padding);
			cv::circle(mRgb, adjustedP,
					6, //radius
					CV_RGB(255,0,0),
					4); // thickness
		}

	// Find top and bottom.
	if ( corners.size() > 0 ) {
		cv::Point2f highest = corners[0];
		cv::Point2f lowest = corners[0];
		for (int i = 0; i < corners.size(); i++)
		{
			cv::Point2f p = corners[i];
			if ( p.y > highest.y ) {
				highest = p;
			}
			if (p.y < lowest.y) {
				lowest = p;
			}
		}

		// Draw lines across screen at detected top and bottom in blue.
		cv::Point start = cv::Point(0, highest.y + padding);
		int highLine = highest.y + padding;
		cv::Point end = cv::Point(mGr.size().width, highLine);
		cv::line(mRgb, start, end, CV_RGB(0,0,255), 4);

		cv::Point start2 = cv::Point(0, lowest.y + padding);
		int lowLine = lowest.y + padding;
		cv::Point end2 = cv::Point(mGr.size().width, lowLine);
		cv::line(mRgb, start2, end2, CV_RGB(0,0,255), 4);

		// Output the pixel size.
		int pixelLength = highLine - lowLine;
		std::stringstream readout;
		readout << " 5\" = ";
		readout << pixelLength;
		readout << " pixels ";

		cv::Point center = cv::Point(mGr.size().width / 2, mGr.size().height / 2);
		int lineType = 8;
		cv::putText(mRgb, readout.str(),
				center, // bottom left corner
				0, // font type
		        2, // scale
		        CV_RGB(0,0,255),
		        2, // line thickness
		        lineType);

	}

}
}


