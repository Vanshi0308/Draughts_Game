#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
#include <math.h>
using namespace std::experimental::filesystem::v1;
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;

int parametricIntersect(float r1, float t1, float r2, float t2, int& x, int& y);
// Data provided:  Filename, White pieces, Black pieces
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the images.
const string GROUND_TRUTH_FOR_BOARD_IMAGES[][3] = {
	{"DraughtsGame1Move0.JPG", "1,2,3,4,5,6,7,8,9,10,11,12", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move1.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move2.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move3.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move4.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "17,20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move5.JPG", "1,2,3,4,5,7,8,9,10,11,12,22", "20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move6.JPG", "1,2,3,4,5,7,8,9,10,11,12", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move7.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move8.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move9.JPG", "1,2,3,4,5,7,8,10,11,12,22", "20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move10.JPG", "1,2,3,4,5,7,8,10,11,12", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move11.JPG", "1,2,3,4,5,7,8,10,11,16", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move12.JPG", "1,2,3,4,5,7,8,10,11,16", "14,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move13.JPG", "1,2,3,4,5,7,8,11,16,17", "20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move14.JPG", "1,2,3,4,5,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move15.JPG", "1,3,4,5,6,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move16.JPG", "1,3,4,5,6,7,8,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move17.JPG", "1,3,4,5,7,8,9,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move18.JPG", "1,3,4,5,7,8,9,11,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move19.JPG", "1,3,4,5,7,8,9,15,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move20.JPG", "1,3,4,5,8,9,16", "K2,14,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move21.JPG", "1,3,4,5,8,16,18", "K2,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move22.JPG", "1,3,4,5,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move23.JPG", "1,4,5,7,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move24.JPG", "1,4,5,7,8", "K2,11,14,27,28,29,31,32"},
	{"DraughtsGame1Move25.JPG", "1,4,5,8,16", "K2,14,27,28,29,31,32"},
	{"DraughtsGame1Move26.JPG", "1,4,5,8,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move27.JPG", "1,4,5,11,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move28.JPG", "1,4,5,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move29.JPG", "4,5,6,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move30.JPG", "4,5,6,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move31.JPG", "4,5,9,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move32.JPG", "4,5,9,11,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move33.JPG", "4,5,11,14,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move34.JPG", "4,5,11,14,16", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move35.JPG", "4,5,11,16,17", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move36.JPG", "4,5,11,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move37.JPG", "4,5,15,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move38.JPG", "4,5,15,16,17", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move39.JPG", "4,5,15,17,19", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move40.JPG", "4,5,15,17,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move41.JPG", "4,5,17,18,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move42.JPG", "4,5,17,18,19", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move43.JPG", "4,5,17,19,22", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move44.JPG", "4,5,17,19,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move45.JPG", "4,5,19,21,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move46.JPG", "4,5,19,21,22", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move47.JPG", "4,5,19,22,25", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move48.JPG", "4,5,19,25", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move49.JPG", "4,5,19,K30", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move50.JPG", "4,5,19,K30", "K2,20,K26,27,28,29,32"},
	{"DraughtsGame1Move51.JPG", "4,5,19,K23", "K2,20,27,28,29,32"},
	{"DraughtsGame1Move52.JPG", "4,5,19", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move53.JPG", "4,5,23", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move54.JPG", "4,5,23", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move55.JPG", "4,5,26", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move56.JPG", "4,5,26", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move57.JPG", "4,5,K31", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move58.JPG", "4,5,K31", "K2,11,20,27,28,29"},
	{"DraughtsGame1Move59.JPG", "4,5,K24", "K2,11,20,28,29"},
	{"DraughtsGame1Move60.JPG", "4,5", "K2,11,19,20,29"},
	{"DraughtsGame1Move61.JPG", "4,9", "K2,11,19,20,29"},
	{"DraughtsGame1Move62.JPG", "4,9", "K2,11,19,20,25"},
	{"DraughtsGame1Move63.JPG", "4,14", "K2,11,19,20,25"},
	{"DraughtsGame1Move64.JPG", "4", "K2,11,15,19,20"},
	{"DraughtsGame1Move65.JPG", "8", "K2,11,15,19,20,29"},
	{"DraughtsGame1Move66.JPG", "", "K2,K4,15,19,20,29"}
};

// Data provided:  Approx. frame number, From square number, To square number
// Note that the first move is a White move (and then the moves alternate Black, White, Black, White...)
// This data corresponds to the video:  DraughtsGame1.avi
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the video.
const int GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[][3] = {
{ 17, 9, 13 },
{ 37, 24, 20 },
{ 50, 6, 9 },
{ 65, 22, 17 },
{ 85, 13, 22 },
{ 108, 26, 17 },
{ 123, 9, 13 },
{ 161, 30, 26 },
{ 180, 13, 22 },
{ 201, 25, 18 },
{ 226, 12, 16 },
{ 244, 18, 14 },
{ 266, 10, 17 },
{ 285, 21, 14 },
{ 308, 2, 6 },
{ 326, 26, 22 },
{ 343, 6, 9 },
{ 362, 22, 18 },
{ 393, 11, 15 },
{ 433, 18, 2 },
{ 453, 9, 18 },
{ 472, 23, 14 },
{ 506, 3, 7 },
{ 530, 20, 11 },
{ 546, 7, 16 },
{ 582, 2, 7 },
{ 617, 8, 11 },
{ 641, 27, 24 },
{ 673, 1, 6 },
{ 697, 7, 2 },
{ 714, 6, 9 },
{ 728, 14, 10 },
{ 748, 9, 14 },
{ 767, 10, 7 },
{ 781, 14, 17 },
{ 801, 7, 3 },
{ 814, 11, 15 },
{ 859, 24, 20 },
{ 870, 16, 19 },
{ 891, 3, 7 },
{ 923, 15, 18 },
{ 936, 7, 10 },
{ 955, 18, 22 },
{ 995, 10, 14 },
{ 1014, 17, 21 },
{ 1034, 14, 17 },
{ 1058, 21, 25 },
{ 1075, 17, 26 },
{ 1104, 25, 30 },
{ 1129, 31, 27 },
{ 1147, 30, 23 },
{ 1166, 27, 18 },
{ 1182, 19, 23 },
{ 1201, 18, 15 },
{ 1213, 23, 26 },
{ 1243, 15, 11 },
{ 1266, 26, 31 },
{ 1280, 32, 27 },
{ 1298, 31, 24 },
{ 1324, 28, 19 },
{ 1337, 5, 9 },
{ 1358, 29, 25 },
{ 1387, 9, 14 },
{ 1450, 25, 15 },
{ 1465, 4, 8 },
{ 1490, 11, 4 }
};


#define EMPTY_SQUARE 0
#define WHITE_MAN_ON_SQUARE 1
#define BLACK_MAN_ON_SQUARE 3
#define WHITE_KING_ON_SQUARE 2
#define BLACK_KING_ON_SQUARE 4
#define NUMBER_OF_SQUARES_ON_EACH_SIDE 8
#define NUMBER_OF_SQUARES (NUMBER_OF_SQUARES_ON_EACH_SIDE*NUMBER_OF_SQUARES_ON_EACH_SIDE/2)


class DraughtsBoard
{
private:
	int mBoardGroundTruth[NUMBER_OF_SQUARES];
	Mat mOriginalImage;
	void loadGroundTruth(string pieces, int man_type, int king_type);
public:
	DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth);
};

DraughtsBoard::DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth)
{
	for (int square_count = 1; square_count <= NUMBER_OF_SQUARES; square_count++)
	{
		mBoardGroundTruth[square_count - 1] = EMPTY_SQUARE;
	}
	loadGroundTruth(white_pieces_ground_truth, WHITE_MAN_ON_SQUARE, WHITE_KING_ON_SQUARE);
	loadGroundTruth(black_pieces_ground_truth, BLACK_MAN_ON_SQUARE, BLACK_KING_ON_SQUARE);
	string full_filename = "Media/" + filename;
	mOriginalImage = imread(full_filename, -1);
	if (mOriginalImage.empty())
		cout << "Cannot open image file: " << full_filename << endl;
	else imshow(full_filename, mOriginalImage);
}

void DraughtsBoard::loadGroundTruth(string pieces, int man_type, int king_type)
{
	int index = 0;
	while (index < pieces.length())
	{
		bool is_king = false;
		if (pieces.at(index) == 'K')
		{
			is_king = true;
			index++;
		}
		int location = 0;
		while ((index < pieces.length()) && (pieces.at(index) >= '0') && (pieces.at(index) <= '9'))
		{
			location = location * 10 + (pieces.at(index) - '0');
			index++;
		}
		index++;
		if ((location > 0) && (location <= NUMBER_OF_SQUARES))
			mBoardGroundTruth[location - 1] = (is_king) ? king_type : man_type;
	}
}

class Histogram
{
protected:
	Mat mImage;
	int mNumberChannels;
	int* mChannelNumbers;
	int* mNumberBins;
	float mChannelRange[2];
public:
	Histogram(Mat image, int number_of_bins)
	{
		mImage = image;
		mNumberChannels = mImage.channels();
		mChannelNumbers = new int[mNumberChannels];
		mNumberBins = new int[mNumberChannels];
		mChannelRange[0] = 0.0;
		mChannelRange[1] = 255.0;
		for (int count = 0; count < mNumberChannels; count++)
		{
			mChannelNumbers[count] = count;
			mNumberBins[count] = number_of_bins;
		}
		//ComputeHistogram();
	}
	virtual void ComputeHistogram() = 0;
	virtual void NormaliseHistogram() = 0;
	static void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
	{
		int number_of_bins = histograms[0].size[0];
		double max_value = 0, min_value = 0;
		double channel_max_value = 0, channel_min_value = 0;
		for (int channel = 0; (channel < number_of_histograms); channel++)
		{
			minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
			max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
			min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
		}
		float scaling_factor = ((float)256.0) / ((float)number_of_bins);

		Mat histogram_image((int)(((float)number_of_bins) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) + 1, CV_8UC3, Scalar(255, 255, 255));
		display_image = histogram_image;
		line(histogram_image, Point(0, 0), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
		line(histogram_image, Point(histogram_image.cols - 1, histogram_image.rows - 1), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
		int highest_point = static_cast<int>(0.9 * ((float)number_of_bins) * scaling_factor);
		for (int channel = 0; (channel < number_of_histograms); channel++)
		{
			int last_height;
			for (int h = 0; h < number_of_bins; h++)
			{
				float value = histograms[channel].at<float>(h);
				int height = static_cast<int>(value * highest_point / max_value);
				int where = (int)(((float)h) * scaling_factor);
				if (h > 0)
					line(histogram_image, Point((int)(((float)(h - 1)) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) - last_height),
						Point((int)(((float)h) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) - height),
						Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
				last_height = height;
			}
		}
	}
};

class ColourHistogram : public Histogram
{
private:
	MatND mHistogram;
public:
	ColourHistogram(Mat all_images[], int number_of_images, int number_of_bins) :
		Histogram(all_images[0], number_of_bins)
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		for (int index = 0; index < number_of_images; index++)
			calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges, true, true);
	}
	ColourHistogram(Mat image, int number_of_bins) :
		Histogram(image, number_of_bins)
	{
		ComputeHistogram();
	}
	void ComputeHistogram()
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges);
	}
	void NormaliseHistogram()
	{
		normalize(mHistogram, mHistogram, 1.0);
	}
	Mat BackProject(Mat& image)
	{
		Mat& result = image.clone();
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcBackProject(&image, 1, mChannelNumbers, mHistogram, result, channel_ranges, 255.0);
		return result;
	}
	MatND getHistogram()
	{
		return mHistogram;
	}
};

//Function to find the intersection point of two lines using their parametric representation
int parametricIntersect(float r1, float t1, float r2, float t2, int& x, int& y)
{
	float ct1 = cosf(t1);  
	float st1 = sinf(t1);   
	float ct2 = cosf(t2);
	float st2 = sinf(t2);   
	float d = ct1 * st2 - st1 * ct2;        
	if (d != 0.0f) {
		x = (int)((st2 * r1 - st1 * r2) / d);
		y = (int)((-ct2 * r1 + ct1 * r2) / d);
		return(1);
	}
	else {
		return(0);
	}
}

void MyApplication()
{
	string video_filename("Media/DraughtsGame1.avi");
	VideoCapture video;
	video.open(video_filename);

	int pieces[32];
	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, -1);
	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, -1);
	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, -1);
	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, -1);
	string background_filename("Media/DraughtsGame1EmptyBoard.JPG");
	Mat static_background_image = imread(background_filename, -1);
	if ((!video.isOpened()) || (black_pieces_image.empty()) || (white_pieces_image.empty()) || (black_squares_image.empty()) || (white_squares_image.empty())  || (static_background_image.empty()))
	{
		// Error attempting to load something.
		if (!video.isOpened())
			cout << "Cannot open video file: " << video_filename << endl;
		if (black_pieces_image.empty())
			cout << "Cannot open image file: " << black_pieces_filename << endl;
		if (white_pieces_image.empty())
			cout << "Cannot open image file: " << white_pieces_filename << endl;
		if (black_squares_image.empty())
			cout << "Cannot open image file: " << black_squares_filename << endl;
		if (white_squares_image.empty())
			cout << "Cannot open image file: " << white_squares_filename << endl;
		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
	}
	else
	{
		// Sample loading of image and ground truth
		int image_index = 21;
		DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);

		string draughts_game_filename("Media/DraughtsGame1Move0.jpg");
		Mat draughts_game_image = imread(draughts_game_filename, -1);

		if (draughts_game_image.empty())
		{
			cout << "Couldn't open DraughtsGame1 image file" << endl;
		}

		Mat hls_draughts_game_image;
		cvtColor(draughts_game_image, hls_draughts_game_image, cv::COLOR_BGR2HLS);

		// Part-1
		Mat hls_white_squares_image, hls_black_squares_image, hls_white_piece, hls_black_piece;
		Mat output1, output2;
		cvtColor(white_squares_image, hls_white_squares_image, COLOR_BGR2HLS);
		cvtColor(black_squares_image, hls_black_squares_image, COLOR_BGR2HLS);
		cvtColor(white_pieces_image, hls_white_piece, COLOR_BGR2HLS);
		cvtColor(black_pieces_image, hls_black_piece, COLOR_BGR2HLS);
		ColourHistogram hist_white_square_3D(hls_white_squares_image, 8);
		ColourHistogram hist_black_square_3D(hls_black_squares_image, 8);
		ColourHistogram hist_white_piece_3D(hls_white_piece, 8);
		ColourHistogram hist_black_piece_3D(hls_black_piece, 8);
		hist_white_square_3D.NormaliseHistogram();
		hist_black_square_3D.NormaliseHistogram();
		hist_white_piece_3D.NormaliseHistogram();
		hist_black_piece_3D.NormaliseHistogram();
		Mat back_proj_prob_white_square = hist_white_square_3D.BackProject(hls_draughts_game_image);
		Mat back_proj_prob_black_square = hist_black_square_3D.BackProject(hls_draughts_game_image);
		Mat back_proj_prob_white_piece = hist_white_piece_3D.BackProject(hls_draughts_game_image);
		Mat back_proj_prob_black_piece = hist_black_piece_3D.BackProject(hls_draughts_game_image);
		back_proj_prob_white_square = StretchImage(back_proj_prob_white_square);
		back_proj_prob_black_square = StretchImage(back_proj_prob_black_square);
		back_proj_prob_white_piece = StretchImage(back_proj_prob_white_piece);
		back_proj_prob_black_piece = StretchImage(back_proj_prob_black_piece);
		Mat back_proj_prob_white_square_display, back_proj_prob_black_square_display, back_proj_prob_white_piece_display, back_proj_prob_black_piece_display;
		/*Mat BGR_draughts_image, back_projection_probabilities_display;
		cvtColor(hls_draughts_game_image, BGR_draughts_image, COLOR_HLS2BGR);
		cvtColor(back_projection_probabilities, back_projection_probabilities_display, COLOR_GRAY2BGR);
		for(int row = 0; row < BGR_draughts_image.rows; row++)
			for (int col = 0; col < BGR_draughts_image.cols; col++)
			{
				uchar B = back_projection_probabilities_display.at<Vec3b>(row, col)[0];
				uchar G = back_projection_probabilities_display.at<Vec3b>(row, col)[1];
				uchar R = back_projection_probabilities_display.at<Vec3b>(row, col)[2];
				// double luminance_saturation_ratio = ((double)luminance) / ((double)saturation);
				if (B > 230 && G > 230 && R > 230)
				{
					BGR_draughts_image.at<Vec3b>(row, col)[0] = 0;
					BGR_draughts_image.at<Vec3b>(row, col)[1] = 0;
					BGR_draughts_image.at<Vec3b>(row, col)[2] = 255 * (B+G+R)/3;
				}		
			}*/
		
		cvtColor(back_proj_prob_white_square, back_proj_prob_white_square_display, COLOR_GRAY2BGR);
		cvtColor(back_proj_prob_black_square, back_proj_prob_black_square_display, COLOR_GRAY2BGR);
		cvtColor(back_proj_prob_white_piece, back_proj_prob_white_piece_display, COLOR_GRAY2BGR);
		cvtColor(back_proj_prob_black_piece, back_proj_prob_black_piece_display, COLOR_GRAY2BGR);

		Mat final_image;
		cvtColor(hls_draughts_game_image, final_image, COLOR_HLS2BGR);

		for (int i = 0; i < final_image.rows; i++) {
			for (int j = 0; j < final_image.cols; j++) {
				if (back_proj_prob_white_square_display.at<Vec3b>(i, j)[0] > 80 && back_proj_prob_white_square_display.at<Vec3b>(i, j)[1] > 80 && back_proj_prob_white_square_display.at<Vec3b>(i, j)[2] > 80)
				{
					final_image.at<Vec3b>(i, j)[0] = 0;
					final_image.at<Vec3b>(i, j)[1] = 0;
					final_image.at<Vec3b>(i, j)[2] = 255;
				}
				else if (back_proj_prob_black_square_display.at<Vec3b>(i, j)[0] > 10 && back_proj_prob_black_square_display.at<Vec3b>(i, j)[1] > 10 && back_proj_prob_black_square_display.at<Vec3b>(i, j)[2] > 10)
				{
					final_image.at<Vec3b>(i, j)[0] = 0;
					final_image.at<Vec3b>(i, j)[1] = 255;
					final_image.at<Vec3b>(i, j)[2] = 0;
				}
				else if (back_proj_prob_white_piece_display.at<Vec3b>(i, j)[0] > 50 && back_proj_prob_white_piece_display.at<Vec3b>(i, j)[1] > 50 && back_proj_prob_white_piece_display.at<Vec3b>(i, j)[2] > 50)
				{
					final_image.at<Vec3b>(i, j)[0] = 255;
					final_image.at<Vec3b>(i, j)[1] = 255;
					final_image.at<Vec3b>(i, j)[2] = 255;
				}
				else if(back_proj_prob_black_piece_display.at<Vec3b>(i, j)[0] > 50 && back_proj_prob_black_piece_display.at<Vec3b>(i, j)[1] > 50 && back_proj_prob_black_piece_display.at<Vec3b>(i, j)[2] > 50)
				{
					final_image.at<Vec3b>(i, j)[0] = 0;
					final_image.at<Vec3b>(i, j)[1] = 0;
					final_image.at<Vec3b>(i, j)[2] = 0;
				}
				else
				{
					final_image.at<Vec3b>(i, j)[0] = 255;
					final_image.at<Vec3b>(i, j)[1] = 0;
					final_image.at<Vec3b>(i, j)[2] = 0;
				}
			}
		}

		string ground_truth_filename("Media/DraughtsGame1Move0GroundTruth.jpg");
		Mat ground_truth_image = imread(ground_truth_filename, -1);

		if (ground_truth_image.empty())
		{
			cout << "Couldn't open DraughtsGame1GroundTruth image file" << endl;
		}

		Mat hls_ground_truth_image;
		cvtColor(ground_truth_image, hls_ground_truth_image, COLOR_BGR2HLS);
		ColourHistogram hist_ground_truth_3D(hls_ground_truth_image, 8);
		hist_ground_truth_3D.NormaliseHistogram();

		Mat hls_final_image;
		cvtColor(final_image, hls_final_image, COLOR_BGR2HLS);
		ColourHistogram hist_final_3D(hls_final_image, 8);
		hist_final_3D.NormaliseHistogram();

		double matching_score = compareHist(hist_final_3D.getHistogram(), hist_ground_truth_3D.getHistogram(), cv::HISTCMP_CORREL);
		char output[100];
		sprintf(output, "%.4f", matching_score);
		cout << "Comparison of ground truth with image segmented using Back Projection " << matching_score << endl;

		output1 = JoinImagesHorizontally(draughts_game_image, "Original Draughts Image", ground_truth_image, "Ground Truth Image", 4);
		output2 = JoinImagesHorizontally(output1, "", final_image, "Image with pixels classified", 4);
		imshow("Image Pixels Classification", output2);

		// Part-2
		vector<vector<double>> board_corners = { {114.0, 17.0}, {53.0, 245.0}, {355.0, 20.0}, {433.0, 241.0} };
		Mat flip_draughts_image, flip_prob_white_piece, flip_prob_black_piece;
		flip(draughts_game_image, flip_draughts_image, 0);
		flip(back_proj_prob_white_piece_display, flip_prob_white_piece, 0);
		flip(back_proj_prob_black_piece_display, flip_prob_black_piece, 0);
		imshow("Flipped Draughts Image", flip_draughts_image);

		vector<vector<double>> top_boundary_points = { {53.0, 245.0} };
		for (int i = 1; i <= 7; i++)
		{
			top_boundary_points.push_back({ (53.0 + i * (433.0 - 53.0) / 8), 243.0 });
		}
		top_boundary_points.push_back({ 433.0, 241.0 });

		vector<vector<double>> bottom_boundary_points = { {114.0, 17.0} };
		for (int i = 1; i <= 7; i++)
		{
			bottom_boundary_points.push_back({ (114.0 + i * (355.0 - 114.0) / 8), 18.5 });
		}
		bottom_boundary_points.push_back({ 355.0, 20.0 });

		vector<vector<double>> left_boundary_points = { {53.0, 245.0} };
		for (int i = 1; i <= 7; i++)
		{
			left_boundary_points.push_back({ (53.0 + i * (114.0 - 53.0) / 8), (245.0 + i * (17.0-245.0) / 8)});
		}
		left_boundary_points.push_back({ 114.0, 17.0 });

		vector<vector<double>> right_boundary_points = { {433.0, 241.0} };
		for (int i = 1; i <= 7; i++)
		{
			right_boundary_points.push_back({ (433.0 + i * (355.0 - 433.0) / 8), (241.0 + i * (20.0-241.0) / 8)});
		}
		right_boundary_points.push_back({ 355.0, 20.0 });

		vector<vector<double>> left_centre_points;
		for (int i = 0; i < 8; i++)
		{
			left_centre_points.push_back({ (left_boundary_points[i][0] + left_boundary_points[i + 1][0]) / 2, (left_boundary_points[i][1] + left_boundary_points[i + 1][1]) / 2 });
		}

		vector<vector<double>> right_centre_points;
		for (int i = 0; i < 8; i++)
		{
			right_centre_points.push_back({ (right_boundary_points[i][0] + right_boundary_points[i + 1][0]) / 2, (right_boundary_points[i][1] + right_boundary_points[i + 1][1]) / 2 });
		}

		vector<vector<vector<double>>> square_centre_points(8);
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
			{
				if (j == 0)
				{
					square_centre_points[i].push_back({left_centre_points[i][0] + (right_centre_points[i][0] - left_centre_points[i][0]) / 16, left_centre_points[i][1] + (right_centre_points[i][1] - left_centre_points[i][1]) / 16 });
				}
				else if (j > 0)
				{
					square_centre_points[i].push_back({square_centre_points[i][j - 1][0] + (right_centre_points[i][0] - left_centre_points[i][0]) / 8, square_centre_points[i][j - 1][1] + (right_centre_points[i][1] - left_centre_points[i][1]) / 8 });
				}
			}

		/*for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
			{
				cout << square_centre_points[i][j][0] << " " << square_centre_points[i][j][1] << endl;
			}*/

		vector<int> white_piece_squares, black_piece_squares;
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
			{
				if (i % 2 == 0 && j % 2 == 0)
				{
					if (flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[0] > 50 && flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[1] > 50 && flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[2] > 50)
					{
						white_piece_squares.push_back(4 - i / 2 + j * 4);
					}
				}
				else if (i % 2 == 1 && j % 2 == 1)
				{
					/*if (flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[0] > 50 && flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[1] > 50 && flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[2] > 50)
					{
						black_piece_squares.push_back(8 - (i - 1) / 2 + (j - 1) / 2 * 8);
					}*/
					// cout << (int)square_centre_points[i][j][0] << " " << (int)square_centre_points[i][j][1] << endl;
				}
			}

		sort(white_piece_squares.begin(), white_piece_squares.end());
		sort(black_piece_squares.begin(), black_piece_squares.end());
		int DRAUGHTS_IMAGE_NUMBER = 0;
		
		string white_piece_ground_truth, black_piece_ground_truth;
		for (int i = 0; i < white_piece_squares.size() - 1; i++)
		{
			white_piece_ground_truth += to_string(white_piece_squares[i]) + ", ";
		}

		white_piece_ground_truth += to_string(white_piece_squares.size() - 1);

		/*for (int i = 0; i < black_piece_squares.size() - 1; i++)
		{
			black_piece_ground_truth += to_string(black_piece_squares[i]) + ", ";
		}*/

		// black_piece_ground_truth += to_string(black_piece_squares.size() - 1);

		for (int i = 0; i < white_piece_squares.size(); i++)
		{
			cout << "White piece square" << endl;
			cout << white_piece_squares[i] << endl;
		}
		if (white_piece_ground_truth == GROUND_TRUTH_FOR_BOARD_IMAGES[DRAUGHTS_IMAGE_NUMBER][1])
		{
			cout << "White piece ground truth matches" << endl;
		}

		/*if (black_piece_ground_truth == GROUND_TRUTH_FOR_BOARD_IMAGES[DRAUGHTS_IMAGE_NUMBER][2])
		{
			cout << "Black piece ground truth matches" << endl;
		}*/

		// Part-4
		Mat draughts_canny_edge;
		Canny(draughts_game_image, draughts_canny_edge, 200, 150);
		imshow("Canny Edge Image", draughts_canny_edge);
		
		// Hough transform for (full) line detection
		vector<Vec2f> hough_lines;
		HoughLines(draughts_canny_edge, hough_lines, 1, PI / 200.0, 100);
		Mat hough_lines_image = draughts_game_image.clone();
		DrawLines(hough_lines_image, hough_lines);
		
		// Finding intersection points of all the lines to find corners using Hough Transform
		int intx, inty;
		for (vector<cv::Vec2f>::const_iterator line1 = hough_lines.begin();
			(line1 != hough_lines.end()); line1++)
			for (vector<cv::Vec2f>::const_iterator line2 = hough_lines.begin();
				(line2 != hough_lines.end()); line2++)
		{
			if (line1 != line2)
			{
				float rho1 = (*line1)[0];
				float theta1 = (*line1)[1];
				float rho2 = (*line2)[0];
				float theta2 = (*line2)[1];
				parametricIntersect(rho1, theta1, rho2, theta2, intx, inty);

				if (intx >= 0 && intx <= hough_lines_image.cols && inty >= 0 && inty <= hough_lines_image.rows)
				{
					circle(hough_lines_image, Point(intx, inty), 2, Scalar(0, 0, 255), FILLED, LINE_8);
				}
			}
		}

		imshow("Hough Transformation", hough_lines_image);

		// Contours and straight line segments
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(draughts_canny_edge, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		vector<Vec4i> line_segments;
		vector<vector<Point>> approx_contours(contours.size());
		for (int contour_number = 0; (contour_number < (int)contours.size()); contour_number++)
		{	// Approximate each contour as a series of line segments.
			approxPolyDP(Mat(contours[contour_number]), approx_contours[contour_number], 6, true);
		}
		// Extract line segments from the contours.
		for (int contour_number = 0; (contour_number < (int)contours.size()); contour_number++)
		{
			for (int line_segment_number = 0; (line_segment_number < (int)approx_contours[contour_number].size() - 1); line_segment_number++)
			{
				line_segments.push_back(Vec4i(approx_contours[contour_number][line_segment_number].x, approx_contours[contour_number][line_segment_number].y,
					approx_contours[contour_number][line_segment_number + 1].x, approx_contours[contour_number][line_segment_number + 1].y));
			}
		}
		// Draw the contours and then the segments
		Mat contour_line_segment_output;
		Mat contours_image = Mat::zeros(draughts_canny_edge.size(), CV_8UC3);
		Mat line_segments_image = Mat::zeros(draughts_canny_edge.size(), CV_8UC3);
		for (int contour_number = 0; (contour_number < (int)contours.size()); contour_number++)
		{
			Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
			drawContours(contours_image, contours, contour_number, colour, 1, 8, hierarchy);
		}
		DrawLines(line_segments_image, line_segments);

		for (vector<cv::Vec4i>::const_iterator current_line = line_segments.begin();
			(current_line != line_segments.end()); current_line++)
		{
			Point point1((*current_line)[0], (*current_line)[1]);
			Point point2((*current_line)[2], (*current_line)[3]);

			circle(line_segments_image, point1, 2, Scalar(0, 0, 255), FILLED, LINE_8);
			circle(line_segments_image, point2, 2, Scalar(0, 0, 255), FILLED, LINE_8);
		}

		Mat canny_edge_image_display;
		cvtColor(draughts_canny_edge, canny_edge_image_display, COLOR_GRAY2BGR);
		contour_line_segment_output = JoinImagesHorizontally(contours_image, "Contour Image", line_segments_image, "Line Segments", 4);
		imshow("Line segment extraction", contour_line_segment_output);

		Mat gray_draughts_image;
		Mat draughts_game_image_chessboard_corners = draughts_game_image.clone();
		Size pattern_size(7, 7);
		cvtColor(draughts_game_image, gray_draughts_image, COLOR_BGR2GRAY);
		vector<Point2f> corners;

		bool pattern_found = findChessboardCorners(gray_draughts_image, pattern_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ CALIB_CB_FAST_CHECK);
		cout << pattern_found << endl;

		drawChessboardCorners(draughts_game_image_chessboard_corners, pattern_size, Mat(corners), pattern_found);

		imshow("Corners detected using findChessboardCorners() function", draughts_game_image_chessboard_corners);

		// Part-3
		// Process video frame by frame
		Mat current_frame;
		video.set(cv::CAP_PROP_POS_FRAMES, 1);
		video >> current_frame;
		double last_time = static_cast<double>(getTickCount());
		double frame_rate = video.get(cv::CAP_PROP_FPS);
		double time_between_frames = 1000.0 / frame_rate;
		bool frame_to_process = false;
		while (!current_frame.empty())
		{
			frame_to_process = false;
			double current_time = static_cast<double>(getTickCount());
			double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
			int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
			last_time = current_time;
			imshow("Draughts video", current_frame);
			video >> current_frame;

			Mat s1, draughts_game_image_resized;
			resize(draughts_game_image, draughts_game_image_resized, current_frame.size());
			absdiff(current_frame, draughts_game_image_resized, s1);
			s1.convertTo(s1, CV_32F);  
			s1 = s1.mul(s1);          

			Scalar s = sum(s1);       

			double sse = s.val[0] + s.val[1] + s.val[2];


			double mse = sse / (double)(current_frame.channels() * current_frame.total());
			
			if (mse < 950)
			{
				frame_to_process = true;
			}
		}

		// Part-5
		vector<int> white_men_squares, black_men_squares, white_king_squares, black_king_squares;
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
			{
				if (i % 2 == 0 && j % 2 == 0)
				{
					if (flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[0] > 50 && flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[1] > 50 && flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[2] > 50)
					{
						if (flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1] - 10)[0] > 20 && flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1] - 10)[1] > 20 && flip_prob_white_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1] - 10)[2] > 20)
						{
							white_king_squares.push_back(4 - i / 2 + j * 4);
						}
						else
						{
							white_men_squares.push_back(4 - i / 2 + j * 4);
						}
					}
				}
				else if (i % 2 == 1 && j % 2 == 1)
				{
					if (flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[0] > 50 && flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[1] > 50 && flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1])[2] > 50)
					{
						if (flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1] - 10)[0] > 20 && flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1] - 10)[1] > 20 && flip_prob_black_piece.at<Vec3b>((int)square_centre_points[i][j][0], (int)square_centre_points[i][j][1] - 10)[2] > 20)
						{
							black_king_squares.push_back(8 - (i - 1) / 2 + (j - 1) / 2 * 8);
						}
						else
						{
							black_men_squares.push_back(8 - (i - 1) / 2 + (j - 1) / 2 * 8);
						}
					}
					cout << (int)square_centre_points[i][j][0] << " " << (int)square_centre_points[i][j][1] << endl;
				}
			}

		sort(white_men_squares.begin(), white_men_squares.end());
		sort(black_men_squares.begin(), black_men_squares.end());
		sort(white_king_squares.begin(), white_king_squares.end());
		sort(black_king_squares.begin(), black_king_squares.end());
	}
}
