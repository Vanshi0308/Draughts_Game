# Draughts Game

#### By Vanshika Sinha

This project aimed to develop a system to automatically analyse a game of draughts (also known as checkers). The board  has a size of 8 squares by 8 squares and 69 static images of the board of the game in play, empty board images and a video of the game in process are used. The project is coded in C++ using the OpenCV library. 

## Part-1

All the pixels in the image are classified as white piece, black piece, white square, black square or none of these. 

## Part-2

According to the locations of the four corners of the board, a black or white piece is determined in each square. The locations of the pieces are recorded and comparison is made to the provided ground truth for the 69 frames.

## Part-3

The video of the draughts game is processed by identifying appropriate frames. The moves made again are found and recorded using the PDN notation. 

## Part-4

The static images of the board and the locations of the four corners are determined by comparing the results of the 3 approaches: Hough transformation, contour following and straight line segment extraction and by using the findChessboardCorners() routine in OpenCV.

## Part-5

A distinction is made between normal pieces and kings. All the 67 static images are evaluated considering White Men, White Kings, Black Men and Black Kings as separate classes. 