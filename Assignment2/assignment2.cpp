#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;

int main(int argc, char** argv)
{
	char * imageName = argv[1];

	cv::Mat outputImage;
	cv::Mat image = cv::imread(imageName, cv::IMREAD_GRAYSCALE);
	
	if (argc != 2) //|| image.data	)
	{
		cout << "Image not found" << endl;
		return 0;
	}

	cv::equalizeHist(image, outputImage);
	
	string originalWindowTitle = "Original Image";
	string outputWindowTitle = "Output Image";

	cv::namedWindow(originalWindowTitle);
	cv::imshow(originalWindowTitle, image);
	cv::namedWindow(outputWindowTitle);
	cv::imshow(outputWindowTitle, outputImage);
	
	cv::waitKey(0);
	
	return 0;
}