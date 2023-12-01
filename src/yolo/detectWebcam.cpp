#include "yolov8.h"

// Runs object detection on video stream then displays annotated results.

int main(int argc, char *argv[])
{
	// Parse the command line arguments
	if (argc != 2)
	{
		std::cout << "Error: Must specify the model path" << std::endl;
		std::cout << "Usage:   " << argv[0] << " <model.onnx>" << std::endl;
		std::cout << "Example: " << argv[0] << " ../models/yolov8m.onnx" << std::endl;
		return -1;
	}

	// Ensure the onnx model exists
	const std::string onnxModelPath = argv[1];
	if (!doesFileExist(onnxModelPath))
	{
		std::cout << "Error: Unable to find file at path: " << onnxModelPath << std::endl;
		return -1;
	}

	// Create our YoloV8 engine
	// Use default probability threshold, nms threshold, and top k
	YoloV8 yoloV8(onnxModelPath);

	// Initialize the video stream
	cv::VideoCapture cap;

	// open webcam
	cap.open(0);

	// Try to use HD resolution (or closest resolution
	auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

	if (!cap.isOpened())
		throw std::runtime_error("Unable to open video capture!");

	while (true)
	{
		// Grab frame
		cv::Mat img;
		cap >> img;

		if (img.empty())
		{
			throw std::runtime_error("Unable to decode image from video stream.");
		}

		// Run inference
		const auto objects = yoloV8.detectObjects(img);

		// Draw the bounding boxes on the image
		yoloV8.drawObjectLabels(img, objects);

		// Display the results
		cv::imshow("Object Detection", img);
		if (cv::waitKey(1) >= 0)
		{
			break;
		}
	}
	return 0;
}
