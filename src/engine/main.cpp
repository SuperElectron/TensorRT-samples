#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main(int argc, char *argv[])
{

	if (argc != 3)
	{
		std::cout << "Error: Must specify the model path" << std::endl;
		std::cout << "Usage:   " << argv[0] << " /path/to/onnx/model.onnx /path/to/image.jpg" << std::endl;
		std::cout << "Example: " << argv[0] << " /src/models/yolov8m.onnx /src/images/cars.jpg" << std::endl;
		return EXIT_FAILURE;
	}

	// Specify our GPU inference configuration options
	Options options;
	// load basic options for the model
	options.doesSupportDynamicBatchSize = false;
	options.precision = Precision::FP16; // Use fp16 precision for faster inference.

	if (options.doesSupportDynamicBatchSize)
	{
		options.optBatchSize = 4;
		options.maxBatchSize = 16;
	} else
	{
		options.optBatchSize = 1;
		options.maxBatchSize = 1;
	}

	Engine engine(options);

	// Must specify a dynamic batch size when exporting the model from onnx.
	// If model only specifies a static batch size, must set the above variable doesSupportDynamicBatchSize to false.
	const std::string onnxModelpath = argv[1];

	bool succ = engine.build(onnxModelpath);
	if (!succ)
	{
		throw std::runtime_error("Unable to build TRT engine.");
	}

	succ = engine.loadNetwork();
	if (!succ)
	{
		throw std::runtime_error("Unable to load TRT engine.");
	}

	// Let's use a batch size which matches that which we set the Options.optBatchSize option
	size_t batchSize = options.optBatchSize;

	const std::string inputImage = argv[2];
	auto cpuImg = cv::imread(inputImage);
	if (cpuImg.empty())
	{
		throw std::runtime_error("Unable to read image at path: " + inputImage);
	}

	// The model expects RGB input
	cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

	// Upload to GPU memory
	cv::cuda::GpuMat img;
	img.upload(cpuImg);

	// Populate the input vectors
	const auto &inputDims = engine.getInputDims();
	std::vector <std::vector<cv::cuda::GpuMat>> inputs;

	// feeding the same image to all the inputs
	for (const auto &inputDim: inputDims)
	{
		std::vector <cv::cuda::GpuMat> input;
		for (size_t j = 0; j < batchSize; ++j)
		{
			cv::cuda::GpuMat resized;

			// TODO: Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio
			// TRT dims are (height, width) whereas OpenCV is (width, height)
			cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1]));
			input.emplace_back(std::move(resized));
		}
		inputs.emplace_back(std::move(input));
	}

	// Preprocessing
	// The default Engine::runInference method normalizes values [0.f, 1.f]
	// bool normalize=false will leave values between [0.f, 255.f])
	std::array<float, 3> subVals{0.5f, 0.5f, 0.5f};
	std::array<float, 3> divVals{0.5f, 0.5f, 0.5f};
	bool normalize = true;

	// Discard the first inference time as it takes longer
	std::vector < std::vector < std::vector < float>>> featureVectors;
	succ = engine.runInference(inputs, featureVectors, subVals, divVals, normalize);
	if (!succ)
		throw std::runtime_error("Unable to run inference.");

	// Print the feature vectors
	for (size_t batch = 0; batch < featureVectors.size(); ++batch)
	{
		for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum)
		{
			std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
			int i = 0;
			for (const auto &e: featureVectors[batch][outputNum])
			{
				std::cout << e << " ";
				if (++i == 10)
				{
					std::cout << "...";
					break;
				}
			}
			std::cout << "\n" << std::endl;
		}
	}

	return EXIT_SUCCESS;
}
