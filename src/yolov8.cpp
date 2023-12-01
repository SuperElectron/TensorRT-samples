#include <opencv2/cudaimgproc.hpp>
#include "yolov8.h"

YoloV8::YoloV8(
		const std::string &onnxModelPath,
		const float probabilityThreshold,
		const float nmsThreshold,
		const int topK)
		: PROBABILITY_THRESHOLD(probabilityThreshold), NMS_THRESHOLD(nmsThreshold), TOP_K(topK)
{
	// Specify options for GPU inference
	Options options;
	// YoloV8 models only supports a fixed batch size of 1
	options.doesSupportDynamicBatchSize = false;
	options.optBatchSize = 1;
	options.maxBatchSize = 1;
	options.maxWorkspaceSize = 2000000000;

	// Use FP16 precision to speed up inference
	options.precision = Precision::FP16;

	// Create our TensorRT inference engine
	trt_engine = std::make_unique<Engine>(options);

	// Build the onnx model into a TensorRT engine file (returns immediately if engine file already exists)
	auto succ = trt_engine->build(onnxModelPath);
	if (!succ)
	{
		const std::string errMsg = "Error: Unable to build the TensorRT engine. "
								   "Try increasing TensorRT log severity to kVERBOSE (in tensorrt-cpp-api/engine.cpp).";
		throw std::runtime_error(errMsg);
	}

	// Load the TensorRT engine file
	succ = trt_engine->loadNetwork();
	if (!succ)
		throw std::runtime_error("Error: Unable to load TensorRT engine weights into memory.");
}

std::vector <std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::Mat &img_BGR)
{
	// Upload the image to GPU memory
	cv::cuda::GpuMat gpu_img;
	gpu_img.upload(img_BGR);

	// Convert the input image to RGB (model expects it)
	cv::cuda::cvtColor(gpu_img, gpu_img, cv::COLOR_BGR2RGB);

	// Populate the input vectors
	const auto &input_dims = trt_engine->getInputDims();

	// Resize to the model expected input size while maintaining the aspect ratio with the use of padding
	auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpu_img, input_dims[0].d[1], input_dims[0].d[2]);

	// Convert to format expected by our inference engine
	std::vector <cv::cuda::GpuMat> input{std::move(resized)};
	std::vector <std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

	// These params will be used in the post-processing stage
	m_img_height = gpu_img.rows;
	m_img_width = gpu_img.cols;
	m_ratio = 1.f / std::min(input_dims[0].d[2] / static_cast<float>(gpu_img.cols),
							 input_dims[0].d[1] / static_cast<float>(gpu_img.rows));
	return inputs;
}

std::vector <Object> YoloV8::detectObjects(const cv::Mat &img_BGR)
{
	// Preprocess the input image
	const auto input = preprocess(img_BGR);

	// Run inference using the TensorRT engine
	std::vector <std::vector<std::vector<float>>> feature_vectors;
	auto succ = trt_engine->runInference(input, feature_vectors, SUB_VALS, DIV_VALS, NORMALIZE);
	if (!succ)
		throw std::runtime_error("Error: Unable to run inference.");

	// Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
	std::vector<float> feature_vector;
	transformOutput(feature_vectors, feature_vector);

	// Postprocess the output
	return postprocess(feature_vector);
}

void YoloV8::transformOutput(std::vector <std::vector<std::vector <float>>>& input,std::vector<float> &output)
{
	if (input.size() != 1 || input[0].size() != 1)
		throw std::logic_error("The feature vector has incorrect dimensions!");

	output = std::move(input[0][0]);
}

std::vector <Object> YoloV8::postprocess(std::vector<float> &featureVector)
{
	// get channel and anchors from the model engine output layer
	const auto &output_dims = trt_engine->getOutputDims();
	auto num_channels = output_dims[0].d[1];
	auto num_anchors = output_dims[0].d[2];

	// load number of classes from pre-defined yoloV8 list
	auto num_classes = class_names.size();

	std::vector <cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> labels;
	std::vector<int> indices;

	cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, featureVector.data());
	output = output.t();

	// Get all the YOLO proposals
	for (int x = 0; x < num_anchors; x++)
	{
		auto row_ptr = output.row(x).ptr<float>();
		auto bboxes_ptr = row_ptr;
		auto scores_ptr = row_ptr + 4;
		auto max_score_ptr = std::max_element(scores_ptr, scores_ptr + num_classes);
		float score = *max_score_ptr;
		if (score > PROBABILITY_THRESHOLD)
		{
			// get bbox coordinates
			float x = *bboxes_ptr++;
			float y = *bboxes_ptr++;
			float w = *bboxes_ptr++;
			float h = *bboxes_ptr;
			// convert coordinate to P1(xmin, ymin), P2(xmax, ymax)
			float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_img_width);
			float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_img_height);
			float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_img_width);
			float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_img_height);
			// reformat bbox, label, and score
			int label = max_score_ptr - scores_ptr;
			cv::Rect_<float> bbox;
			bbox.x = x0;
			bbox.y = y0;
			bbox.width = x1 - x0;
			bbox.height = y1 - y0;
			// store results
			bboxes.push_back(bbox);
			labels.push_back(label);
			scores.push_back(score);
		}
	}

	// Run NMS (non-maximum suppression)
	cv::dnn::NMSBoxes(bboxes, scores, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

	// Choose the top k detections
	std::vector <Object> objects;
	int counter = 0;
	for (auto &chosenIdx: indices)
	{
		// if we have reached the maximum amount of desired detections, exit
		if (counter >= TOP_K)
			break;
		// capture objects
		Object obj{};
		obj.probability = scores[chosenIdx];
		obj.label = labels[chosenIdx];
		obj.rect = bboxes[chosenIdx];
		objects.push_back(obj);

		counter += 1;
	}
	return objects;
}

void YoloV8::drawObjectLabels(cv::Mat &image, const std::vector <Object> &objects, unsigned int scale)
{
	for (auto &object: objects)
	{
		// color to base text and bounding box from (from a classified list for each yoloV8 object)
		cv::Scalar color = cv::Scalar(
				COLOR_LIST[object.label][0],
				COLOR_LIST[object.label][1],
				COLOR_LIST[object.label][2]
				);

		// set text color to black or white (based on mean of bbox color)
		float mean_color = cv::mean(color)[0];
		cv::Scalar text_color;
		if (mean_color > 0.5)
			text_color = cv::Scalar(0, 0, 0);
		else
			text_color = cv::Scalar(255, 255, 255);

		// set the display text at top left corner of the bounding box
		char text[256];
		sprintf(text, "%s %.1f%%", class_names[object.label].c_str(), object.probability * 100);
		// position and color the display text
		int baseline = 0;
		cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseline);
		cv::Scalar text_background_color = color * 0.7 * 255;

		// set bounding boxes from the detected
		const auto &rect = object.rect;
		int x = object.rect.x;
		int y = object.rect.y + 1;
		cv::rectangle(image, rect, color * 255, scale + 1);
		// write the background color for the display text
		cv::rectangle(image,
					  cv::Rect(cv::Point(x, y),cv::Size(labelSize.width,labelSize.height + baseline)),
					  text_background_color,
					  -1);
		// write the display text
		cv::putText(image,
					text,
					cv::Point(x, y + labelSize.height),
					cv::FONT_HERSHEY_SIMPLEX,
					0.35 * scale,
					text_color,
					scale);
	}
}