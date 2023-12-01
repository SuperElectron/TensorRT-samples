#include <iostream>
#include <fstream>

#include "engine.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

std::string splitStringBeforeLastSlash(const std::string &str)
{
	size_t last_slash_index = str.rfind('/');
	// return either the current working directory or the substring with path
	if (last_slash_index == std::string::npos)
		return std::string("./");
	else
		return str.substr(0, last_slash_index);
}

void Logger::log(Severity severity, const char *msg) noexcept
{
	// Only log Warnings or more important.
	if (severity <= Severity::kWARNING)
	std::cout << msg << std::endl;

}

bool Engine::doesFileExist(const std::string &filepath)
{
	std::ifstream f(filepath.c_str());
	return f.good();
}


Engine::Engine(const Options &options) : m_options(options)
{
	if (!m_options.doesSupportDynamicBatchSize && (m_options.optBatchSize > 1 || m_options.maxBatchSize > 1))
	{
		std::cout << "Model does not support dynamic batch size, using optBatchSize and maxBatchSize of 1" << std::endl;
		m_options.optBatchSize = 1;
		m_options.maxBatchSize = 1;
	}
}

Engine::~Engine()
{
	// Free the GPU memory
	for (auto &buffer: m_buffers)
		checkCudaErrorCode(cudaFree(buffer));

	m_buffers.clear();
}

/**
 * @brief Build the network for an onnx model
 * @param options runtime options struct
 */
bool Engine::build(std::string onnx_model_path)
{
	// Only regenerate the engine file if it has not already been generated for the specified options
	m_engineName = serializeEngineOptions(m_options, onnx_model_path);
	std::cout << "Searching for engine file with name: " << m_engineName << std::endl;

	if (doesFileExist(m_engineName))
	{
		std::cout << "Engine found, not regenerating..." << std::endl;
		return true;
	}

	if (!doesFileExist(onnx_model_path))
	{
		throw std::runtime_error("Could not find model at path: " + onnx_model_path);
	}

	// Was not able to find the engine file, generate...
	std::cout << "Engine not found, generating. This could take a while..." << std::endl;

	// Create our engine builder.
	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
	if (!builder)
	{
		return false;
	}

	// Set the max supported batch size
	builder->setMaxBatchSize(m_options.maxBatchSize);

	// Define an explicit batch size and then create the network.
	// More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
	auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
	if (!network)
		return false;

	// Create a parser for reading the onnx file.
	auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
	if (!parser)
		return false;

	// We are going to first read the onnx file into memory, then pass that buffer to the parser.
	std::ifstream file(onnx_model_path, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size))
		throw std::runtime_error("Unable to read engine file");

	auto parsed = parser->parse(buffer.data(), buffer.size());
	if (!parsed)
		return false;

	auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
		return false;


	// Register a single optimization profile
	IOptimizationProfile *optimized_profile = builder->createOptimizationProfile();
	const int32_t num_inputs = network->getNbInputs();
	for (int32_t i = 0; i < num_inputs; ++i)
	{
		const auto input = network->getInput(i);
		const auto input_name = input->getName();
		const auto input_dims = input->getDimensions();
		int32_t input_channel = input_dims.d[1];
		int32_t input_height = input_dims.d[2];
		int32_t input_width = input_dims.d[3];

		// Specify the optimization profile
		optimized_profile->setDimensions(input_name, OptProfileSelector::kMIN, Dims4(1, input_channel, input_height, input_width));
		optimized_profile->setDimensions(input_name, OptProfileSelector::kOPT,
								  Dims4(m_options.optBatchSize, input_channel, input_height, input_width));
		optimized_profile->setDimensions(input_name, OptProfileSelector::kMAX,
								  Dims4(m_options.maxBatchSize, input_channel, input_height, input_width));
	}
	config->addOptimizationProfile(optimized_profile);

	config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

	if (m_options.precision == Precision::FP16)
		config->setFlag(BuilderFlag::kFP16);

	// CUDA stream used for profiling by the builder.
	cudaStream_t profile_stream;
	checkCudaErrorCode(cudaStreamCreate(&profile_stream));
	config->setProfileStream(profile_stream);

	// Build the engine
	std::unique_ptr <IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
	if (!plan)
		return false;

	// Write the engine to disk
	std::ofstream outfile(m_engineName, std::ofstream::binary);
	outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

	std::cout << "Success, saved engine to " << m_engineName << std::endl;

	checkCudaErrorCode(cudaStreamDestroy(profile_stream));
	return true;
}

/**
 * @brief Load and prepare the network for inference
 * @return true if runtime engine was successfully loaded
 */
bool Engine::loadNetwork()
{
	// Read the serialized model from disk
	std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size))
		throw std::runtime_error("Unable to read engine file");

	m_runtime = std::unique_ptr < IRuntime > {createInferRuntime(m_logger)};
	if (!m_runtime)
		return false;

	// Set the device index
	auto ret = cudaSetDevice(m_options.deviceIndex);
	if (ret != 0)
	{
		int num_GPUs;
		cudaGetDeviceCount(&num_GPUs);
		auto err_msg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) +
					  ". Note, your device has " + std::to_string(num_GPUs) + " CUDA-capable GPU(s).";
		throw std::runtime_error(err_msg);
	}

	m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
	if (!m_engine)
		return false;

	m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
	if (!m_context)
		return false;

	// Storage for holding the input and output buffers
	// This will be passed to TensorRT for inference
	m_buffers.resize(m_engine->getNbBindings());

	// Create a cuda stream
	cudaStream_t cuda_stream;
	checkCudaErrorCode(cudaStreamCreate(&cuda_stream));

	// Allocate GPU memory for input and output buffers
	m_outputLengthsFloat.clear();
	for (int i = 0; i < m_engine->getNbBindings(); ++i)
	{
		if (m_engine->bindingIsInput(i))
		{
			auto binding_dims = m_engine->getBindingDimensions(i);

			// Allocate memory for the input
			// Allocate enough to fit the max batch size (we could end up using less later)
			checkCudaErrorCode(cudaMallocAsync(&m_buffers[i],
											   m_options.maxBatchSize * binding_dims.d[1] * binding_dims.d[2] *
													   binding_dims.d[3] * sizeof(float), cuda_stream));

			// Store the input dims for later use
			m_inputDims.emplace_back(binding_dims.d[1], binding_dims.d[2], binding_dims.d[3]);
		} else
		{
			// The binding is an output
			uint32_t outputLenFloat = 1;
			auto output_dims = m_engine->getBindingDimensions(i);
			m_outputDims.push_back(output_dims);

			// We ignore j = 0 because that is the batch size, and we will take that into account when sizing the buffer
			for (int j = 1; j < output_dims.nbDims; ++j)
				outputLenFloat *= output_dims.d[j];

			m_outputLengthsFloat.push_back(outputLenFloat);
			// Now size the output buffer appropriately,
			// taking into account the max possible batch size (although we could actually end up using less memory)
			checkCudaErrorCode(
					cudaMallocAsync(
							&m_buffers[i],
							outputLenFloat * m_options.maxBatchSize * sizeof(float),
							cuda_stream)
							);
		}
	}

	// Synchronize and destroy the cuda stream
	checkCudaErrorCode(cudaStreamSynchronize(cuda_stream));
	checkCudaErrorCode(cudaStreamDestroy(cuda_stream));
	return true;
}

/**
 * @brief runtime inference
 * @param inputs [input][batch][image]
 * @return bool [batch][output][feature_vector]
 */
bool Engine::runInference(
		const std::vector <std::vector<cv::cuda::GpuMat>> &inputs,
		std::vector <std::vector<std::vector< float>>>& feature_vectors,
		const std::array<float, 3> &subVals,
		const std::array<float, 3> &divVals,
		bool normalize)
{
	// First we do some error checking
	if (inputs.empty() || inputs[0].empty())
	{
		std::cout << "===== Error =====\nProvided input vector is empty!" << std::endl;
		return false;
	}
	const auto num_inputs = m_inputDims.size();
	if (inputs.size() != num_inputs)
	{
		std::cout << "===== Error =====\nIncorrect number of inputs provided!" << std::endl;
		return false;
	}

	// Ensure the dynamic batch size param was not set
	if (!m_options.doesSupportDynamicBatchSize)
	{
		if (inputs[0].size() > 1)
			{
				std::cout << "===== Error =====\n"
							 "Model does not support running batch inference!\n"
							 "Please only provide a single input" << std::endl;
				return false;
			}
	}

	const auto batch_size = static_cast<int32_t>(inputs[0].size());
	// Make sure the same batch size was provided for all inputs
	for (size_t i = 1; i<inputs.size();	++i)
	{
		if (inputs[i].size() != static_cast<size_t>(batch_size))
		{
			std::cout << "===== Error =====\nThe batch size needs to be constant for all inputs!" << std::endl;
			return false;
		}
	}

	// Create the cuda stream that will be used for inference
	cudaStream_t inference_cuda_stream;
	checkCudaErrorCode(cudaStreamCreate(&inference_cuda_stream));
	// Preprocess all the inputs
	for (size_t i = 0; i<num_inputs; ++i)
	{
		const auto &batch_input = inputs[i];
		const auto &dims = m_inputDims[i];

		auto &input = batch_input[0];
		if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2])
		{
			std::cout << "===== Error =====\nInput does not have correct size!" << std::endl;
			std::cout << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", " << dims.d[2] << ")" << std::endl;
			std::cout << "Got: (" << input.channels() << ", " << input.rows << ", " << input.cols << ")" << std::endl;
			std::cout << "Ensure you resize your input image to the correct size" << std::endl;
			return false;
		}
		nvinfer1::Dims4 inputDims = {batch_size, dims.d[0], dims.d[1], dims.d[2]};
		// Define the batch size
		m_context->setBindingDimensions(i, inputDims);
		// Copy over the input data and perform the preprocessing
		cv::cuda::GpuMat gpu_dst(1, batch_input[0].rows * batch_input[0].cols * batch_size, CV_8UC3);

		// Convert NHWC (OpenCV) to NCHW (TensorRT).
		size_t width = batch_input[0].cols * batch_input[0].rows;
		for (size_t img = 0; img<batch_input.size(); img++)
		{
			std::cout << "Split channels: [img=" << img << "] ["  << 0 << "," << width << "," << width * 2 << "]" << std::endl;
			std::vector <cv::cuda::GpuMat> input_channels
			{
				cv::cuda::GpuMat(batch_input[0].rows, batch_input[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
				cv::cuda::GpuMat(batch_input[0].rows, batch_input[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
				cv::cuda::GpuMat(batch_input[0].rows, batch_input[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))
			};
			// HWC -> CHW
			cv::cuda::split(batch_input[img], input_channels);
		}

		cv::cuda::GpuMat mfloat;
		// normalize [0.f, 1.f] or not [0.f, 255.f]
		if (normalize)
			gpu_dst.convertTo(mfloat, CV_32FC3,	1.f / 255.f);
		else
			gpu_dst.convertTo(mfloat, CV_32FC3);

		// Apply normalizations
		cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
		cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

		auto *data_ptr = mfloat.ptr<void>();
		checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i],
										   data_ptr,
										   mfloat.cols * mfloat.rows * mfloat.channels() * sizeof(float),
										   cudaMemcpyDeviceToDevice, inference_cuda_stream));
	}

	// Ensure all dynamic bindings have been defined.
	if (!m_context->allInputDimensionsSpecified())
		throw std::runtime_error("Error, not all required dimensions specified.");

	// Run inference.
	bool status = m_context->enqueueV2(m_buffers.data(), inference_cuda_stream, nullptr);
	if (!status)
		return false;

	// Copy the outputs back to CPU
	feature_vectors.clear();

	for (int batch = 0; batch<batch_size; ++batch)
	{
		// Batch
		std::vector <std::vector<float>> batchOutputs{};
		for (int32_t outputBinding = num_inputs; outputBinding<m_engine->getNbBindings(); ++outputBinding)
		{
			// We start at index m_inputDims.size() to account for the inputs in our m_buffers
			std::vector<float> output;
			auto outputLenFloat = m_outputLengthsFloat[outputBinding - num_inputs];
			output.resize(outputLenFloat);
			// Copy the output
			checkCudaErrorCode(
					cudaMemcpyAsync(
							output.data(),
							static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(float) * outputLenFloat),
							outputLenFloat * sizeof(float),
							cudaMemcpyDeviceToHost,
							inference_cuda_stream)
							);
			batchOutputs.emplace_back(std::move(output));
		}
		feature_vectors.emplace_back(std::move(batchOutputs));
	}

	// Synchronize the cuda stream
	checkCudaErrorCode(cudaStreamSynchronize(inference_cuda_stream));
	checkCudaErrorCode(cudaStreamDestroy(inference_cuda_stream));
	return true;
}

/**
 * @brief resize the cv::Mat and keep the correct aspect ration when resize for NN dimensions
 * @param input GPU cuda Mat
 * @param height desired height
 * @param width  desired width
 * @param bgcolor background color
 * @return cv::cuda::GpuMat fit to desired dimension with padding to keep aspect ratio
 */
cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(
		const cv::cuda::GpuMat &input,
		size_t height,
		size_t width,
		const cv::Scalar &bgcolor)
{
	// get ratio
	float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
	// adjust unpad dimensions
	int unpad_w = r * input.cols;
	int unpad_h = r * input.rows;

	cv::cuda::GpuMat gpu_resize_mat(unpad_h, unpad_w, CV_8UC3);
	cv::cuda::resize(input, gpu_resize_mat, gpu_resize_mat.size());
	cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
	gpu_resize_mat.copyTo(out(cv::Rect(0, 0, gpu_resize_mat.cols, gpu_resize_mat.rows)));
	return out;
}


/**
 * @brief create string name of the serialize engine file with its engine options in the name
 * @param options the options used to create the serialized engine file
 * @param onnx_model_path path to the onnx file that was used to create the serialized engine file
 * @return std::string of the name where the engine file is saved/located
 */
std::string Engine::serializeEngineOptions(const Options &options, const std::string &onnx_model_path)
{
	// extract the name of the file from its '/' path
	const auto position = onnx_model_path.find_last_of('/') + 1;
	std::string engine_name = onnx_model_path.substr(position,onnx_model_path.find_last_of('.') - position) + ".engine";

	// Add the GPU device name to the file to ensure that the model is only used on devices with the exact same GPU
	std::vector <std::string> device_names;
	getDeviceNames(device_names);

	if (static_cast<size_t>(options.deviceIndex) >= device_names.size())
		throw std::runtime_error("Error, provided device index is out of range!");

	auto deviceName = device_names[options.deviceIndex];

	// Remove spaces from the device name
	deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

	engine_name += "." + deviceName;

	// Serialize the specified options into the filename
	if (options.precision == Precision::FP16)
		engine_name += ".fp16";
	else
		engine_name += ".fp32";

	engine_name += "." + std::to_string(options.maxBatchSize);
	engine_name += "." + std::to_string(options.optBatchSize);
	engine_name += "." + std::to_string(options.maxWorkspaceSize);

	std::string file_path = splitStringBeforeLastSlash(onnx_model_path);
	return file_path + std::string("/") + engine_name;
}

/**
 * @brief returns the number of compute capable devices (GPUS) and their properties
 * @param device_names vector to hold properties
 */
void Engine::getDeviceNames(std::vector <std::string> &device_names)
{
	int num_GPUs;
	cudaGetDeviceCount(&num_GPUs);

	for (int device = 0; device < num_GPUs; device++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device);
		device_names.push_back(std::string(prop.name));
	}
}

/**
 * @brief handles when cuda errors occur
 * @param code error code to check
 */
void Engine::checkCudaErrorCode(cudaError_t code)
{
	if (code != 0)
	{
		std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
							 "), with message: " + cudaGetErrorString(code);
		std::cout << errMsg << std::endl;
		throw std::runtime_error(errMsg);
	}
}