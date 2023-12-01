#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;

class MyLogger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg)

	noexcept override;
};

void MyLogger::log(Severity severity, const char *msg) noexcept
{
	// Only log Warnings or more important.
	if (severity <= Severity::kWARNING)
		std::cout << "[WARNING] " << msg << std::endl;
	else if (severity <= Severity::kINFO)
		std::cout << "[INFO] " << msg << std::endl;

}



std::string create_output_model_name(std::string onnxFilePath) {

	// Get the directory from the provided ONNX file path
	size_t lastSlash = onnxFilePath.find_last_of('/');
	size_t lastBackslash = onnxFilePath.find_last_of('\\');
	size_t lastIndex = std::max(lastSlash, lastBackslash);

	std::string directory = (lastIndex != std::string::npos) ? onnxFilePath.substr(0, lastIndex) : "";

	// Remove the ".onnx" extension from the ONNX file name
	size_t extensionPos = onnxFilePath.rfind(".onnx");
	std::string modelName = onnxFilePath.substr(lastIndex + 1, extensionPos - lastIndex - 1);

	// Print the directory and model name
	std::cout << "Model Name: " << modelName << std::endl;
	std::string outputModelName = directory + (std::string) "/" + modelName + (std::string) ".trt";
	return outputModelName;
}



int main(int argc, char** argv) {

	/**
	 * @brief broken for TensorRT 8.2
	 * 		$ trtexec --onnx=model.onnx --saveEngine=model.trt --best --buildOnly --workspace=4096 --verbose
	 * @copybrief or with no logging
	 * 		$ trtexec --onnx=model.onnx --saveEngine=model.trt --best --buildOnly --workspace=4096
	 *
	 */
	std::cout << "** Convert ONNX TO TensorRT **\n";
	if (argc != 2) {
		std::cerr << "Usage    \t\t$ " << argv[0] << " <path_to_onnx_model>" << std::endl;
		std::cout << "Try this \t\t$ " << argv[0] << " /src/models/yourModel.onnx" << std::endl;
		return 1;
	}

	MyLogger gLogger;
	const std::string onnxFilePath = argv[1];
	std::string trt_model = create_output_model_name(argv[1]);

	std::cout << "[main] Create builder" << std::endl;
	IBuilder* builder = createInferBuilder(gLogger);
	builder->setMaxBatchSize(1);
	auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	std::cout << "[main] Create network" << std::endl;
	INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
	std::cout << "[main] Create parser" << std::endl;
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(ILogger::Severity::kINFO));
	std::cout << "[main] Create builderConfig" << std::endl;
	IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(4000000000);
	config->setFlag(BuilderFlag::kFP16);

	std::cout << "[main] Register a single optimization profile" << std::endl;
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
		int batch_size = 1;
		optimized_profile->setDimensions(input_name, OptProfileSelector::kMIN, Dims4(batch_size, input_channel, input_height, input_width));
		optimized_profile->setDimensions(input_name, OptProfileSelector::kOPT, Dims4(batch_size, input_channel, input_height, input_width));
		optimized_profile->setDimensions(input_name, OptProfileSelector::kMAX, Dims4(batch_size, input_channel, input_height, input_width));
	}
	config->addOptimizationProfile(optimized_profile);

	std::cout << "[main] Build engine with config" << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "[main] Create serialized engine" << std::endl;
	IHostMemory* serializedEngine = engine->serialize();
	std::ofstream engineFile(trt_model.c_str(), std::ios::binary);
	std::cout << "[main] Saving engine file (" << trt_model << ")" << std::endl;
	engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
	engineFile.close();

	serializedEngine->destroy();
	engine->destroy();
	parser->destroy();
	network->destroy();
	builder->destroy();

	return 0;
}
