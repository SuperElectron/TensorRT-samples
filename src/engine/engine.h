#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"

/**
 * @enum Precision
 * @brief Precision used for GPU inference
 * @var FP32 floating point 32
 * @var FP16 floating point 16
 */
enum class Precision {
    FP32,
    FP16
};

/**
 * @struct Options
 * @brief Options for the network
 * @var doesSupportDynamicBatchSize false no dynamic batch sizing is supported
 * @var precision Precision to use for GPU inference. 16 bit is faster but may reduce accuracy.
 * @var optBatchSize The batch size which should be optimized for.
 * @var maxBatchSize Maximum allowable batch size
 * @var maxWorkspaceSize Max allowable GPU memory to be used for model conversion, in bytes
 * @var deviceIndex GPU device index
 */
struct Options {
    bool doesSupportDynamicBatchSize = true;
    Precision precision = Precision::FP16;
    int32_t optBatchSize = 1;
    int32_t maxBatchSize = 16;
    size_t maxWorkspaceSize = 4000000000;
    int deviceIndex = 0;
};

/**
 * @class Logger
 * @brief Class to extend TensorRT logger
 */
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

/**
 * @class Engine
 * @brief InferenceEngine class to convert and run models
 */
class Engine {
public:
    Engine(const Options& options);
    ~Engine();

    bool build(std::string onnx_model_path);
    bool loadNetwork();

    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>>& inputs,
					  std::vector<std::vector<std::vector<float>>>& feature_vectors,
					  const std::array<float, 3>& subVals = {0.f, 0.f, 0.f},
                      const std::array<float, 3>& divVals = {1.f, 1.f, 1.f},
					  bool normalize = true);

    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(
			const cv::cuda::GpuMat& input,
			size_t height,
			size_t width,
			const cv::Scalar& bgcolor = cv::Scalar(0, 0, 0));

    const std::vector<nvinfer1::Dims3>& getInputDims() const { return m_inputDims; };
    const std::vector<nvinfer1::Dims>& getOutputDims() const { return m_outputDims ;};
private:
    std::string serializeEngineOptions(const Options& options, const std::string& onnx_model_path);

    void getDeviceNames(std::vector<std::string>& device_names);

    bool doesFileExist(const std::string& filepath);

    // Holds pointers to the input and output GPU buffers
    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;

    // Must keep IRuntime around for inference, see: https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Options m_options;
    Logger m_logger;
    std::string m_engineName;

    inline void checkCudaErrorCode(cudaError_t code);
};
