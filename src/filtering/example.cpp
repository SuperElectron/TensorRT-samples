#include <iostream>
#include <vector>
#include <cmath>

// Sample distance kernel (5x5 grid)
std::vector<std::vector<double>> distanceKernel = {
		{1.5, 2.3, 1.8, 2.7, 1.2},
		{3.1, 1.2, 2.5, 1.9, 2.2},
		{2.0, 1.6, 3.4, 2.9, 1.7},
		{2.8, 1.9, 2.3, 1.5, 2.1},
		{1.3, 2.7, 1.1, 2.0, 3.0}
};

// Function to perform local mean smoothing
double localMeanSmoothing(const std::vector<double>& input, int windowSize) {
	double smoothedValues;

	for (int i = 0; i < input.size(); ++i) {
		int start = std::max(0, i - windowSize);
		int end = std::min(static_cast<int>(input.size()) - 1, i + windowSize);

		double sum = 0.0;
		for (int j = start; j <= end; ++j) {
			sum += input[j];
		}

		smoothedValues += sum / (end - start + 1);
	}

	return smoothedValues/input.size();
}

// Function to perform box (rectangular) smoothing
double boxSmoothing(const std::vector<double>& input, int windowSize) {
	double smoothedValues;

	for (int i = 0; i < input.size(); ++i) {
		int start = std::max(0, i - windowSize);
		int end = std::min(static_cast<int>(input.size()) - 1, i + windowSize);

		double sum = 0.0;
		for (int j = start; j <= end; ++j) {
			sum += input[j];
		}
		smoothedValues += sum / (end - start + 1);
	}

	return smoothedValues/input.size();
}

// Function to perform Gaussian smoothing
double gaussianSmoothing(const std::vector<double>& input, double sigma) {
	double smoothedValues;

	int kernelSize = static_cast<int>(6 * sigma + 1); // Adjust kernel size based on sigma

	for (int i = 0; i < input.size(); ++i) {
		double sum = 0.0;
		double weightSum = 0.0;

		for (int j = -kernelSize; j <= kernelSize; ++j) {
			int idx = i + j;
			if (idx >= 0 && idx < input.size()) {
				double weight = std::exp(-0.5 * std::pow(static_cast<double>(j) / sigma, 2));
				sum += input[idx] * weight;
				weightSum += weight;
			}
		}

		smoothedValues += sum/weightSum;
	}

	return smoothedValues/input.size();
}

int main() {
	/**
	 * TO COMPILE THE PROGRAM
	 * g++ -o example example.cpp
	 * TO RUN THE PROGRAM
	 * ./example
	 */
	std::vector<double> distances = {2.9, 2.3, 3.4, 3.5, 3.3, 1.2, 2.5, 3.1, 3.3, 4.6, 5.9, 3.6, 3.9, 3.3, 3.8, 3.7, 3.2};
	double unfiltered_value = distances[(int) distances.size()/2];

	std::cout << "unfiltered_value: " << unfiltered_value << std::endl;

	// Local mean smoothing
	double localMeanSmoothed = localMeanSmoothing(distances, 1);
	std::cout << "Local Mean Smoothing: " << localMeanSmoothed << std::endl;;

	// Box (rectangular) smoothing
	double boxSmoothed = boxSmoothing(distances, 1);
	std::cout << "Box Smoothing: " << boxSmoothed << std::endl;

	// Gaussian smoothing
	double gaussianSmoothed = gaussianSmoothing(distances, 1.0);
	std::cout << "Gaussian Smoothing: " << gaussianSmoothed << std::endl;

	return 0;
}
