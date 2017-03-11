#include "stdafx.h"
#include <random>
#include <iostream>
#include <fstream>
// ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix_proxy.hpp"

// Boost headers

#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"
#include <boost/algorithm/string.hpp>

using namespace std;

typedef float											ScalarType;
typedef boost::numeric::ublas::matrix<ScalarType>       MatrixType;
typedef viennacl::matrix<ScalarType, viennacl::row_major>    VCLMatrixType;

struct frequencyData {
	int* frequency;
	double* magnitude;
	int segmentStart;
	int segmentEnd;
};

struct layer {
	layer() {

	}
	layer(boost::numeric::ublas::vector<float> i, MatrixType w, int n) {
		inputs = i;
		weights = w;
		numberNeurons = n;
	}
	boost::numeric::ublas::vector<float> inputs;
	MatrixType weights;
	int numberNeurons;
	boost::numeric::ublas::vector<ScalarType> outputs;
	boost::numeric::ublas::vector<ScalarType> deltas;
};

struct networkLayout {
	std::vector<layer> layers;
	boost::numeric::ublas::vector<ScalarType> inputs;
	boost::numeric::ublas::vector<ScalarType> finalOutputs;
	
};





static networkLayout initialize(networkLayout network, std::vector<int> numberNeurons, int initialInputSize) {


	std::vector<layer> layers = std::vector<layer>(numberNeurons.size());
	std::random_device rd;
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0, 0.1);
	for (int i = 0; i < numberNeurons.size(); i++) {
		layers[i].numberNeurons = numberNeurons[i];
		
		if(i!=0) {

			layers[i].weights = MatrixType(numberNeurons[i-1], numberNeurons[i]);

			for (int j = 0; j < numberNeurons[i-1]; j++) {
				for (int k = 0; k < numberNeurons[i]; k++) {
					layers[i].weights(j, k) = dis(gen);
				}
			}
		}
	}
	
	network.layers = layers;

	return network;
}

static networkLayout forward_step(networkLayout network, double * frequency) {


	for (int i = 0; i < network.layers.size() ; i++) {

		
		boost::numeric::ublas::vector<float> temp = boost::numeric::ublas::vector<float>(network.layers[i].numberNeurons);
		if (i == 0) {
			float max = 0.0;
			float min = 10000000000;
			float sum = 0.0;
			
			for (int k = 0; k < network.layers[i].numberNeurons; k++) {
				if (frequency[k] > max) {
					max = frequency[k];
				}
				if (frequency[k] < min) {
					min = frequency[k];
				}
				sum += frequency[k];
			}
			for (int k = 0; k < network.layers[i].numberNeurons; k++) {
				if (max > 0) {
					temp(k) = (float)(abs(frequency[k] - (sum/ network.layers[i].numberNeurons))/ (max- min));
				}
				else {
					temp(k) = 0.0;
				}
			}
			network.layers[i + 1].inputs = temp;
			network.layers[i].inputs = temp;
			network.layers[i].outputs = temp;
		}
		else {
			VCLMatrixType tempWeights(network.layers[i].weights.size1(), network.layers[i].weights.size2());
			viennacl::vector<float> tempInputs(network.layers[i].weights.size1());
			copy(network.layers[i].weights, tempWeights);
			int test = network.layers[i].weights.size1();

			boost::numeric::ublas::vector<float> temp = boost::numeric::ublas::vector<float>(test);
			for (int k = 0; k < network.layers[i].weights.size1(); k++) {
				temp(k) = network.layers[i].inputs[k];
			}
			copy(temp.begin(), temp.end(), tempInputs.begin());
			viennacl::vector<ScalarType> vcl_result = viennacl::vector<ScalarType>(tempWeights.size2());

			vcl_result = viennacl::linalg::prod(trans(tempWeights), tempInputs);
			network.layers[i].outputs = boost::numeric::ublas::vector<float>(tempWeights.size2());
			boost::numeric::ublas::vector<float> outputs = boost::numeric::ublas::vector<float>(tempWeights.size2());
			copy(vcl_result, outputs);

			for (int j = 0; j < outputs.size(); j++) {
				outputs(j) = 1 / (1 + std::exp(-1 * outputs(j)));
			}
			network.layers[i].outputs = outputs;
			if (i + 1 < network.layers.size()) {
				network.layers[i + 1].inputs = outputs;
			}
		}
		
	}
	//network.finalOutputs = network.layers[network.layers.size() - 1].outputs;

	return network;
}

static networkLayout backprop(networkLayout network, boost::numeric::ublas::vector<float> correctOutputs) {
	boost::numeric::ublas::vector<float> transfer;
	for (int i = network.layers.size() -1; i >= 0; i--) {
		boost::numeric::ublas::vector<float> errors = boost::numeric::ublas::vector<float>(network.layers[i].numberNeurons);
		if (i == network.layers.size() -1) {				
			for (int j = 0; j < correctOutputs.size(); j++) {
				errors(j) =(correctOutputs(j) - network.layers[i].outputs[j]);
			}				
		}
		else {
			

			for (int j = 0; j < network.layers[i].numberNeurons; j++) {
				float error = 0.0;
				for (int k = 0; k < network.layers[i+1].numberNeurons; k++) {
					error += (column(network.layers[i+1].weights, k)[j] * network.layers[i+1].deltas(k));
				}
				errors(j) = error;
			}
			
		}
		transfer = boost::numeric::ublas::vector<float>(network.layers[i].numberNeurons);
		for (int j = 0; j < network.layers[i].numberNeurons; j++) {
			transfer(j) = 1.0;
		}

		viennacl::vector<float> outputsV;
		copy(network.layers[i].outputs, outputsV);
		viennacl::vector<float> errorsV;
		copy(errors, errorsV);
		viennacl::vector<float> transferVector;
		copy(transfer, transferVector);
		viennacl::vector<float> deltasV;

		deltasV = viennacl::linalg::element_prod(errorsV, viennacl::linalg::element_prod(outputsV, transferVector - outputsV));
		network.layers[i].deltas = boost::numeric::ublas::vector<float>(deltasV.size());
		copy(deltasV, network.layers[i].deltas);

	}
	return network;
}

static networkLayout update_weights(networkLayout network, double * frequency) {
	
	for (int i = 0; i < network.layers.size(); i++) {
		int test = network.layers[i].weights.size1();
		boost::numeric::ublas::vector<float> inputs = boost::numeric::ublas::vector<float>(test);
		boost::numeric::ublas::vector<float> currentInputs;	
		
		if(i!=0)  {
			currentInputs = network.layers[i].inputs;
			MatrixType updatedWeight = MatrixType(network.layers[i].inputs.size(), network.layers[i].numberNeurons);
			for (int k = 0; k < network.layers[i].numberNeurons; k++) {				
				for (int j = 0; j < network.layers[i].inputs.size(); j++) {			
					
					updatedWeight(j,k) = 0.5 * network.layers[i].deltas[k] * currentInputs[j];					
				}
				
				updatedWeight(currentInputs.size() - 1, k) = 0.5 * network.layers[i].deltas[k];
			}
			network.layers[i].weights = network.layers[i].weights + updatedWeight;
		}
		
		

	}
	return network;
}

static networkLayout train_network(networkLayout network, frequencyData * frequencyData, int sampleChunk, int start, int sampleSize, boost::numeric::ublas::vector<ScalarType> correctOutputs) {
	
	for (int i = start; i < sampleSize; i++) {		
		network = forward_step(network, frequencyData[i].magnitude);
		
		network = backprop(network, correctOutputs);
		network = update_weights(network, frequencyData[i].magnitude);
		

	}

	
	return network;
}

static int saveTrained(networkLayout network) {
	ofstream myfile;
	myfile.open("output.csv");

	for (int i = 0; i < network.layers.size(); i++) {
		myfile << network.layers[i].numberNeurons << ",";		
	}
	myfile << std::endl;
	for (int i = 0; i < network.layers.size(); i++) {				
		
		for (int j = 0; j < network.layers[i].weights.size1(); j++) {
			for (int k = 0; k < network.layers[i].weights.size2(); k++) {
				myfile << network.layers[i].weights(j, k);
				myfile << ",";
			}
			myfile << std::endl;
		}
		
	}
	myfile.close();
	return 0;
}



static networkLayout loadTrained(std::string filePath) {
	networkLayout network = networkLayout();
	std::vector<layer> layers;
	
	ifstream myfile;
	myfile.open("output.csv");
	
	std::string firstline;
	getline(myfile, firstline);
	std::vector<std::string> neurons;
	boost::trim_if(firstline, boost::is_any_of(","));
	boost::split(neurons, firstline, boost::is_any_of(","));
	vector<int> numNeurons = vector<int>(neurons.size());
	for (int i = 0; i < neurons.size(); i++) {
		numNeurons[i] =  stoi(neurons[i]);
	}
	layers = vector<layer>(numNeurons.size());
	for (int i = 0; i < neurons.size(); i++) {
		layers[i].numberNeurons = numNeurons[i];
	}
	string buffer;
	vector<string> bufferV;
	std::string::size_type sz;
	for (int j = 1; j < neurons.size(); j++) {
		layers[j].weights = MatrixType(numNeurons[j - 1], numNeurons[j]);
		for (int k = 0; k < numNeurons[j-1]; k++) {
			getline(myfile, buffer);
			boost::trim_if(buffer, boost::is_any_of(","));
			boost::split(bufferV, buffer, boost::is_any_of(","));
			for (int l = 0; l < bufferV.size(); l++) {
				layers[j].weights(k, l) = stof(bufferV[l],&sz);
			}
		}
		
	}
	network.layers = layers;
	return network;
}

