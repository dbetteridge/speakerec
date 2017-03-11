// SpeakerRec.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ReadWav.cpp"
#include "NeuralNet.cpp"
#include "fftw3.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "viennacl.hpp"

// sample rate for a 20ms chunk
int sampleChunk = 16000/50;
int sampleRateDivisor;


int time_to_frequency(frequencyData * frequency, sound audio) {

	sampleRateDivisor = ceil((float)audio.sampleRate / 16000);

	double *in;
	in = (double *)fftw_malloc(sizeof(double) * sampleChunk);
	short * sampledItems;

	sampledItems = (short *)fftw_malloc(sizeof(short) * audio.totalItems / (audio.channels * sampleRateDivisor));

	for (int k = 0; k < audio.totalItems / (audio.channels * sampleRateDivisor); k += audio.channels *sampleRateDivisor) {
		if (k == 0) {
			sampledItems[k] = audio.audioItems[k];
		}
		sampledItems[(k - (audio.channels * sampleRateDivisor)) + 1] = audio.audioItems[k];
	}

	for (int j = 0; j < audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor)); j++) {

		for (int i = 0; i < sampleChunk; i++) {
			float index = (float)j * (float)(sampleChunk)+i;

			if (index < audio.totalItems / (audio.channels * sampleRateDivisor)) {
				in[i] = sampledItems[(int)index];
			}
		}

		fftw_complex *out;
		out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *sampleChunk);
		int dimension = ((int)sampleChunk);
		fftw_plan p;
		p = fftw_plan_dft_r2c_1d(dimension, in, out, FFTW_ESTIMATE);
		if (p != NULL) {
			fftw_execute(p);

			frequency[j].frequency = (int *)fftw_malloc(sizeof(int) *sampleChunk / 2);
			frequency[j].magnitude = (double *)fftw_malloc(sizeof(fftw_complex) * sampleChunk / 2);
			for (int i = 0; i < (float)sampleChunk / 2; i++) {
				frequency[j].frequency[i] = i * 50;
				frequency[j].magnitude[i] = abs(out[i][0]);
				int test = 0;
			}
			frequency[j].segmentStart = j * sampleChunk;
			frequency[j].segmentEnd = (j + 1) * sampleChunk;
		}
		fftw_free(out);
		fftw_free(p);
	}

	return 0;
}


int main()
{

	frequencyData * frequency;
	sound audio = Read("C:/Mega/Programming/SpeakerRec/Samples/president-is-moron.wav");
	int sampleRateDivisor = ceil((float)audio.sampleRate / 16000);
	frequency = (frequencyData *)fftw_malloc(sizeof(frequencyData) * audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor)));

	time_to_frequency(frequency, audio);

	frequencyData * frequency2;
	sound audio2 = Read("C:/Mega/Programming/SpeakerRec/Samples/172213__acclivity__merrychristmasdarling.wav");
	int sampleRateDivisor2 = ceil((float)audio2.sampleRate / 16000);
	frequency2 = (frequencyData *)fftw_malloc(sizeof(frequencyData) * audio2.totalItems / (sampleChunk * (audio2.channels * sampleRateDivisor2)));

	time_to_frequency(frequency2, audio2);

	frequencyData * frequency3;
	sound audio3 = Read("C:/Mega/Programming/SpeakerRec/Samples/LDC1996S36.wav");
	int sampleRateDivisor3 = ceil((float)audio3.sampleRate / 16000);
	frequency3 = (frequencyData *)fftw_malloc(sizeof(frequencyData) * audio3.totalItems / (sampleChunk * (audio3.channels * sampleRateDivisor3)));

	time_to_frequency(frequency3, audio3);

	frequencyData * frequency4;
	sound audio4 = Read("C:/Mega/Programming/SpeakerRec/Samples/oh-yeah-everything-is-fine.wav");
	int sampleRateDivisor4 = ceil((float)audio4.sampleRate / 16000);
	frequency4 = (frequencyData *)fftw_malloc(sizeof(frequencyData) * audio4.totalItems / (sampleChunk * (audio4.channels * sampleRateDivisor4)));

	time_to_frequency(frequency4, audio4);

	networkLayout network = networkLayout();
	/**
	std::vector<int> numNeurons = std::vector<int>(3);

	numNeurons[0] = sampleChunk / 2;
	numNeurons[1] = 50;
	numNeurons[2] = 3;

	
	network = initialize(network, numNeurons, sampleChunk / 2);

	boost::numeric::ublas::vector<ScalarType> correctOutputs1 = boost::numeric::ublas::vector<ScalarType>(3);
	correctOutputs1(0) = 1.0;
	correctOutputs1(1) = 0.0;
	correctOutputs1(2) = 0.0;
	

	boost::numeric::ublas::vector<ScalarType> correctOutputs2 = boost::numeric::ublas::vector<ScalarType>(3);
	correctOutputs2(0) = 0.0;
	correctOutputs2(1) = 1.0;
	correctOutputs2(2) = 0.0;
	

	boost::numeric::ublas::vector<ScalarType> correctOutputs3 = boost::numeric::ublas::vector<ScalarType>(3);
	correctOutputs3(0) = 0.0;
	correctOutputs3(1) = 0.0;
	correctOutputs3(2) = 1.0;
	

	for (int i = 0; i < 100; i++) {
		
		sampleRateDivisor = ceil((float)audio.sampleRate / 16000);
		network = train_network(network, frequency, sampleChunk, 0,(audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor))/3), correctOutputs1);

		
		
		sampleRateDivisor = ceil((float)audio2.sampleRate / 16000);
		network = train_network(network, frequency2, sampleChunk, 0,audio2.totalItems / (sampleChunk * (audio2.channels * sampleRateDivisor))/3, correctOutputs2);

		
		
		sampleRateDivisor = ceil((float)audio3.sampleRate / 16000);
		network = train_network(network, frequency3, sampleChunk,0, audio3.totalItems / (sampleChunk * (audio3.channels * sampleRateDivisor))/3, correctOutputs3);

		sampleRateDivisor = ceil((float)audio.sampleRate / 16000);
		network = train_network(network, frequency, sampleChunk, audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor)) / 3, 2*(audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor))) / 3, correctOutputs1);



		sampleRateDivisor = ceil((float)audio2.sampleRate / 16000);
		network = train_network(network, frequency2, sampleChunk, audio2.totalItems / (sampleChunk * (audio2.channels * sampleRateDivisor)) / 3, 2*(audio2.totalItems / (sampleChunk * (audio2.channels * sampleRateDivisor))) / 3, correctOutputs2);



		sampleRateDivisor = ceil((float)audio3.sampleRate / 16000);
		network = train_network(network, frequency3, sampleChunk, audio3.totalItems / (sampleChunk * (audio3.channels * sampleRateDivisor)) / 3, 2*(audio3.totalItems / (sampleChunk * (audio3.channels * sampleRateDivisor))) / 3, correctOutputs3);

		sampleRateDivisor = ceil((float)audio.sampleRate / 16000);
		network = train_network(network, frequency, sampleChunk, 2*(audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor))) / 3, (audio.totalItems / (sampleChunk * (audio.channels * sampleRateDivisor))), correctOutputs1);



		sampleRateDivisor = ceil((float)audio2.sampleRate / 16000);
		network = train_network(network, frequency2, sampleChunk, 2*(audio2.totalItems / (sampleChunk * (audio2.channels * sampleRateDivisor))) / 3, (audio2.totalItems / (sampleChunk * (audio2.channels * sampleRateDivisor))), correctOutputs2);



		sampleRateDivisor = ceil((float)audio3.sampleRate / 16000);
		network = train_network(network, frequency3, sampleChunk, 2*(audio3.totalItems / (sampleChunk * (audio3.channels * sampleRateDivisor))) / 3, (audio3.totalItems / (sampleChunk * (audio3.channels * sampleRateDivisor))), correctOutputs3);

				
	}
	
	saveTrained(network);
	**/

	network = loadTrained("");
	network = forward_step(network, frequency[10].magnitude);
	float result = network.layers[2].outputs(0);
	float result2 = network.layers[2].outputs(1);
	float result3 = network.layers[2].outputs(2);
	int test = 0;

	network = forward_step(network, frequency2[10].magnitude);
	result = network.layers[2].outputs(0);
	result2 = network.layers[2].outputs(1);
	result3 = network.layers[2].outputs(2);
	test = 0;

	network = forward_step(network, frequency4[10].magnitude);
	result = network.layers[2].outputs(0);
	result2 = network.layers[2].outputs(1);
	result3 = network.layers[2].outputs(2);
	test = 0;
	/**
	
	**/
}