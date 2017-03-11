#include "stdafx.h"
#include <stdlib.h> 
#include "sndfile.hh"


struct sound {
	int frames;
	int sampleRate;
	int channels;
	int totalItems;
	short* audioItems;
};

static sound Read(const char * fname) {
	

	SndfileHandle file;

	file = SndfileHandle(fname);
	const int num_items = file.frames() * file.channels();
	short *buffer;
	buffer = (short *) malloc(sizeof(short) * num_items);
	file.read(buffer, num_items);
	sound soundData;
	
	soundData.sampleRate = file.samplerate();
	soundData.channels = file.channels();
	soundData.frames = file.frames();
	soundData.audioItems = buffer;
	soundData.totalItems = num_items;

	puts("");
	return soundData;
}
