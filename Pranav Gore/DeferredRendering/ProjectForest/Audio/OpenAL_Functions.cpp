#include "OpenAL_Header.h"

struct RIFF_Header
{
	int8_t chunkID[4];
	int32_t chunkSize;
	int8_t format[4];
};

struct WAVE_Format
{
	int8_t subChunkID[4];
	int32_t subChunkSize;
	int16_t audioFormat;
	int16_t numChannels;
	int32_t sampleRate;
	int32_t byteRate;
	int16_t blockAlign;
	int16_t bitsPerSample;
};

struct WAVE_Data
{
	int8_t subChunkID[4];
	int32_t subChunk2Size;
};

ALCdevice *ALDevice = NULL;
ALCcontext *ALContext = NULL;

ALuint SrcWalking;
ALuint SrcRunning;
ALuint SrcBackground;
ALuint SrcAmbience;
ALuint SrcObjectPushed;
ALuint SrcFinalWin;
ALuint SrcEndCredits;

ALuint BufWalking;
ALuint BufRunning;
ALuint BufBackground;
ALuint BufAmbience;
ALuint BufObjectPushed;
ALuint BufFinalWin;
ALuint BufEndCredits;

FILE *gpALFile = NULL;

BOOL InitializeAL(void)
{

	if (fopen_s(&gpALFile, "OpenAL_LogFile.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("OpenAL Log File can not be created"), TEXT("Error"), MB_OK);
		ExitProcess(EXIT_FAILURE);
	}
	else
		fprintf(gpALFile, "OpenAL Log file created successfully\n");

	ALDevice = alcOpenDevice(NULL);
	if (!ALDevice)
	{
		fprintf(gpALFile, "Error while opening openAL device\n");
		UninitializeAL();
		return FALSE;
	}
	else
		fprintf(gpALFile, "OpenAL device opened successfully\n");

	ALContext = alcCreateContext(ALDevice, NULL);
	if (!alcMakeContextCurrent(ALContext))
	{
		fprintf(gpALFile, "error while making openAL context current\n");
		UninitializeAL();
		return FALSE;
	}

	return TRUE;
}

void InitializeAL_Listener()
{
	ALfloat listenerOri[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f };

	alListener3f(AL_POSITION, 0.0f, 0.0f, 0.0f);
	alListener3f(AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alListenerfv(AL_ORIENTATION, listenerOri);
}

void ChangeListenerAndSourcePos(glm::vec3 cameraPos)
{	
	//fprintf(gpALFile,"I m in ChangeListener() func...\n");
	alListener3f(AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
	alSource3f(SrcBackground, AL_POSITION,cameraPos.x, cameraPos.y, cameraPos.z);
	alSource3f(SrcAmbience, AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
	alSource3f(SrcWalking, AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
	alSource3f(SrcRunning, AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
	alSource3f(SrcEndCredits, AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
}

BOOL InitializeAL_BackgroundSource(void)
{
	if (!loadWAV("Resources/Audio/final.wav", &BufBackground))
		return FALSE;

	alGenSources((ALuint)1, &SrcBackground);

	alSourcei(SrcBackground, AL_BUFFER, BufBackground);
	alSourcef(SrcBackground, AL_PITCH, 1.0f);
	alSourcef(SrcBackground, AL_GAIN, 0.3f);
	alSource3f(SrcBackground, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcBackground, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcBackground, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcBackground, AL_LOOPING, AL_TRUE);
	return TRUE;
}

BOOL InitializeAL_WalkingSource(void)
{
	if (!loadWAV("Resources/Audio/Walking.wav", &BufWalking))
		return FALSE;

	alGenSources((ALuint)1, &SrcWalking);

	alSourcei(SrcWalking, AL_BUFFER, BufWalking);
	alSourcef(SrcWalking, AL_PITCH, 1.0f);
	alSourcef(SrcWalking, AL_GAIN, 0.8f);
	alSource3f(SrcWalking, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcWalking, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcWalking, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcWalking, AL_LOOPING, AL_TRUE);

	return TRUE;
}

BOOL InitializeAL_RunningSource(void)
{
	if (!loadWAV("Resources/Audio/Running.wav", &BufRunning))
		return FALSE;

	alGenSources((ALuint)1, &SrcRunning);

	alSourcei(SrcRunning, AL_BUFFER, BufRunning);
	alSourcef(SrcRunning, AL_PITCH, 1.0f);
	alSourcef(SrcRunning, AL_GAIN, 0.8f);
	alSource3f(SrcRunning, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcRunning, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcRunning, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcRunning, AL_LOOPING, AL_TRUE);

	return TRUE;
}

BOOL InitializeAL_AmbienceSource(void)
{
	if (!loadWAV("Resources/Audio/ambience_day.wav", &BufAmbience))
		return FALSE;

	alGenSources((ALuint)1, &SrcAmbience);

	alSourcei(SrcAmbience, AL_BUFFER, BufAmbience);
	alSourcef(SrcAmbience, AL_PITCH, 1.0f);
	alSourcef(SrcAmbience, AL_GAIN, 0.8f);
	alSource3f(SrcAmbience, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcAmbience, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcAmbience, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcAmbience, AL_LOOPING, AL_TRUE);

	return TRUE;
}

BOOL InitializeAL_ObjectPushed(void)
{
	if (!loadWAV("Resources/Audio/throw.wav", &BufObjectPushed))
		return FALSE;

	alGenSources((ALuint)1, &SrcObjectPushed);

	alSourcei(SrcObjectPushed, AL_BUFFER, BufObjectPushed);
	alSourcef(SrcObjectPushed, AL_PITCH, 1.0f);
	alSourcef(SrcObjectPushed, AL_GAIN, 1.0f);
	alSource3f(SrcObjectPushed, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcObjectPushed, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcObjectPushed, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcObjectPushed, AL_LOOPING, AL_FALSE);

	return TRUE;
}

BOOL InitializeAL_EndCredits(void)
{
	if (!loadWAV("Resources/Audio/endcredittrack.wav", &BufEndCredits))
		return FALSE;

	alGenSources((ALuint)1, &SrcEndCredits);

	alSourcei(SrcEndCredits, AL_BUFFER, BufEndCredits);
	alSourcef(SrcEndCredits, AL_PITCH, 1.0f);
	alSourcef(SrcEndCredits, AL_GAIN, 1.0f);
	alSource3f(SrcEndCredits, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcEndCredits, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcEndCredits, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcEndCredits, AL_LOOPING, AL_TRUE);

	return TRUE;
}

BOOL InitializeAL_FinalWinSource(void)
{
	if (!loadWAV("Resources/Audio/classic_tada.wav", &BufFinalWin))
		return FALSE;

	alGenSources((ALuint)1, &SrcFinalWin);

	alSourcei(SrcFinalWin, AL_BUFFER, BufFinalWin);
	alSourcef(SrcFinalWin, AL_PITCH, 1.0f);
	alSourcef(SrcFinalWin, AL_GAIN, 1.0f);
	alSource3f(SrcFinalWin, AL_POSITION, 0.0f, 0.0f, 0.0f);
	alSource3f(SrcFinalWin, AL_DIRECTION, 0.0f, 0.0f, -1.0f);
	alSource3f(SrcFinalWin, AL_VELOCITY, 0.0f, 0.0f, 0.0f);
	alSourcei(SrcFinalWin, AL_LOOPING, AL_FALSE);

	return TRUE;
}

void PlayAL_WalkingSource(void)
{
	alSourcePlay(SrcWalking);
}

void PauseAL_WalkingSource(void)
{
	alSourcePause(SrcWalking);
}

void PlayAL_RunningSource(void)
{
	alSourcePlay(SrcRunning);
}

void PauseAL_RunningSource(void)
{
	alSourcePause(SrcRunning);
}

void PlayAL_BackgroundSource(void)
{
	alSourcePlay(SrcBackground);
}

void StopAL_BackgroundSource(void)
{
	alSourceStop(SrcBackground);
}

void PlayAL_AmbienceSource(void)
{
	alSourcePlay(SrcAmbience);
}

void PlayAL_ObjectPushed(void)
{
	alSourcePlay(SrcObjectPushed);
}

void PlayAL_FinalWinSource(void)
{
	alSourcePlay(SrcFinalWin);
}

void PlayAL_EndCredits(void)
{
	alSourcePlay(SrcEndCredits);
}

BOOL loadWAV(string filename, ALuint *buff)
{
	//functions declarations
	void uninitialize(void);

	//local variables
	ifstream soundFile;
	struct WAVE_Format wave_format;
	struct RIFF_Header riff_header;
	struct WAVE_Data wave_data;
	unsigned char *data;

	ALsizei size;
	ALsizei frequency;
	ALenum format = 0;
	ALenum error;
	//ALuint BufWalking;

	//attempt to open the file and see if it's readable
	soundFile.open(filename.c_str(), ifstream::binary);
	if (!soundFile.is_open())
	{
		fprintf(gpALFile, "Error while opening the wav file\n");
		UninitializeAL();
		return FALSE;
	}

	//read the first chunk into the struct 
	soundFile.read((char *)&riff_header, sizeof(RIFF_Header));
	//check for RIFF and WAVE tag in memory
	if ((riff_header.chunkID[0] != 'R' || riff_header.chunkID[1] != 'I' || riff_header.chunkID[2] != 'F' || riff_header.chunkID[3] != 'F') ||
		(riff_header.format[0] != 'W' || riff_header.format[1] != 'A' || riff_header.format[2] != 'V' || riff_header.format[3] != 'E'))
	{
		fprintf(gpALFile, "Error while checking the RIFF and  WAVE tag\n");
		UninitializeAL();
		return FALSE;
	}

	//read the second chunk for wave info
	soundFile.read((char *)&wave_format, sizeof(WAVE_Format));
	//makeing sure that the wav format is something that we can work with
	if (wave_format.subChunkID[0] != 'f' || wave_format.subChunkID[1] != 'm' || wave_format.subChunkID[2] != 't' || wave_format.subChunkID[3] != ' ')
	{
		fprintf(gpALFile, "Error while checking the second chunk for wave info\n");
		UninitializeAL();
		return FALSE;
	}

	//check for extra parameters
	if (wave_format.subChunkSize > 16)
		soundFile.seekg(sizeof(int16_t), ios_base::cur);

	//read the last byte of data before the sound file
	soundFile.read((char *)&wave_data, sizeof(WAVE_Data));
	//check for the data tag in memory
	if (wave_data.subChunkID[0] != 'd' || wave_data.subChunkID[1] != 'a' || wave_data.subChunkID[2] != 't' || wave_data.subChunkID[3] != 'a')
	{
		fprintf(gpALFile, "Error while checking the data tag in memory\n");
		UninitializeAL();
		return FALSE;
	}

	//allocate memory for wave data
	data = (unsigned char *)malloc(wave_data.subChunk2Size);

	//read in the sound data in soundData variable
	if (!soundFile.read((char *)data, wave_data.subChunk2Size))
	{
		fprintf(gpALFile, "Error while reading the soundData\n");
		UninitializeAL();
		return FALSE;
	}

	//set the variables we passed in with the data from the structs
	size = wave_data.subChunk2Size;
	frequency = wave_format.sampleRate;

	// the format is worked out by looking at the number of channels and the bits per sample.
	if (wave_format.numChannels == 1)
	{
		if (wave_format.bitsPerSample == 8)
			format = AL_FORMAT_MONO8;
		else if (wave_format.bitsPerSample == 16)
			format = AL_FORMAT_MONO16;
	}
	else if (wave_format.numChannels == 2)
	{
		if (wave_format.bitsPerSample == 8)
			format = AL_FORMAT_STEREO8;
		else if (wave_format.bitsPerSample == 16)
			format = AL_FORMAT_STEREO16;
	}

	//now we put our data BufWalking in the openAL BufWalking and check for error
	alGenBuffers(1, buff);
	alBufferData(*buff, format, (void *)data, size, frequency);
	error = alGetError();
	if (error != AL_NO_ERROR)
	{
		fprintf(gpALFile, "Error while alBufferData\n");
		UninitializeAL();
		return FALSE;
	}

	//clean up
	soundFile.close();
	fprintf(gpALFile, "returning for loadWAV\n");

	return TRUE;
}

void UninitializeAL(void)
{
	
	alDeleteSources(1, &SrcWalking);
	alDeleteBuffers(1, &BufWalking);
	alDeleteSources(1, &SrcRunning);
	alDeleteBuffers(1, &BufRunning);
	alDeleteSources(1, &SrcBackground);
	alDeleteBuffers(1, &BufBackground);
	alDeleteSources(1, &SrcAmbience);
	alDeleteBuffers(1, &BufAmbience);
	alDeleteSources(1, &SrcObjectPushed);
	alDeleteBuffers(1, &BufObjectPushed);
	alDeleteSources(1, &SrcFinalWin);
	alDeleteBuffers(1, &BufFinalWin);
	alDeleteSources(1, &SrcEndCredits);
	alDeleteBuffers(1, &BufEndCredits);

	if (ALContext)
	{
		ALDevice = alcGetContextsDevice(ALContext);
		alcMakeContextCurrent(NULL);
		alcDestroyContext(ALContext);
		ALContext = NULL;
		if (ALDevice)
		{
			alcCloseDevice(ALDevice);
			ALDevice = NULL;
		}
	}

	if (gpALFile)
	{
		fclose(gpALFile);
		gpALFile = NULL;
	}
}