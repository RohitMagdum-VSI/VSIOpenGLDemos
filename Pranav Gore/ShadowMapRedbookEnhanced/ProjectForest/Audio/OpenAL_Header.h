#pragma once
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>

#include <AL/al.h>
#include <AL/alc.h>

#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtx\rotate_vector.hpp>

using namespace std;

BOOL loadWAV(string filename, ALuint *buff);

BOOL InitializeAL(void);

void InitializeAL_Listener();
void ChangeListenerAndSourcePos(glm::vec3 cameraPos);

BOOL InitializeAL_WalkingSource(void);
BOOL InitializeAL_RunningSource(void);
BOOL InitializeAL_BackgroundSource(void);
BOOL InitializeAL_AmbienceSource(void);
BOOL InitializeAL_ObjectPushed(void);
BOOL InitializeAL_FinalWinSource(void);
BOOL InitializeAL_EndCredits(void);

void PlayAL_WalkingSource(void);
void PauseAL_WalkingSource(void);

void PlayAL_RunningSource(void);
void PauseAL_RunningSource(void);

void PlayAL_BackgroundSource(void);
void StopAL_BackgroundSource(void);
void PlayAL_AmbienceSource(void);
void PlayAL_ObjectPushed(void);
void PlayAL_FinalWinSource(void);
void PlayAL_EndCredits(void);

void UninitializeAL(void);