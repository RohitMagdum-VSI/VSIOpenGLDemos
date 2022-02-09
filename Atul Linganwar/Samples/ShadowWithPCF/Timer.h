#pragma once
#include <Windows.h>
#include <stdio.h>

class Timer
{
private:
	double Resolution;
	unsigned __int64 gu64_base;
	float fDeltaTime;
	float fElapsedTime;
	int iFrameCount;
	__int64 iPreviousTime;
	__int64 iCurrentTime;
	__int64 iCountPerSecond;
	int fps__;
	float fLastFrame;
	float fSecondsPerCount;

public:
	void InitializeTimer(void);
	void SetDeltaTime(void);
	void CalculateFPS(void);
	float GetDeltaTime(void);
};