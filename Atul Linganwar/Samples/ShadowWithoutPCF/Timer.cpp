#include "Timer.h"

extern HWND ghwnd;

void Timer::InitializeTimer(void)
{
	// counts per second
	QueryPerformanceFrequency((LARGE_INTEGER *)&iCountPerSecond);

	// seconds per count
	fSecondsPerCount = 1.0f / iCountPerSecond;

	// previous time
	QueryPerformanceCounter((LARGE_INTEGER *)&iPreviousTime);
}

void Timer::SetDeltaTime(void)
{
	// get current count
	QueryPerformanceCounter((LARGE_INTEGER *)&iCurrentTime);

	// delta time
	fDeltaTime = (iCurrentTime - iPreviousTime) * fSecondsPerCount;
	fElapsedTime += fDeltaTime;
}

float Timer::GetDeltaTime(void)
{
	return fDeltaTime;
}

void Timer::CalculateFPS(void)
{
	iFrameCount++;

	if (fElapsedTime >= 1.0f)
	{
		fps__ = iFrameCount;

		iFrameCount = 0;
		fElapsedTime = 0.0f;
	}

	iPreviousTime = iCurrentTime;
}