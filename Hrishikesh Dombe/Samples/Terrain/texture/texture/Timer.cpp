#include<Windows.h>
#include"Timer.h"

Timer::Timer():mSecondsPerFrame(0.0),mDeltaTime(-1.0),mBaseTime(0),mPausedTime(0),mPrevTime(0),mCurrTime(0),mStopped(false)
{
    __int64 framesPerSecond;
    QueryPerformanceFrequency((LARGE_INTEGER*)&framesPerSecond);
    mSecondsPerFrame = 1.0/(double)framesPerSecond;
}

float Timer::TotalTime()
{
    if(mStopped)
    {
        return(float)(((mStopTime - mPausedTime)-mBaseTime)*mSecondsPerFrame);
    }
    else
    {
        return(float)(((mCurrTime-mPausedTime)-mBaseTime)*mSecondsPerFrame);
    }
}

float Timer::DeltaTime()
{
    return (float)mDeltaTime;
}

void Timer::Reset()
{
    __int64 currTime;
    QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

    mBaseTime = currTime;
    mPrevTime = currTime;
    mStopTime = 0;
    mStopped = false;
}

void Timer::Start()
{
    __int64 startTime;
    QueryPerformanceCounter((LARGE_INTEGER*)&startTime);

    if(mStopped)
    {
        mPausedTime += (startTime - mStopTime);

        mPrevTime = startTime;
        mStopTime = 0;
        mStopTime = false;
    }
}

void Timer::Stop()
{
    if(!mStopped)
    {
        __int64 currTime;
        QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

        mStopTime = currTime;
        mStopped = true;
    }
}

void Timer::Tick()
{
    if(mStopped)
    {
        mDeltaTime = 0.0;
        return;
    }

    __int64 currTime;
    QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
    mCurrTime = currTime;

    mDeltaTime = (mCurrTime - mPrevTime)*mSecondsPerFrame;

    mPrevTime = mCurrTime;
    if(mDeltaTime < 0.0)
    {
        mDeltaTime = 0.0;
    }
}
