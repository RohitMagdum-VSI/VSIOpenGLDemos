
class Timer
{
public:
    Timer();

    float TotalTime();
    float DeltaTime();
    
    void Reset();
    void Start();
    void Stop();
    void Tick();

private:
    double mSecondsPerFrame;
    double mDeltaTime;

    __int64 mBaseTime;
    __int64 mPausedTime;
    __int64 mStopTime;
    __int64 mPrevTime;
    __int64 mCurrTime;

    bool mStopped;
};

