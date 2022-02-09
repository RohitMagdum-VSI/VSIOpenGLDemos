#pragma once


void SetNoiseFrequency(int frequency);
void normalize2(double v[2]);
void normalize3(double v[3]);
void initNoise();
double noise2(double vec[2]);
double noise3(double vec[3]);

void make3DNoiseTexture(void);

void GenerateNoiseTexture(void);
