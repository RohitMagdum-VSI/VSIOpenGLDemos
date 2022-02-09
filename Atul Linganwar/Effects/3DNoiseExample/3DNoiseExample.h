#pragma once

#define IDBITMAP_SMILEY 101

// Functions

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize(void);
void ToggleFullScreen(void);
void Display(void);
void Update(void);
void Resize(int iWidth, int iHeight);
int LoadGLTexture(GLuint* texture, TCHAR imageResourceId[]);
void UnInitialize(void);

void make3DNoiseTexture(void);

