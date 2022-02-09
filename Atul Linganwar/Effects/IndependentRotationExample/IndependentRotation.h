#pragma once

// Functions

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize(void);
void ToggleFullScreen(void);
void Display(void);
void Update(void);
void Resize(int iWidth, int iHeight);
void UnInitialize(void);

int LoadGLTextures(GLuint* texture, TCHAR imageResourceId[]);