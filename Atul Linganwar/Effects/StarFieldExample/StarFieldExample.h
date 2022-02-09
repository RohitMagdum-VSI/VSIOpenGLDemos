#pragma once

#define OGL_WINDOW_WIDTH				800
#define OGL_WINDOW_HEIGHT				600

// Functions

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize(void);
void ToggleFullScreen(void);
void Display(void);
void Update(void);
void Resize(int iWidth, int iHeight);
int LoadGLTexture(GLuint* texture, TCHAR imageResourceId[]);
void UnInitialize(void);