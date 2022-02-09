#pragma once

#define OGL_WINDOW_WIDTH				800
#define OGL_WINDOW_HEIGHT				600

typedef enum _OGL_ATTRIBUTES
{
	OGL_ATTRIBUTE_POSITION = 0,
	OGL_ATTRIBUTE_COLOR,
	OGL_ATTRIBUTE_NORMAL,
	OGL_ATTRIBUTE_TEXTURE

}OGL_ATTRIBUTES;

// Functions

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize(void);
void UnInitialize(void);

void Display(void);
void ToggleFullScreen(void);
void Resize(int iWidth, int iHeight);