#pragma once

#define OGL_WINDOW_WIDTH			800
#define OGL_WINDOW_HEIGHT			600

typedef enum _OGL_VERTEX_ATTRIBUTES
{
	OGL_ATTRIBUTE_POSITION = 0,
	OGL_ATTRIBUTE_COLOR,
	OGL_ATTRIBUTE_NORMAL,
	OGL_ATTRIBUTE_TEXTURE,

}OGL_VERTEX_ATTRIBUTE;

// WndProc
LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize();
void Uninitialize();

void ToggleFullscreen();
void Display();
void Resize(int iWidth, int iHeight);