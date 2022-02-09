#pragma once

#define OGL_WINDOW_WIDTH				800
#define OGL_WINDOW_HEIGHT				600

#define to_radians(degree) (degree * M_PI / 180.0)
#define to_degrees(radian) (radian * 180.0 / M_PI)

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
void ToggleFullScreen(void);
void Display(void);
void Update(void);
void Resize(int iWidth, int iHeight);
void UnInitialize(void);