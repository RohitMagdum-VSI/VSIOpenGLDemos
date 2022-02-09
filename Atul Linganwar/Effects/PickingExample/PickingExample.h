#pragma once

#define OGL_WINDOW_WIDTH				800
#define OGL_WINDOW_HEIGHT				600

#define IDBITMAP_KUNDALI	101
#define IDBITMAP_STONE		102
#define IDBITMAP_SMILEY		103

typedef enum _OGL_ATTRIBUTES
{
	OGL_ATTRIBUTE_POSITION = 0,
	OGL_ATTRIBUTE_COLOR,
	OGL_ATTRIBUTE_NORMAL,
	OGL_ATTRIBUTE_TEXTURE

}OGL_ATTRIBUTES;

extern FILE* gpFile;

enum PLANETS
{
	SUN = 1,
	MERCURY,
	VENUS,
	EARTH,
	MARS,
	JUPITER,
	SATURN,
	URANUS,
	NEPTUNE,
	PLUTO
};

struct PIXEL_DATA
{
	GLfloat fObjectIndex;
	GLfloat fY;
	GLfloat fZ;
	GLfloat fA;
};

extern vmath::mat4 gPerspectiveProjectionMatrix;

// Functions

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize(void);
void ToggleFullScreen(void);
void Display(void);
void Update(void);
void Resize(int iWidth, int iHeight);
void UnInitialize(void);

