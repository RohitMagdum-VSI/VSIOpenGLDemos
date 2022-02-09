#pragma once

enum PLANET_AND_MOONS
{
	NONE = 0,
	SUN = 1,
	MERCURY,
	VENUS,
	EARTH,
	EARTH_MOON,
	MARS,
	MARS_PHOBOS,
	JUPITER,
	JUPITER_EUROPA,
	SATURN,
	SATURN_TITAN,
	URANUS,
	URANUS_AERIAL,
	NEPTUNE,
	NEPTUNE_TRITON,
	PLUTO
};

#define TOTAL_PLANETS		20

#define SCALE_FACTOR_SUN				2.4f
#define SCALE_FACTOR_MERCURY			1.0f
#define SCALE_FACTOR_VENUS				1.1f
#define SCALE_FACTOR_EARTH				1.2f
#define SCALE_FACTOR_EARTH_MOON			0.3f
#define SCALE_FACTOR_MARS				1.1f
#define SCALE_FACTOR_MARS_PHOBOS		0.2f
#define SCALE_FACTOR_JUPITER			1.8f
#define SCALE_FACTOR_JUPITER_EUROPA		0.3f
#define SCALE_FACTOR_SATURN				1.6f
#define SCALE_FACTOR_SATURN_TITAN		0.3f
#define SCALE_FACTOR_URANUS				1.5f
#define SCALE_FACTOR_URANUS_AERIAL		0.2f
#define SCALE_FACTOR_NEPTUNE			1.4f
#define SCALE_FACTOR_NEPTUNE_TRITON		0.2f
#define SCALE_FACTOR_PLUTO				0.7f

// Functions

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool Initialize(void);
void ToggleFullScreen(void);
void Display(void);
void Update(void);
void Resize(int iWidth, int iHeight);
void UnInitialize(void);

int LoadGLTextures(GLuint* texture, TCHAR imageResourceId[]);

void DrawPlanetsForPicking();
void DrawPlanetsForRendering();
PLANET_AND_MOONS ProcessForPicking();

void DrawPlanetAtOrbitPicking(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLint iObjectId);
void DrawPlanetAtOrbitRendering(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLuint uiTexture);
