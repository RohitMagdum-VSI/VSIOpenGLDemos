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
	PLUTO,
	SATURN_RING_Tex,
};

#define TOTAL_PLANETS		20

#define SCALE_FACTOR_SUN				2.4f
#define SCALE_FACTOR_MERCURY			1.0f
#define SCALE_FACTOR_VENUS				1.1f
#define SCALE_FACTOR_EARTH				1.2f
#define SCALE_FACTOR_EARTH_MOON			0.6f
#define SCALE_FACTOR_MARS				1.1f
#define SCALE_FACTOR_MARS_PHOBOS		0.5f
#define SCALE_FACTOR_JUPITER			1.8f
#define SCALE_FACTOR_JUPITER_EUROPA		0.6f
#define SCALE_FACTOR_SATURN				1.6f
#define SCALE_FACTOR_SATURN_TITAN		0.7f
#define SCALE_FACTOR_URANUS				1.5f
#define SCALE_FACTOR_URANUS_AERIAL		0.6f
#define SCALE_FACTOR_NEPTUNE			1.4f
#define SCALE_FACTOR_NEPTUNE_TRITON		0.5f
#define SCALE_FACTOR_PLUTO				0.7f

#define ORBITAL_VELOCITY_FACTOR_MERCURY			2.5f * 2
#define ORBITAL_VELOCITY_FACTOR_VENUS			1.8f * 2
#define ORBITAL_VELOCITY_FACTOR_EARTH			1.4f * 2
#define ORBITAL_VELOCITY_FACTOR_MARS			1.2f * 2
#define ORBITAL_VELOCITY_FACTOR_JUPITER			1.0f * 2
#define ORBITAL_VELOCITY_FACTOR_SATURN			0.8f * 2
#define ORBITAL_VELOCITY_FACTOR_URANUS			0.6f * 2
#define ORBITAL_VELOCITY_FACTOR_NEPTUNE			0.4f * 2
#define ORBITAL_VELOCITY_FACTOR_PLUTO			0.2f * 2


extern GLfloat gfMoonAngleTranslate;
extern GLfloat gfMercuryAngleTranslate;
extern GLfloat gfVenusAngleTranslate;
extern GLfloat gfEarthAngleTranslate;
extern GLfloat gfMarsAngleTranslate;
extern GLfloat gfJupiterAngleTranslate;
extern GLfloat gfSaturnAngleTranslate;
extern GLfloat gfUranusAngleTranslate;
extern GLfloat gfNeptuneAngleTranslate;
extern GLfloat gfPlutoAngleTranslate;

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

void DrawPlanetAtOrbitPicking(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLfloat fOrbitalVelocity, GLint iObjectId);
void DrawPlanetAtOrbitRendering(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLfloat fOrbitalVelocity, GLuint uiTexture);

void DrawSaturnRingAtOrbitPicking(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLfloat fOrbitalVelocity, GLint iObjectId);
void DrawSaturnRingAtOrbitRendering(PELLIPTICAL_PATH pEllipsePath, GLfloat fScaleFactor, GLfloat fOrbitalVelocity, GLuint uiTexture);

void ClearMatrices();
