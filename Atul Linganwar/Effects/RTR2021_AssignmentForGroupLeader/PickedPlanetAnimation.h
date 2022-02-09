#pragma once

typedef struct _PICKING_DATA
{
	GLfloat gfAngleTranslate;

}PICKING_DATA;

typedef struct _POINT_TRACE
{
	float x;
	float y;
	float z;

}POINT_TRACE;

void DrawPlanetPickAnimation(PLANET_AND_MOONS pick);

void AnimateSun();
void AnimateMercury();
void AnimateVenus();
void AnimateEarth();
void AnimateMars();
void AnimateJupiter();
void AnimateSaturn();
void AnimateUranus();
void AnimateNeptune();
void AnimatePluto();

void FillPickingData(PLANET_AND_MOONS pick);
void ClearPickingData();

// range of z should be (p1.z to p2.z)
POINT_TRACE GetPointOnLine(POINT_TRACE p1, POINT_TRACE p2, float z);

void DrawBorderedViewport(GLint x, GLint y, GLfloat fWidth, GLfloat fHeight);
void ShowSunViewport();
void ShowMercuryViewport();
void ShowVenusViewport();
void ShowEarthViewport();
void ShowMarsViewport();
void ShowJupiterViewport();
void ShowSaturnViewport();
void ShowUranusViewport();
void ShowNeptuneViewport();
void ShowPlutoViewport();

void RenderPlanetOntoABorderedViewport(GLuint uiPlanetTexture, GLfloat fScaleFactor, GLuint uiMoonTexture, GLfloat fScaleFactorMoon, bool bMoon);

void DrawNoiseSun(GLfloat fScaleFactor);
void DrawNoiseMarble(GLfloat fScaleFactor, GLfloat fScaleFactorMoon, bool bMoon, vmath::vec3 vecColor1, vmath::vec3 vecColor2);
