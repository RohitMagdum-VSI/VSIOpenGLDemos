#pragma once

typedef struct _ORBIT 
{
	ELLIPSE_DATA EllipseData;

	GLuint vaoOrbit;
	GLuint vboOrbitPosition;
	GLuint vboOrbitColor;
}ORBIT, *P_ORBIT;

extern FLOAT gfPlanetAngle;

BOOL InitOrbit(ORBIT &Orbit, UINT uiMajorAxis, UINT uiMinorAxis);
void DrawOrbit(ORBIT &Orbit, COLOR_SHADER &ColorShader, vmath::mat4 &mvpMatrix);
void CleanupOrbit(ORBIT &Orbit);
FLOAT GetPlanetXPosition(ORBIT &Orbit, PLANETS_AND_SATELLITES PlanetEnum, FLOAT fAngle = gfPlanetAngle);
FLOAT GetPlanetZPosition(ORBIT &Orbit, PLANETS_AND_SATELLITES PlanetEnum, FLOAT fAngle = gfPlanetAngle);
void UpdatePlanetsAngle();
void DrawAllOrbits();
