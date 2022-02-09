#pragma once

#define PI 3.14159
#define ORBIT_VERTICES_COUNT			100

typedef struct _ELLIPSE_DATA
{
	GLfloat fSemiMajorAxis;
	GLfloat fSemiMinorAxis;

	GLfloat* pfVertices;
	GLfloat* pfColors;

	GLint iVerticesSize;
	GLint iColorsSize;
	GLint iVerticesCount;

}ELLIPSE_DATA, *PELLIPSE_DATA;

typedef struct _ELLIPTICAL_PATH
{
	ELLIPSE_DATA EllipseData;

	GLuint uiVAO;
	GLuint uiVBOPosition;
	GLuint uiVBOColor;

}ELLIPTICAL_PATH, *PELLIPTICAL_PATH;

extern PELLIPTICAL_PATH gpEllipsePathMercury;
extern PELLIPTICAL_PATH gpEllipsePathVenus;
extern PELLIPTICAL_PATH gpEllipsePathEarth;
extern PELLIPTICAL_PATH gpEllipsePathMars;
extern PELLIPTICAL_PATH gpEllipsePathJupiter;
extern PELLIPTICAL_PATH gpEllipsePathSaturn;
extern PELLIPTICAL_PATH gpEllipsePathUranus;
extern PELLIPTICAL_PATH gpEllipsePathNeptune;
extern PELLIPTICAL_PATH gpEllipsePathPluto;

bool InitializeAllPlanetsOrbitPath();
void UnInitializeAllPlanetsOrbitPath();

PELLIPTICAL_PATH InitializeEllipsePath(GLint iVerticesCount, GLfloat fSemiMajorAxis, GLfloat fSemiMinorAxis);
bool InitializeEllipsePathVAOs(PELLIPTICAL_PATH pEllipse);
void DrawEllipsePath(PELLIPTICAL_PATH pEllipse);
void FreeEllipseData(PELLIPTICAL_PATH* ppEllipse);
