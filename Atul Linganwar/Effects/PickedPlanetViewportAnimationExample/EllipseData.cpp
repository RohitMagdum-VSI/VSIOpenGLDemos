#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "EllipseData.h"

PELLIPTICAL_PATH gpEllipsePathMercury;
PELLIPTICAL_PATH gpEllipsePathVenus;
PELLIPTICAL_PATH gpEllipsePathEarth;
PELLIPTICAL_PATH gpEllipsePathMars;
PELLIPTICAL_PATH gpEllipsePathJupiter;
PELLIPTICAL_PATH gpEllipsePathSaturn;
PELLIPTICAL_PATH gpEllipsePathUranus;
PELLIPTICAL_PATH gpEllipsePathNeptune;
PELLIPTICAL_PATH gpEllipsePathPluto;

#define SEMIMAJOR_AXIS			10
#define SEMIMINOR_AXIS			6

#define SCALE_AXIS_MERCURY		2
#define SCALE_AXIS_VENUS		3
#define SCALE_AXIS_EARTH		4
#define SCALE_AXIS_MARS			5
#define SCALE_AXIS_JUPITER		6
#define SCALE_AXIS_SATURN		7.2
#define SCALE_AXIS_URANUS		8.4
#define SCALE_AXIS_NEPTUNE		9.5
#define SCALE_AXIS_PLUTO		10.4


bool InitializeAllPlanetsOrbitPath()
{
	// Mercury
	gpEllipsePathMercury = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_MERCURY, SEMIMINOR_AXIS * SCALE_AXIS_MERCURY);
	if (NULL == gpEllipsePathMercury)
	{
		return false;
	}

	// Venus
	gpEllipsePathVenus = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_VENUS, SEMIMINOR_AXIS * SCALE_AXIS_VENUS);
	if (NULL == gpEllipsePathVenus)
	{
		return false;
	}

	// Earth
	gpEllipsePathEarth = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_EARTH, SEMIMINOR_AXIS * SCALE_AXIS_EARTH);
	if (NULL == gpEllipsePathEarth)
	{
		return false;
	}

	// Mars
	gpEllipsePathMars = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_MARS, SEMIMINOR_AXIS * SCALE_AXIS_MARS);
	if (NULL == gpEllipsePathMars)
	{
		return false;
	}

	// Jupiter
	gpEllipsePathJupiter = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_JUPITER, SEMIMINOR_AXIS * SCALE_AXIS_JUPITER);
	if (NULL == gpEllipsePathJupiter)
	{
		return false;
	}

	// Saturn
	gpEllipsePathSaturn = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_SATURN, SEMIMINOR_AXIS * SCALE_AXIS_SATURN);
	if (NULL == gpEllipsePathSaturn)
	{
		return false;
	}

	// Uranus
	gpEllipsePathUranus = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_URANUS, SEMIMINOR_AXIS * SCALE_AXIS_URANUS);
	if (NULL == gpEllipsePathUranus)
	{
		return false;
	}

	// Neptune
	gpEllipsePathNeptune = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_NEPTUNE, SEMIMINOR_AXIS * SCALE_AXIS_NEPTUNE);
	if (NULL == gpEllipsePathNeptune)
	{
		return false;
	}

	// Pluto
	gpEllipsePathPluto = InitializeEllipsePath(ORBIT_VERTICES_COUNT, SEMIMAJOR_AXIS * SCALE_AXIS_PLUTO, SEMIMINOR_AXIS * SCALE_AXIS_PLUTO);
	if (NULL == gpEllipsePathPluto)
	{
		return false;
	}

	return true;
}

void UnInitializeAllPlanetsOrbitPath()
{
	FreeEllipseData(&gpEllipsePathMercury);
	FreeEllipseData(&gpEllipsePathVenus);
	FreeEllipseData(&gpEllipsePathEarth);
	FreeEllipseData(&gpEllipsePathMars);
	FreeEllipseData(&gpEllipsePathJupiter);
	FreeEllipseData(&gpEllipsePathSaturn);
	FreeEllipseData(&gpEllipsePathUranus);
	FreeEllipseData(&gpEllipsePathNeptune);
	FreeEllipseData(&gpEllipsePathPluto);
}

PELLIPTICAL_PATH InitializeEllipsePath(GLint iVerticesCount, GLfloat fSemiMajorAxis, GLfloat fSemiMinorAxis)
{
	int iPosIndex = 0;
	int iColorIndex = 0;
	float fIncrement = 0;
	GLfloat fAngle = 0.0f;
	PELLIPTICAL_PATH pEllipsePath = NULL;

	pEllipsePath = (PELLIPTICAL_PATH)malloc(sizeof(ELLIPTICAL_PATH));
	if (NULL == pEllipsePath)
	{
		return NULL;
	}
	memset(pEllipsePath, 0, sizeof(ELLIPTICAL_PATH));

	pEllipsePath->EllipseData.pfVertices = (GLfloat*)calloc(3 * (iVerticesCount + 1), sizeof(GLfloat));
	if (NULL == pEllipsePath->EllipseData.pfVertices)
	{
		free(pEllipsePath);
		pEllipsePath = NULL;
		return NULL;
	}

	pEllipsePath->EllipseData.pfColors = (GLfloat*)calloc(3 * (iVerticesCount + 1), sizeof(GLfloat));
	if (NULL == pEllipsePath->EllipseData.pfColors)
	{
		free(pEllipsePath->EllipseData.pfVertices);
		pEllipsePath->EllipseData.pfVertices = NULL;
		free(pEllipsePath);
		pEllipsePath = NULL;
		return NULL;
	}

	pEllipsePath->EllipseData.fSemiMajorAxis = fSemiMajorAxis;
	pEllipsePath->EllipseData.fSemiMinorAxis = fSemiMinorAxis;

	pEllipsePath->EllipseData.iVerticesSize = 3 * (iVerticesCount + 1) * sizeof(GLfloat);
	pEllipsePath->EllipseData.iColorsSize = 3 * (iVerticesCount + 1) * sizeof(GLfloat);
	
	pEllipsePath->EllipseData.iVerticesCount = iVerticesCount + 1; // +1 for GL_LINE_STRIP

	for (int i = 0; i < iVerticesCount + 1; i++)
	{
		pEllipsePath->EllipseData.pfVertices[iPosIndex++] = fSemiMajorAxis * (GLfloat)cos(fAngle);
		pEllipsePath->EllipseData.pfVertices[iPosIndex++] = 0.0f;
		pEllipsePath->EllipseData.pfVertices[iPosIndex++] = fSemiMinorAxis * (GLfloat)sin(fAngle);
		
		fIncrement++;
		if (fIncrement > iVerticesCount)
		{
			fIncrement = 0.0f;
		}
		fAngle = 2 * (GLfloat)3.1415 * fIncrement / iVerticesCount;

		// Color
		pEllipsePath->EllipseData.pfColors[iColorIndex++] = 1.0f;
		pEllipsePath->EllipseData.pfColors[iColorIndex++] = 0.0f;
		pEllipsePath->EllipseData.pfColors[iColorIndex++] = 0.0f;
	}

	if (false == InitializeEllipsePathVAOs(pEllipsePath))
	{
		free(pEllipsePath->EllipseData.pfColors);
		pEllipsePath->EllipseData.pfColors = NULL;
		free(pEllipsePath->EllipseData.pfVertices);
		pEllipsePath->EllipseData.pfVertices = NULL;
		free(pEllipsePath);
		pEllipsePath = NULL;
		return NULL;
	}

	return pEllipsePath;
}

bool InitializeEllipsePathVAOs(PELLIPTICAL_PATH pEllipse)
{
	glGenVertexArrays(1, &pEllipse->uiVAO);
	glBindVertexArray(pEllipse->uiVAO);

	glGenBuffers(1, &pEllipse->uiVBOPosition);
	glBindBuffer(GL_ARRAY_BUFFER, pEllipse->uiVBOPosition);
	glBufferData(GL_ARRAY_BUFFER, pEllipse->EllipseData.iVerticesSize, (const void*)pEllipse->EllipseData.pfVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &pEllipse->uiVBOColor);
	glBindBuffer(GL_ARRAY_BUFFER, pEllipse->uiVBOColor);
	glBufferData(GL_ARRAY_BUFFER, pEllipse->EllipseData.iColorsSize, (const void*)pEllipse->EllipseData.pfColors, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, NULL, 0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	return true;
}

void DrawEllipsePath(PELLIPTICAL_PATH pEllipse)
{
	glBindVertexArray(pEllipse->uiVAO);

	glDrawArrays(GL_LINE_STRIP, 0, pEllipse->EllipseData.iVerticesCount);

	glBindVertexArray(0);

	return;
}

void FreeEllipseData(PELLIPTICAL_PATH* ppEllipse)
{
	PELLIPTICAL_PATH pData = *ppEllipse;

	if (NULL == ppEllipse || NULL == pData)
	{
		return;
	}

	if (NULL != pData->EllipseData.pfColors)
	{
		free(pData->EllipseData.pfColors);
		pData->EllipseData.pfColors = NULL;
	}

	if (NULL != pData->EllipseData.pfVertices)
	{
		free(pData->EllipseData.pfVertices);
		pData->EllipseData.pfVertices = NULL;
	}

	free(pData);
	pData = NULL;

	return;
}