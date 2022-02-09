#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "EllipseData.h"

PELLIPSE_DATA GetEllipseData(GLint iVerticesCount, GLint iSemiMajorAxis, GLint iSemiMinorAxis)
{
	int iPosIndex = 0;
	float fIncrement = 0;
	GLfloat fAngle = 0.0f;
	PELLIPSE_DATA pEllipseData = NULL;

	pEllipseData = (PELLIPSE_DATA)malloc(sizeof(ELLIPSE_DATA));
	if (NULL == pEllipseData)
	{
		return NULL;
	}
	memset(pEllipseData, 0, sizeof(ELLIPSE_DATA));

	pEllipseData->pfVertices = (GLfloat*)calloc(2 * (iVerticesCount + 1), sizeof(GLfloat));
	if (NULL == pEllipseData->pfVertices)
	{
		free(pEllipseData);
		pEllipseData = NULL;
		return NULL;
	}

	pEllipseData->iVerticesSize = 2 * (iVerticesCount+ 1) * sizeof(GLfloat);
	pEllipseData->iVerticesCount = iVerticesCount + 1; // +1 for GL_LINE_STRIP

	for (int i = 0; i < iVerticesCount + 1; i++)
	{
		
		pEllipseData->pfVertices[iPosIndex++] = iSemiMajorAxis * cos(fAngle);
		pEllipseData->pfVertices[iPosIndex++] = iSemiMinorAxis * sin(fAngle);
		
		fIncrement++;
		if (fIncrement > iVerticesCount)
		{
			fIncrement = 0.0f;
		}
		fAngle = 2 * 3.1415 * fIncrement / iVerticesCount;
	}

	return pEllipseData;
}

void FreeEllipseData(PELLIPSE_DATA *ppData)
{
	PELLIPSE_DATA pData = *ppData;

	if (NULL == ppData || NULL == pData)
	{
		return;
	}

	if (NULL != pData->pfVertices)
	{
		free(pData->pfVertices);
		pData->pfVertices = NULL;
	}

	free(pData);
	pData = NULL;

	return;
}