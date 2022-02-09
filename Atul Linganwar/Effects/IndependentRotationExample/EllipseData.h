#pragma once

#define PI 3.14159

typedef struct _ELLIPSE_DATA
{
	GLfloat* pfVertices;
	GLfloat* pfColors;

	GLint iVerticesSize;
	GLint iColorsSize;
	GLint iVerticesCount;

}ELLIPSE_DATA, *PELLIPSE_DATA;

PELLIPSE_DATA GetEllipseData(GLint iVerticesCount, GLint iSemiMajorAxis, GLint iSemiMinorAxis);
void FreeEllipseData(PELLIPSE_DATA *ppData);