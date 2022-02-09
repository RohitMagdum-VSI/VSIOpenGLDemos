#pragma once

#include <stdio.h>
#include <string>
#include <GL/glew.h>
#include <gl\GL.h>

#define GROUND_HEIGHT	0.0f
#define SKY_DOME_HEIGHT 10.0f

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#define FloorExtent 1000
#define StepSize	10

#define ORTHO_LEFT		-10.0f
#define ORTHO_RIGHT		10.0f
#define ORTHO_BOTTOM	-10.0f
#define ORTHO_TOP		10.0f
#define ORTHO_NEAR		-1000.0f
#define ORTHO_FAR		1000.0f

#define FRUSTUM_LEFT	-1.0f
#define FRUSTUM_RIGHT	1.0f
#define FRUSTUM_BOTTOM	-1.0f
#define FRUSTUM_TOP		1.0f
#define FRUSTUM_NEAR	0.1f
#define FRUSTUM_FAR		100.0f

#define PERSPECTIVE_NEAR	0.1f
#define PERSPECTIVE_FAR		100.0f

#define FOV_Y			45.0f
#define ZOOM_FACTOR		1.0f

#define BUFFER_OFFSET(X)  ((const void *) (X))

#define MSAA_SAMPLES	4


class CCommon
{
public:

	const char *logfilepath = "Logs/GeneralLog.txt";

	CCommon();
	~CCommon();

	FILE *pLogFile = NULL;
	void _check_gl_error(FILE *pFile, const char *filename, const char *currentfile, int currentline);
};

enum
{
	OGL_ATTRIBUTE_VERTEX = 0,
	OGL_ATTRIBUTE_COLOR, // 1
	OGL_ATTRIBUTE_NORMAL, // 2
	OGL_ATTRIBUTE_TEXTURE0, // 3
	OGL_ATTRIBUTE_TANGENT, // 4
	OGL_ATTRIBUTE_BRIGHTNESS, // 5
	OGL_ATTRIBUTE_INSTANCE_MATRIX, // 6, 7, 8, 9
	OGL_ATTRIBUTE_BITANGENT // 10
};



