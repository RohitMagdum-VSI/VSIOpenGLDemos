#pragma once

#define OGL_WINDOW_WIDTH				800
#define OGL_WINDOW_HEIGHT				600

#define ORTHO							100.0f

typedef enum _OGL_ATTRIBUTES
{
	OGL_ATTRIBUTE_POSITION = 0,
	OGL_ATTRIBUTE_COLOR,
	OGL_ATTRIBUTE_NORMAL,
	OGL_ATTRIBUTE_TEXTURE0

}OGL_ATTRIBUTES;

typedef struct _SHADER_OBJECT
{
	GLuint uiVertexShaderObject;
	GLuint uiFragmentShaderObject;
	GLuint uiShaderProgramObject;

}SHADER_OBJECT;