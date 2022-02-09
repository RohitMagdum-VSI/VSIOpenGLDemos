#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "ShaderUtils.h"
#include "MarbalShader.h"

extern FILE* gpFile;

bool InitializeMarbalShaderProgram(MARBAL_SHADER* pShaderObj)
{
	pShaderObj->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* pchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform mat4 MVMatrix;\n" \
		"uniform mat4 ProjectionMatrix;\n" \
		"uniform float offset;\n" \
		"in vec4 MCvertex;\n" \
		"in vec3 MCnormal;\n" \
		"in vec2 MCtexcoord;\n" \
		"out vec3 MCposition;\n" \
		"void main(void)\n" \
		"{\n" \
		"MCposition = vec3(MCvertex) * offset;\n" \
		"gl_Position = ProjectionMatrix * MVMatrix * MCvertex;\n" \
		"}\n";

	glShaderSource(pShaderObj->ShaderObject.uiVertexShaderObject, 1, (const GLchar**)&pchVertexShaderSource, NULL);

	glCompileShader(pShaderObj->ShaderObject.uiVertexShaderObject);
	if (false == CheckCompileStatus(pShaderObj->ShaderObject.uiVertexShaderObject))
	{
		fprintf(gpFile, "Sun Vertex Shader Compilation Error\n");
		return false;
	}

	pShaderObj->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* pchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"uniform sampler3D Noise;\n" \
		"uniform vec3 MarbleColor;\n" \
		"uniform vec3 VeinColor;\n" \
		"uniform float Scale = 1.2;\n" \
		"uniform float offset;\n" \
		"in vec3 MCposition;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 noisevec = texture(Noise, (MCposition));\n" \
		"float intensity = abs(noisevec[0] - 0.25) +\n" \
		"					abs(noisevec[1] - 0.125) + \n" \
		"					abs(noisevec[2] - 0.0625) + \n" \
		"					abs(noisevec[3] - 0.03125);\n" \
		"float sineval = sin(MCposition.y * 6.0 + intensity * 12.0) * 0.5 + 0.5;\n" \
		"vec3 color = mix(VeinColor, MarbleColor, sineval * 1.2);\n" \
		"FragColor = vec4(color, 1.0);\n" \
		"}\n";

	glShaderSource(pShaderObj->ShaderObject.uiFragmentShaderObject, 1, (const GLchar**)&pchFragmentShaderSource, NULL);

	glCompileShader(pShaderObj->ShaderObject.uiFragmentShaderObject);
	if (false == CheckCompileStatus(pShaderObj->ShaderObject.uiFragmentShaderObject))
	{
		fprintf(gpFile, "Sun Fragment Shader Compilation Error\n");
		return false;
	}

	pShaderObj->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "MCvertex");
	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE0, "MCtexcoord");
	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_NORMAL, "MCnormal");

	glLinkProgram(pShaderObj->ShaderObject.uiShaderProgramObject);
	if (false == CheckLinkStatus(pShaderObj->ShaderObject.uiShaderProgramObject))
	{
		fprintf(gpFile, "Sun Shader Program Object Linking Error\n");
		return false;
	}

	pShaderObj->uiMVMatrix = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "MVMatrix");
	pShaderObj->uiProjectionMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "ProjectionMatrix");

	pShaderObj->uiOffsetUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "offset");
	pShaderObj->uiVeinColorUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "VeinColor");
	pShaderObj->uiMarbalColorUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "MarbleColor");
	pShaderObj->uiScaleUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "Scale");

	pShaderObj->uiTextureSamplerUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "Noise");

	return true;
}

void UnInitializeMarbalShaderProgram(MARBAL_SHADER* pShaderObj)
{
	glDetachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glDetachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glDeleteShader(pShaderObj->ShaderObject.uiVertexShaderObject);
	pShaderObj->ShaderObject.uiVertexShaderObject = 0;

	glDeleteShader(pShaderObj->ShaderObject.uiFragmentShaderObject);
	pShaderObj->ShaderObject.uiFragmentShaderObject = 0;

	glDeleteProgram(pShaderObj->ShaderObject.uiShaderProgramObject);
	pShaderObj->ShaderObject.uiShaderProgramObject = 0;

	return;
}