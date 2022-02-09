#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "PickingShader.h"

extern FILE* gpFile;

bool InitializePickingShaderProgram(PICKING_SHADER* pShaderObj)
{
	pShaderObj->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* pchVertexShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec4 vPosition\n;" \
		"uniform mat4 u_model_matrix;\n" \
		"uniform mat4 u_view_matrix;\n" \
		"uniform mat4 u_projection_matrix;\n" \
		"void main(void)\n" \
		"{\n" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;\n" \
		"}\n";

	glShaderSource(pShaderObj->ShaderObject.uiVertexShaderObject, 1, (const GLchar**)&pchVertexShaderSource, NULL);

	glCompileShader(pShaderObj->ShaderObject.uiVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char* szInfoLog = NULL;

	glGetShaderiv(pShaderObj->ShaderObject.uiVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(pShaderObj->ShaderObject.uiVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(pShaderObj->ShaderObject.uiVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Basic Planet VertexShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* pchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"out vec4 FragColor\n;" \
		"uniform int u_object_index_uniform\n;" \
		"void main(void)\n" \
		"{\n" \
		"FragColor = vec4(u_object_index_uniform / 255.0, 1.0, 0.0, 1.0);\n" \
		"}\n";

	glShaderSource(pShaderObj->ShaderObject.uiFragmentShaderObject, 1, (const GLchar**)&pchFragmentShaderSource, NULL);

	glCompileShader(pShaderObj->ShaderObject.uiFragmentShaderObject);
	glGetShaderiv(pShaderObj->ShaderObject.uiFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(pShaderObj->ShaderObject.uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(pShaderObj->ShaderObject.uiFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Basic Planet FragmentShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");

	glLinkProgram(pShaderObj->ShaderObject.uiShaderProgramObject);
	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(pShaderObj->ShaderObject.uiShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(pShaderObj->ShaderObject.uiShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			GLsizei written;
			glGetProgramInfoLog(pShaderObj->ShaderObject.uiShaderProgramObject, iInfoLogLength, &written, szInfoLog);
			fprintf(gpFile, "Basic Planet Shader Program Link log: %s\n", szInfoLog);
			free(szInfoLog);
			return false;
		}
	}
	pShaderObj->uiModelMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_model_matrix");
	pShaderObj->uiViewMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_view_matrix");
	pShaderObj->uiProjectionMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_projection_matrix");

	pShaderObj->uiObjectIdUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_object_index_uniform");

	return true;
}

void UnInitializePickingShaderProgram(PICKING_SHADER* pShaderObj)
{
	glDetachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glDetachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glDeleteShader(pShaderObj->ShaderObject.uiVertexShaderObject);
	pShaderObj->ShaderObject.uiVertexShaderObject = 0;

	glDeleteShader(pShaderObj->ShaderObject.uiFragmentShaderObject);
	pShaderObj->ShaderObject.uiFragmentShaderObject = 0;

	glDeleteProgram(pShaderObj->ShaderObject.uiShaderProgramObject);
	pShaderObj->ShaderObject.uiShaderProgramObject = 0;
}