#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "BasicQuadRTTShader.h"

extern FILE* gpFile;

bool InitializeBasicQuadRTTTextureShaderProgram(BASIC_QUAD_RTT_TEXTURE_SHADER* pShaderObj)
{
	pShaderObj->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* pchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexture0_Coord;" \
		"out vec2 out_texture0_coord;" \
		"void main(void)" \
		"{" \
		"gl_Position = vec4(vPosition.x, vPosition.y, 0.0, 1.0);" \
		"out_texture0_coord = vTexture0_Coord;" \
		"}";

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
				fprintf(gpFile, "Basic Color VertexShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* pchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;" \
		"uniform sampler3D u_texture0_sampler;" \
		"void main(void)" \
		"{" \
		"FragColor = texture(u_texture0_sampler, vec3(out_texture0_coord, 0.0));" \
		"}";

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
				fprintf(gpFile, "Basic Color FragmentShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(pShaderObj->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");

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
			fprintf(gpFile, "Basic Color Shader Program Link log: %s\n", szInfoLog);
			free(szInfoLog);
			return false;
		}
	}

	pShaderObj->uiModelMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_model_matrix");
	pShaderObj->uiViewMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_view_matrix");
	pShaderObj->uiProjectionMatrixUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_projection_matrix");
	pShaderObj->uiTextureSamplerUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_texture0_sampler");

	return true;
}

void UnInitializeBasicQuadTextureShaderProgram(BASIC_QUAD_RTT_TEXTURE_SHADER* pShaderObj)
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