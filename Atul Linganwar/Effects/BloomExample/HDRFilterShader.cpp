#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"

#include "HDRFilterShader.h"

extern FILE* gpFile;

bool InitializeHDRFilterShaderProgram(HDR_FILTER_SHADER* pShaderObj)
{
	pShaderObj->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* pchVertexShaderSource =
		"#version 430 core\n" \
		"\n" \
		"void main(void)\n" \
		"{\n" \
		"const vec4 vertices[] = vec4[](vec4(-1.0, -1.0, 0.5, 1.0), \n" \
		"								 vec4(1.0, -1.0, 0.5, 1.0), \n" \
		"								 vec4(-1.0, 1.0, 0.5, 1.0), \n" \
		"								 vec4(1.0, 1.0, 0.5, 1.0)); \n" \
		"gl_Position = vertices[gl_VertexID];\n" \
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
				fprintf(gpFile, "HDR Filter VertexShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* pchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"uniform sampler2D u_hdr_image_sampler;\n" \
		"out vec4 FragColor;\n" \
		"const float weights[] = float[](0.0024499299678342, \n" \
			"0.0043538453346397, \n" \
			"0.0073599963704157, \n" \
			"0.0118349786570722, \n" \
			"0.0181026699707781, \n" \
			"0.0263392293891488, \n" \
			"0.0364543006660986, \n" \
			"0.0479932050577658, \n" \
			"0.0601029809166942, \n" \
			"0.0715974486241365, \n" \
			"0.0811305381519717, \n" \
			"0.0874493212267511, \n" \
			"0.0896631113333857, \n" \
			"0.0874493212267511, \n" \
			"0.0811305381519717, \n" \
			"0.0715974486241365, \n" \
			"0.0601029809166942, \n" \
			"0.0479932050577658, \n" \
			"0.0364543006660986, \n" \
			"0.0263392293891488, \n" \
			"0.0181026699707781, \n" \
			"0.0118349786570722, \n" \
			"0.0073599963704157, \n" \
			"0.0043538453346397, \n" \
			"0.0024499299678342); \n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 c = vec4(0.0);\n" \
		"ivec2 P = ivec2(gl_FragCoord.yx) - ivec2(0, weights.length() >> 1);\n" \
		"int i;\n" \
		"for(i = 0; i < weights.length(); i++)\n" \
		"{\n" \
		"c += texelFetch(u_hdr_image_sampler, P+ivec2(0, i), 0) * weights[i];\n" \
		"}\n" \
		"FragColor = c;\n" \
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
				fprintf(gpFile, "HDR Filter FragmentShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pShaderObj->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiVertexShaderObject);
	glAttachShader(pShaderObj->ShaderObject.uiShaderProgramObject, pShaderObj->ShaderObject.uiFragmentShaderObject);

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

	pShaderObj->uiTextureHDRImageUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_hdr_image_sampler");

	return true;
}

void UnInitializeHDRFilterShaderProgram(HDR_FILTER_SHADER* pShaderObj)
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