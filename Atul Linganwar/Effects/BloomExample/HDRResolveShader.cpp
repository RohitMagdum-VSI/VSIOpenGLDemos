#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"

#include "HDRResolveShader.h"

extern FILE* gpFile;

bool InitializeHDRResolveShaderProgram(HDR_RESOLVE_SHADER* pShaderObj)
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
				fprintf(gpFile, "HDR Resolve VertexShader compilation log: %s\n", szInfoLog);
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
		"uniform sampler2D u_bloom_image_sampler;\n" \
		"out vec4 FragColor;\n" \
		"uniform float exposure = 0.9;\n" \
		"uniform float bloom_factor = 1.0;\n" \
		"uniform float scene_factor = 1.0;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 c = vec4(0.0);\n" \
		"c += texelFetch(u_hdr_image_sampler, ivec2(gl_FragCoord.xy), 0) * scene_factor;\n" \
		"c += texelFetch(u_bloom_image_sampler, ivec2(gl_FragCoord.xy), 0) * bloom_factor;\n" \
		"c.rgb = vec3(1.0) - exp(-c.rgb * exposure);\n" \
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
				fprintf(gpFile, "HDR Resolve FragmentShader compilation log: %s\n", szInfoLog);
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
	pShaderObj->uiTextureBloomImageUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "u_bloom_image_sampler");

	pShaderObj->uiExposureUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "exposure");
	pShaderObj->uiBloomFactorUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "bloom_factor");
	pShaderObj->uiSceneFactorUniform = glGetUniformLocation(pShaderObj->ShaderObject.uiShaderProgramObject, "scene_factor");

	return true;
}

void UnInitializeHDRResolveShaderProgram(HDR_RESOLVE_SHADER* pShaderObj)
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