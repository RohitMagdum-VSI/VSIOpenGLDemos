#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "ShaderUtils.h"

extern FILE* gpFile;

#define GL_CHECK(x) \
    x; \
    { \
        GLenum glError = glGetError(); \
        if(glError != GL_NO_ERROR) { \
            fprintf(gpFile, "glGetError() = %i (0x%.8x) at %s:%i\n", glError, glError, __FILE__, __LINE__); \
            exit(1); \
        } \
    }

bool CheckCompileStatus(GLuint shaderObject)
{
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char* szInfoLog = NULL;

	glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(shaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Compile log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	return true;
}

bool CheckLinkStatus(GLuint programObject)
{
	GLint iInfoLogLength = 0;
	char* szInfoLog = NULL;

	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(programObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			GLsizei written;
			glGetProgramInfoLog(programObject, iInfoLogLength, &written, szInfoLog);
			fprintf(gpFile, "Link log: %s\n", szInfoLog);
			free(szInfoLog);
			return false;
		}
	}

	return true;
}
