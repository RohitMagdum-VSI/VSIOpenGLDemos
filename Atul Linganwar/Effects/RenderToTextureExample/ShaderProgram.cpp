#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl/glew.h>
#include<gl/GL.h>

#include "../common/vmath.h"
#include "VertexData.h"
#include "ShaderProgram.h"
#include "RenderToTexture.h"

ShaderProgram::ShaderProgram(SHADER_TYPE type)
{
	m_uiVertexShaderObject = 0;
	m_uiFragmentShaderObject = 0;
	m_uiShaderProgramObject = 0;

	m_uiMVPUniform = 0;
	m_uiTextureSamplerUniform = 0;

	m_uiTextureKundali = 0;
	m_uiTextureStone = 0;

	m_shaderType = type;
}

ShaderProgram::~ShaderProgram()
{

}

bool ShaderProgram::Init()
{
	bool bRet = false;
	switch (m_shaderType)
	{
	case COLOR_SHADER:
		bRet = InitializeColorProgram();
		break;

	case TEXTURE_SHADER:
		bRet = InitializeTextureProgram();
		break;

	case QUAD_TEXTURE_SHADER:
		bRet = InitializeTextureProgramQuad();
		break;

	case NORMAL_SHADER:
		break;
	}

	return bRet;
}

void ShaderProgram::DeInit()
{
	switch (m_shaderType)
	{
	case COLOR_SHADER:
		UnInitializeColorProgram();
		break;

	case TEXTURE_SHADER:
		UnInitializeTextureProgram();
		break;

	case QUAD_TEXTURE_SHADER:
		UnInitializeTextureProgramQuad();
		break;

	case NORMAL_SHADER:
		break;
	}
}

bool ShaderProgram::InitializeColorProgram(void)
{
	// Create vertex shader
	m_uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec4 vColor;" \
		"out vec4 vOutColor;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"vOutColor = vColor;" \
		"}";

	glShaderSource(m_uiVertexShaderObject, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(m_uiVertexShaderObject);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(m_uiVertexShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(m_uiVertexShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(m_uiVertexShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeColorProgram();
				return false;
			}
		}
	}

	// Create fragment shader
	m_uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec4 vOutColor;" \
		"out vec4 FragColor;" \
		"void main(void)" \
		"{" \
		"FragColor = vOutColor;" \
		"}";

	glShaderSource(m_uiFragmentShaderObject, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(m_uiFragmentShaderObject);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(m_uiFragmentShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(m_uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(m_uiFragmentShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeColorProgram();
				return false;
			}
		}
	}

	// Create shader program
	m_uiShaderProgramObject = glCreateProgram();

	glAttachShader(m_uiShaderProgramObject, m_uiVertexShaderObject);
	glAttachShader(m_uiShaderProgramObject, m_uiFragmentShaderObject);

	glBindAttribLocation(m_uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(m_uiShaderProgramObject, OGL_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(m_uiShaderProgramObject);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(m_uiShaderProgramObject, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(m_uiShaderProgramObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(m_uiShaderProgramObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeColorProgram();
				return false;
			}
		}
	}

	// get mvp uniform location
	m_uiMVPUniform = glGetUniformLocation(m_uiShaderProgramObject, "u_mvp_matrix");

	return true;
}

void ShaderProgram::UnInitializeColorProgram(void)
{
	glDetachShader(m_uiShaderProgramObject, m_uiVertexShaderObject);
	glDetachShader(m_uiShaderProgramObject, m_uiFragmentShaderObject);

	glDeleteShader(m_uiVertexShaderObject);
	m_uiVertexShaderObject = 0;

	glDeleteShader(m_uiFragmentShaderObject);
	m_uiFragmentShaderObject = 0;

	glDeleteProgram(m_uiShaderProgramObject);
	m_uiShaderProgramObject = 0;

	return;
}

bool ShaderProgram::InitializeTextureProgram(void)
{
	// Create vertex shader
	m_uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec4 vPosition;" \
		"in vec2 vTexture0_Coord;" \
		"out vec2 out_texture0_coord;" \
		"uniform mat4 u_mvp_matrix;" \
		"void main(void)" \
		"{" \
		"gl_Position = u_mvp_matrix * vPosition;" \
		"out_texture0_coord = vTexture0_Coord;" \
		"}";

	glShaderSource(m_uiVertexShaderObject, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(m_uiVertexShaderObject);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(m_uiVertexShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(m_uiVertexShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(m_uiVertexShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeTextureProgram();
				return false;
			}
		}
	}

	// Create fragment shader
	m_uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_texture0_sampler;" \
		"void main(void)" \
		"{" \
		"FragColor = texture(u_texture0_sampler, out_texture0_coord);" \
		"}";

	glShaderSource(m_uiFragmentShaderObject, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(m_uiFragmentShaderObject);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(m_uiFragmentShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(m_uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(m_uiFragmentShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeTextureProgram();
				return false;
			}
		}
	}

	// Create shader program
	m_uiShaderProgramObject = glCreateProgram();

	glAttachShader(m_uiShaderProgramObject, m_uiVertexShaderObject);
	glAttachShader(m_uiShaderProgramObject, m_uiFragmentShaderObject);

	glBindAttribLocation(m_uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(m_uiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE, "vTexture0_Coord");

	glLinkProgram(m_uiShaderProgramObject);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(m_uiShaderProgramObject, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(m_uiShaderProgramObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(m_uiShaderProgramObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeTextureProgram();
				return false;
			}
		}
	}

	// get mvp uniform location
	m_uiMVPUniform = glGetUniformLocation(m_uiShaderProgramObject, "u_mvp_matrix");
	m_uiTextureSamplerUniform = glGetUniformLocation(m_uiShaderProgramObject, "u_texture0_sampler");

	glEnable(GL_TEXTURE_2D);

	LoadTexture(&m_uiTextureKundali, MAKEINTRESOURCE(IDBITMAP_KUNDALI));
	LoadTexture(&m_uiTextureStone, MAKEINTRESOURCE(IDBITMAP_STONE));

	return true;
}

void ShaderProgram::UnInitializeTextureProgram(void)
{
	glDetachShader(m_uiShaderProgramObject, m_uiVertexShaderObject);
	glDetachShader(m_uiShaderProgramObject, m_uiFragmentShaderObject);

	glDeleteShader(m_uiVertexShaderObject);
	m_uiVertexShaderObject = 0;

	glDeleteShader(m_uiFragmentShaderObject);
	m_uiFragmentShaderObject = 0;

	glDeleteProgram(m_uiShaderProgramObject);
	m_uiShaderProgramObject = 0;

	return;
}

bool ShaderProgram::InitializeTextureProgramQuad(void)
{
	// Create vertex shader
	m_uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
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

	glShaderSource(m_uiVertexShaderObject, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(m_uiVertexShaderObject);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(m_uiVertexShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(m_uiVertexShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(m_uiVertexShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeTextureProgram();
				return false;
			}
		}
	}

	// Create fragment shader
	m_uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core" \
		"\n" \
		"in vec2 out_texture0_coord;" \
		"out vec4 FragColor;" \
		"uniform sampler2D u_texture0_sampler;" \
		"void main(void)" \
		"{" \
		"FragColor = texture(u_texture0_sampler, out_texture0_coord);" \
		"}";

	glShaderSource(m_uiFragmentShaderObject, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(m_uiFragmentShaderObject);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(m_uiFragmentShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(m_uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(m_uiFragmentShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeTextureProgram();
				return false;
			}
		}
	}

	// Create shader program
	m_uiShaderProgramObject = glCreateProgram();

	glAttachShader(m_uiShaderProgramObject, m_uiVertexShaderObject);
	glAttachShader(m_uiShaderProgramObject, m_uiFragmentShaderObject);

	glBindAttribLocation(m_uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(m_uiShaderProgramObject, OGL_ATTRIBUTE_TEXTURE, "vTexture0_Coord");

	glLinkProgram(m_uiShaderProgramObject);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(m_uiShaderProgramObject, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(m_uiShaderProgramObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(m_uiShaderProgramObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				UnInitializeTextureProgram();
				return false;
			}
		}
	}

	// get mvp uniform location
	m_uiTextureSamplerUniform = glGetUniformLocation(m_uiShaderProgramObject, "u_texture0_sampler");

	glEnable(GL_TEXTURE_2D);

	LoadTexture(&m_uiTextureKundali, MAKEINTRESOURCE(IDBITMAP_KUNDALI));
	LoadTexture(&m_uiTextureStone, MAKEINTRESOURCE(IDBITMAP_STONE));

	return true;
}

void ShaderProgram::UnInitializeTextureProgramQuad(void)
{
	glDetachShader(m_uiShaderProgramObject, m_uiVertexShaderObject);
	glDetachShader(m_uiShaderProgramObject, m_uiFragmentShaderObject);

	glDeleteShader(m_uiVertexShaderObject);
	m_uiVertexShaderObject = 0;

	glDeleteShader(m_uiFragmentShaderObject);
	m_uiFragmentShaderObject = 0;

	glDeleteProgram(m_uiShaderProgramObject);
	m_uiShaderProgramObject = 0;

	return;
}

int ShaderProgram::LoadTexture(GLuint* texture, TCHAR imageResourceId[])
{
	//variables declaration
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

	//code
	glGenTextures(1, texture);
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if (hBitmap)
	{
		iStatus = TRUE;
		GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
	}

	return(iStatus);
}