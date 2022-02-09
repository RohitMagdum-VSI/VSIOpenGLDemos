#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "StarField.h"
#include "KtxLoader.h"

extern FILE* gpFile;
extern vmath::mat4 gPerspectiveProjectionMatrix;

unsigned int uiSeed = 0x13371337;
static double gdCurrentTime = -100;

bool InitializeStarField(STAR_FIELD* pStarField)
{
	pStarField->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* glchVertexShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec3 vPosition;\n" \
		"in vec3 vColor;\n" \
		"uniform float u_time;\n" \
		"uniform mat4 u_projection_matrix;\n" \
		"out vec4 out_star_color;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 newVertex = vec4(vPosition, 1.0);\n" \
		"newVertex.z += u_time;\n" \
		"newVertex.z = fract(newVertex.z);\n" \
		"float size = 25.0 * newVertex.z * newVertex.z;\n" \
		"out_star_color = smoothstep(1.0, 7.0, size) * vec4(vColor, 1.0);" \
		"newVertex.z = (999.9 * newVertex.z) - 1000.0;\n" \
		"gl_Position = u_projection_matrix * newVertex;\n" \
		"gl_PointSize = size;\n" \
		"}\n";

	glShaderSource(pStarField->ShaderObject.uiVertexShaderObject, 1, (const GLchar**)&glchVertexShaderSource, NULL);

	glCompileShader(pStarField->ShaderObject.uiVertexShaderObject);
	GLint gliInfoLogLength = 0;
	GLint gliShaderComileStatus = 0;
	char* pszInfoLog = NULL;

	glGetShaderiv(pStarField->ShaderObject.uiVertexShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(pStarField->ShaderObject.uiVertexShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				GLsizei bytesWritten = 0;
				glGetShaderInfoLog(pStarField->ShaderObject.uiVertexShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Vertex shader compilation Error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				return false;
			}
		}
	}

	// Create fragment shader
	pStarField->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* glchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec4 out_star_color;\n" \
		"uniform sampler2D u_texture0_sampler;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"vec4 textureColor = texture(u_texture0_sampler, gl_PointCoord);\n" \
		"FragColor = out_star_color * textureColor;\n" \
		"}\n";

	glShaderSource(pStarField->ShaderObject.uiFragmentShaderObject, 1, (const GLchar**)&glchFragmentShaderSource, NULL);

	glCompileShader(pStarField->ShaderObject.uiFragmentShaderObject);
	gliInfoLogLength = 0;
	gliShaderComileStatus = 0;
	pszInfoLog = NULL;

	glGetShaderiv(pStarField->ShaderObject.uiFragmentShaderObject, GL_COMPILE_STATUS, &gliShaderComileStatus);
	if (GL_FALSE == gliShaderComileStatus)
	{
		glGetShaderiv(pStarField->ShaderObject.uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetShaderInfoLog(pStarField->ShaderObject.uiFragmentShaderObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Fragment shader compilation error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				return false;
			}
		}
	}

	// Create shader program
	pStarField->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pStarField->ShaderObject.uiShaderProgramObject, pStarField->ShaderObject.uiVertexShaderObject);
	glAttachShader(pStarField->ShaderObject.uiShaderProgramObject, pStarField->ShaderObject.uiFragmentShaderObject);

	glBindAttribLocation(pStarField->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(pStarField->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_COLOR, "vColor");

	glLinkProgram(pStarField->ShaderObject.uiShaderProgramObject);

	GLint gliProgramLinkStatus = 0;
	gliInfoLogLength = 0;
	pszInfoLog = NULL;

	glGetProgramiv(pStarField->ShaderObject.uiShaderProgramObject, GL_LINK_STATUS, &gliProgramLinkStatus);
	if (GL_FALSE == gliProgramLinkStatus)
	{
		glGetProgramiv(pStarField->ShaderObject.uiShaderProgramObject, GL_INFO_LOG_LENGTH, &gliInfoLogLength);
		if (gliInfoLogLength > 0)
		{
			GLsizei bytesWritten = 0;
			pszInfoLog = (char*)malloc((size_t)gliInfoLogLength);
			if (NULL != pszInfoLog)
			{
				glGetProgramInfoLog(pStarField->ShaderObject.uiShaderProgramObject, gliInfoLogLength, &bytesWritten, pszInfoLog);
				fprintf(gpFile, "Shader program link error : %s\n", pszInfoLog);
				free(pszInfoLog);
				pszInfoLog = NULL;
				return false;
			}
		}
	}

	// uniforms
	pStarField->uiMVPUniform = glGetUniformLocation(pStarField->ShaderObject.uiShaderProgramObject, "u_projection_matrix");
	pStarField->uiTextureSamplerUniform = glGetUniformLocation(pStarField->ShaderObject.uiShaderProgramObject, "u_texture0_sampler");
	pStarField->uiTimeUniform = glGetUniformLocation(pStarField->ShaderObject.uiShaderProgramObject, "u_time");

	// load star field texture
	if (0 == loadktx("..//resources//Textures//star.ktx", pStarField->uiStarFieldTexture))
	{
		fprintf(gpFile, "Error while loading texture from star.ktx.\n");
		return false;
	}

	struct star_t
	{
		vmath::vec3 position;
		vmath::vec3 color;
	};

	// initialize VAO
	glGenVertexArrays(1, &pStarField->uiVAO);
	glBindVertexArray(pStarField->uiVAO);

	glGenBuffers(1, &pStarField->uiVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pStarField->uiVBO);
	glBufferData(GL_ARRAY_BUFFER, NUM_STARS * sizeof(struct star_t), NULL, GL_STATIC_DRAW);

	struct star_t* pStar = (struct star_t*)glMapBufferRange(GL_ARRAY_BUFFER, 0, NUM_STARS * sizeof(struct star_t), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

	for (int i = 0; i < NUM_STARS; i++)
	{
		pStar[i].position[0] = (random_float() * 2.0f - 1.0f) * (float)5000;
		pStar[i].position[1] = (random_float() * 2.0f - 1.0f) * (float)5000;
		pStar[i].position[2] = -random_float();

		pStar[i].color[0] = 0.8f + (random_float() * 0.2f);
		pStar[i].color[1] = 0.8f + (random_float() * 0.2f);
		pStar[i].color[2] = 0.8f + (random_float() * 0.2f);
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(struct star_t), NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);

	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(struct star_t), (void*)sizeof(vmath::vec3));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// FBO
	glGenFramebuffers(1, &pStarField->uiFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, pStarField->uiFBO);

	glGenTextures(1, &pStarField->uiStarFieldFBTexture);
	glBindTexture(GL_TEXTURE_2D, pStarField->uiStarFieldFBTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pStarField->uiStarFieldFBTexture, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	pStarField->bIsStarFieldFBOInitialized = true;

	return true;
}
void DrawStarField(STAR_FIELD* pStarField)
{
	float t = (float)gdCurrentTime;

	t = t * 0.03f;
	t = t - floor(t);

	glEnable(GL_CULL_FACE);
	glUseProgram(pStarField->ShaderObject.uiShaderProgramObject);

	glEnable(GL_BLEND);
	glEnable(GL_POINT_SPRITE_ARB);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glBlendFunc(GL_ONE, GL_ONE);

	glUniform1f(pStarField->uiTimeUniform, t);
	glUniformMatrix4fv(pStarField->uiMVPUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, pStarField->uiStarFieldTexture);
	glUniform1i(pStarField->uiTextureSamplerUniform, 0);

	glBindVertexArray(pStarField->uiVAO);
	glDrawArrays(GL_POINTS, 0, NUM_STARS);
	glBindVertexArray(0);
	
	glBindTexture(GL_TEXTURE_2D, 0);
	
	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE_ARB);
	glDisable(GL_BLEND);
	
	glUseProgram(0);
	glDisable(GL_CULL_FACE);
}

void DrawStarFieldToFrameBuffer(STAR_FIELD* pStarField)
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pStarField->uiFBO);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	DrawStarField(pStarField);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void UpdateStarField()
{
	gdCurrentTime += 0.001f;
}

void ResizeStarFieldFBO(STAR_FIELD* pStarField, int iWidth, int iHeight)
{
	if (pStarField->bIsStarFieldFBOInitialized)
	{
		glBindTexture(GL_TEXTURE_2D, pStarField->uiStarFieldFBTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iWidth, iHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glBindFramebuffer(GL_FRAMEBUFFER, pStarField->uiFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pStarField->uiStarFieldFBTexture, 0);
		
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void UnInitializeStarField(STAR_FIELD* pStarField)
{
	if (pStarField->ShaderObject.uiShaderProgramObject)
	{
		if (pStarField->ShaderObject.uiFragmentShaderObject)
		{
			glDetachShader(pStarField->ShaderObject.uiShaderProgramObject, pStarField->ShaderObject.uiFragmentShaderObject);

			glDeleteShader(pStarField->ShaderObject.uiFragmentShaderObject);
			pStarField->ShaderObject.uiFragmentShaderObject = 0;
		}

		if (pStarField->ShaderObject.uiVertexShaderObject)
		{
			glDetachShader(pStarField->ShaderObject.uiShaderProgramObject, pStarField->ShaderObject.uiVertexShaderObject);

			glDeleteShader(pStarField->ShaderObject.uiVertexShaderObject);
			pStarField->ShaderObject.uiVertexShaderObject = 0;
		}

		glDeleteProgram(pStarField->ShaderObject.uiShaderProgramObject);
		pStarField->ShaderObject.uiShaderProgramObject = 0;
	}

	if (pStarField->uiVBO)
	{
		glDeleteBuffers(1, &pStarField->uiVBO);
		pStarField->uiVBO = 0;
	}

	if (pStarField->uiVAO)
	{
		glDeleteBuffers(1, &pStarField->uiVAO);
		pStarField->uiVAO = 0;
	}

	if (pStarField->uiFBO)
	{
		glDeleteBuffers(1, &pStarField->uiFBO);
		pStarField->uiFBO = 0;
	}
}

float random_float(void)
{
	float fRes = 0.0f;
	unsigned int tmp = 0;

	uiSeed *= 16807;

	tmp = uiSeed ^ (uiSeed >> 4) ^ (uiSeed << 15);
	*((unsigned int*)&fRes) = (tmp >> 9) | 0x3F800000;

	return (fRes - 1.0f);
}