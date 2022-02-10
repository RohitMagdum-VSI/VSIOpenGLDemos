#include "Common.h"

extern FILE *gpFile;
extern mat4 gPerspectiveProjectionMatrix;

unsigned int seed = 0x13371337;

static BOOL gStarFieldFBOInitialized = FALSE;

void InitSphereMap(STAR_FIELD &StarField);

float random_float() 
{
	float res;
	unsigned int tmp;

	seed *= 16807;

	tmp = seed ^ (seed >> 4) ^ (seed << 15);

	*((unsigned int *)&res) = (tmp >> 9) | 0x3F800000;
	
	return (res - 1.0f);
}

enum 
{
	NUM_STARS = 1000000
};

BOOL InitStarField(STAR_FIELD &StarField, FLOAT WindowWidth, FLOAT WindowHeight)
{
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char *szInfoLog = NULL;

	static const char *fsSource[] =
	{
		"#version 410 core															\n"
		"																			\n"
		"layout (location = 0) out vec4 color;										\n"
		"																			\n"
		"uniform sampler2D tex_star;												\n"
		"flat in vec4 starColor;													\n"
		"																			\n"
		"void main(void)															\n"
		"{																			\n"
		"	vec4 texColor = texture(tex_star, gl_PointCoord);						\n"
		"	if((texColor.r < 0.01) && (texColor.g < 0.01) && (texColor.b < 0.01))	\n"
		"		discard;															\n"
		"	color = starColor * texColor;											\n"
		"}																			\n"
	};
	
	static const char *vsSource[] =
	{
		"#version 410 core														\n"
		"																		\n"
		"layout (location = 0) in vec3 position;								\n"
		"layout (location = 1) in vec3 color;									\n"
		"																		\n"
		"uniform float time;													\n"
		"uniform mat4 proj_matrix;												\n"
		"																		\n"
		"flat out vec4 starColor;												\n"
		"																		\n"
		"void main(void)														\n"
		"{																		\n"
		"	vec4 newVertex = vec4(position, 1.0f);								\n"
		"																		\n"
		"	newVertex.z += time;												\n"
		"	newVertex.z = fract(newVertex.z);									\n"
		"																		\n"
		"	float size = (15.0 * newVertex.z * newVertex.z);					\n"
		"																		\n"
		"	starColor = smoothstep(1.0, 7.0, size) * vec4(color, 1.0f);			\n"
		"																		\n"
		"	newVertex.z = (999.9 * newVertex.z) - 1000.0;						\n"
		"	gl_Position = proj_matrix * newVertex;								\n"
		"	gl_PointSize = size;												\n"
		"}																		\n"
	};

	StarField.vso = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(StarField.vso, 1, vsSource, NULL);
	glCompileShader(StarField.vso);
	if (!CheckCompileStatus(StarField.vso))
	{
		return FALSE;
	}

	StarField.fso = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(StarField.fso, 1, fsSource, NULL);
	glCompileShader(StarField.fso);
	if (!CheckCompileStatus(StarField.fso))
	{
		return FALSE;
	}

	//
	//	Program Object
	//
	StarField.renderProgram = glCreateProgram();

	glAttachShader(StarField.renderProgram, StarField.vso);
	glAttachShader(StarField.renderProgram, StarField.fso);

	glBindAttribLocation(StarField.renderProgram, OGL_ATTRIBUTE_POSITION, "position");
	glBindAttribLocation(StarField.renderProgram, OGL_ATTRIBUTE_COLOR, "color");

	glLinkProgram(StarField.renderProgram);
	if (!CheckLinkStatus(StarField.renderProgram))
	{
		return FALSE;
	}

	StarField.mvpUniform = glGetUniformLocation(StarField.renderProgram, "proj_matrix");
	StarField.textureSampler = glGetUniformLocation(StarField.renderProgram, "tex_star");
	StarField.time = glGetUniformLocation(StarField.renderProgram, "time");

	loadktx("..\\Resources\\Textures\\star.ktx", StarField.starTexture);
	glGenVertexArrays(1, &StarField.vao);
	glBindVertexArray(StarField.vao);

	struct star_t 
	{
		vmath::vec3 position;
		vmath::vec3 color;
	};

	glGenBuffers(1, &StarField.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, StarField.vbo);
	glBufferData(GL_ARRAY_BUFFER, NUM_STARS * sizeof(star_t), NULL, GL_STATIC_DRAW);

	star_t *star = (star_t *)glMapBufferRange(GL_ARRAY_BUFFER, 0, NUM_STARS * sizeof(star_t), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	int i;

	for (i = 0; i < NUM_STARS; i++)
	{
		star[i].position[0] = (random_float() * 2.0f - 1.0f) * (FLOAT)5000;
		star[i].position[1] = (random_float() * 2.0f - 1.0f) * (FLOAT)5000;
		star[i].position[2] = -random_float();
		star[i].color[0] = 0.5f + (random_float() * 0.1f);
		star[i].color[1] = 0.5f + (random_float() * 0.1f);
		star[i].color[2] = 0.5f + (random_float() * 0.1f);
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, sizeof(star_t), NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);

	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(star_t), (void *)sizeof(vmath::vec3));
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	InitSphereMap(StarField);

	return FALSE;
}

void InitSphereMap(STAR_FIELD &StarField)
{
	InitPlanet(StarField.SphereMap, DEFAULT_PLANET_RADIUS, TRUE);
}

void DrawStarField(double currentTime, STAR_FIELD &StarField)
{
	float t = (float)currentTime;

	t *= 0.03f;
	t -= floor(t);

	glEnable(GL_CULL_FACE);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(StarField.renderProgram);	
			glUniform1f(StarField.time, t);
			glUniformMatrix4fv(StarField.mvpUniform, 1, GL_FALSE, gPerspectiveProjectionMatrix * translate(0.0f, 0.0f, -10.0f));
			glEnable(GL_BLEND);
			glEnable(GL_POINT_SPRITE_ARB);
			glEnable(GL_PROGRAM_POINT_SIZE);
				glBlendFunc(GL_ONE, GL_ONE);
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, StarField.starTexture);
					glBindVertexArray(StarField.vao);
						glDrawArrays(GL_POINTS, 0, NUM_STARS);
					glBindVertexArray(0);
				glBindTexture(GL_TEXTURE_2D, 0);
			glDisable(GL_PROGRAM_POINT_SIZE);
			glDisable(GL_POINT_SPRITE_ARB);
			glDisable(GL_BLEND);
		glUseProgram(0);
	glDisable(GL_CULL_FACE);
}

void CleanupStarField(STAR_FIELD &StarField) 
{
	if(StarField.renderProgram && StarField.fso)
		glDetachShader(StarField.renderProgram, StarField.fso);
	if (StarField.renderProgram && StarField.vso)
		glDetachShader(StarField.renderProgram, StarField.vso);
	
	if (StarField.fso)
	{
		glDeleteShader(StarField.fso);
		StarField.fso = 0;
	}

	if (StarField.vso)
	{
		glDeleteShader(StarField.vso);
		StarField.vso = 0;
	}

	if (StarField.renderProgram)
	{
		glDeleteProgram(StarField.renderProgram);
		StarField.renderProgram = 0;
	}

	if (StarField.vbo)
	{
		glDeleteBuffers(1, &StarField.vbo);
		StarField.vbo = 0;
	}

	if (StarField.vao)
	{
		glDeleteVertexArrays(1, &StarField.vao);
		StarField.vao = 0;
	}

	CleanupPlanet(StarField.SphereMap);
}
