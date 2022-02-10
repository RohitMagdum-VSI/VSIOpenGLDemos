#pragma once

typedef struct _STAR_FIELD
{
	GLuint renderProgram;

	// Planet
	PLANET SphereMap;

	// Uniforms
	GLuint mvpUniform;
	GLuint textureSampler;
	GLuint time;

	// Star Texture
	GLuint starTexture;

	// Vao, vbo
	GLuint vao;
	GLuint vbo;

	// Shader objects
	GLuint vso;
	GLuint fso;

}STAR_FIELD, *P_STAR_FIELD;

BOOL InitStarField(STAR_FIELD &StarField, FLOAT WindowWidth, FLOAT WindowHeight);
void DrawStarField(double currentTime, STAR_FIELD &StarField);
void CleanupStarField(STAR_FIELD &StarField);
