#pragma once

typedef struct _SHADER 
{
	GLuint vso;
	GLuint fso;

	GLuint renderProgram;
}SHADER, *P_SHADER;

typedef struct _PLANET_SHADER
{
	SHADER Shader;

	// Uniforms
	GLuint modelMatrixUniform;
	GLuint viewMatrixUniform;
	GLuint projectionMatrixUniform;
	GLuint textureSamplerUniform;

	GLuint laUniform;
	GLuint ldUniform;
	GLuint lsUniform;
	GLuint lightPosUniform;

	GLuint kaUniform;
	GLuint kdUniform;
	GLuint ksUniform;
	GLuint materialShininessUniform;

}PLANET_SHADER, *P_PLANET_SHADER;

typedef struct _COLOR_SHADER 
{
	SHADER Shader;

	//Uniforms
	GLuint mvpUniform;
}COLOR_SHADER, *P_COLOR_SHADER;

typedef struct _PICKING_SHADER 
{
	SHADER Shader;
	
	// Uniforms
	GLuint modelMatrixUniform;
	GLuint viewMatrixUniform;
	GLuint projectionMatrixUniform;
	GLuint objectIDUniform;
}PICKING_SHADER, *P_PICKING_SHADER;

//
//	Shader Utility api's
//
BOOLEAN CheckCompileStatus(GLuint shaderObject);
BOOLEAN CheckLinkStatus(GLuint programObject);
void CleanupShader(P_SHADER pShader);

//
//	Planet Shader
//
BOOL InitPlanetShaders(PLANET_SHADER &Planet);
void CleanupPlanetShader(PLANET_SHADER &pPlanet);

//
//	basic Color shader
//
BOOL InitColorShaders(COLOR_SHADER &ColorShader);
void CleanupColorShader(COLOR_SHADER &ColorShader);

// 
//	Picking Shader
//
BOOL InitPickingShader(PICKING_SHADER &PickingShader);
void CleanupPickingShader(PICKING_SHADER &PickingShader);
