#pragma once

#include <GL/glew.h>
#include <GL/GL.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm/gtx/normal.hpp>
#include <glm\gtc\random.hpp>

#include "../Data.h"
#include "../Camera/Player.h"
#include "../Terrain/Terrain.h"

#define MAX_GRASS_BLADES	50
#define GRASS_RADIUS		5.0f

#define GRASS_DAY		0
#define GRASS_NIGHT		1

class CGrass
{
private:
	GLuint ShaderProgramObjectGrass;
	GLuint ShaderProgramObjectGrassNight;

	GLuint VAOGrass;
	GLuint VBOGrassPosition;
	GLuint VBOGrassColor;
	GLuint VBOBrightness;
	GLuint VBOGrassNormal;
	GLuint VBOModelMatrices;

	GLboolean bIsGrassDrawn;

	GLuint ModelMatrixBuffer;
	GLfloat WaveValue;

	GLfloat GLM_RAND_FIX;

	GLfloat GrassMinHeight;
	GLfloat GrassMaxHeight;

	GLfloat GrassMinBrightness;
	GLfloat GrassMaxBrightness;
	GLfloat BrightnessRange;

	GLfloat PerlinFactor;

	glm::vec3 RandomPosition;
	glm::mat4 Temp;

	glm::vec4 *ModelMatrices;
	glm::vec4 *ModelMatricesUpdateChunk;
	GLint UpdateChunkIndex;

	GLfloat *BrightnessValues;
	GLfloat *BrightnessValuesPtr;

	glm::vec4 *SidePtr;
	glm::vec4 *UpPtr;
	glm::vec4 *ForwardPtr;
	glm::vec4 *PosPtr;

	GLint NumBladesPerUpdate;
	GLint ResetPointerSteps;

	float SimplePerlin(int x, int y);

	CPlayer *pCPlayer;
	CTerrain *pCTerrain;

public:
	CGrass();
	~CGrass();

	void PlaceGrass(CPlayer *pCPlayer, GLuint ShaderObject);
	void InitGrass(CPlayer *pCPlayer, CTerrain *pCTerrain, GLuint ShaderObject, GLuint ShaderObjectNight);
	void RenderGrass(glm::mat4 ModelMatrix, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix);
	void RenderGrassNight(glm::mat4 ModelMatrix, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix);
	void ControlGrassWaving(GLfloat fDelta);

	GLuint GetShaderObject(int type);

	void UpdateGrassLoop(void);
	void DeletGrass(void);
};