#pragma once

#include <GL/glew.h>
#include <GL/GL.h>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtx\string_cast.hpp>

#include "../Data.h"
#include "../Texture/Texture.h"
#include "../TerrainTexture.h"
#include "../Texture/Tga.h"


// Enable multitexture blending accross terrain
#ifndef ENABLE_MULTITEXTURE
#define ENABLE_MULTITEXTURE 1
#endif // !ENABLE_MULTITEXTURE

// Enable the blend constant based on slope of terrain
#ifndef ENABLE_SLOPE_BASED_BLEND
#define ENABLE_SLOPE_BASED_BLEND 1
#endif

#define TERRAIN_DAY		0
#define TERRAIN_NIGHT	1

class CTerrain
{
private:
	GLuint VAOTerrainMultitextured;
	GLuint VBOTerrainPosition;
	GLuint VBOTerrainColor;
	GLuint VBOTerrainNormal;
	GLuint VBOTerrainTangent;
	GLuint VBOTerrainTexCoord;
	GLuint VBOTerrainIndex;

	GLuint ShaderProgramObjectTerrrain;
	GLuint ShaderProgramObjectTerrrainNight;

	GLuint stoneTextureID;
	GLuint bgTextureID;
	GLuint rTextureID;
	GLuint gTextureID;
	GLuint bTextureID;
	GLuint blendMapTextureID;
	GLuint NormalMapID;

	GLuint Texture0BGUniform;
	GLuint Texture1RUniform;
	GLuint Texture2GUniform;
	GLuint Texture3BUniform;
	GLuint Texture4BlendmapUniform;

	std::vector<glm::vec3> TerrainPositionBuffer;
	std::vector<glm::vec4> TerrainColorBuffer;
	std::vector<glm::vec3> TerrainNormalBuffer;
	std::vector<glm::vec3> TerrainTangentBuffer;
	std::vector<glm::vec2> TerrainTexCoordBuffer;
	std::vector<GLuint> TerrainIndexBuffer;

	GLfloat BlockScale;
	GLfloat HeightScale;

	void GenerateIndexBuffer();
	void GenerateNormals();
	int GetFileLength(std::istream&);
	float GetHeightValue(const unsigned char*, unsigned char);
	float GetPercentage(float, const float, const float);

	CCommon *pCCommon = NULL;
	FILE *TerrainLogFile;

	const char *filepath = "Logs/TerrainLog.txt";

	glm::uvec2	HeightMapDimensions;

	int LoadBMPTexture(TCHAR imageResourceId[], GLuint *Texture);
	TGALoader tgaLoader;

public:
	CTerrain();
	~CTerrain();

	bool LoadHeightMap(const std::string &filename, unsigned char bitsPerPixel, unsigned int width, unsigned int height);
	void InitializeTerrain(GLuint ShaderObjectDay, GLuint ShaderObjectNight);
	//get height of the terrain at position in word space
	float GetHeightAt(const glm::vec3& position);
	void DeleteTerrain(void);
	void RenderTerrain(glm::mat4 Model, glm::mat4 View, glm::mat4 Projection, glm::vec4 clipPlane);
	void RenderTerrainNight(glm::mat4 Model, glm::mat4 View, glm::mat4 Projection, glm::vec4 clipPlane);

	GLuint GetShaderObject(GLuint Time);

	glm::uvec2 GetTerrainWidthHeight();
};