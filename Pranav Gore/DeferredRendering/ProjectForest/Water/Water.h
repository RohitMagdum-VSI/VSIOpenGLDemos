#pragma once
#include <windows.h>									// Standard windows header
#include <stdio.h>										// Standard I/O header 
#include <gl\glew.h>
#include <gl\gl.h>										// Header for OpenGL32 library
#include <glm\glm.hpp>
#include "glm\gtc\matrix_transform.hpp"
#include <glm\gtc\type_ptr.hpp>

#include "FrameBufferUtilities.h"
#include "../Terrain/Terrain.h"
#include "../Camera/Player.h"





class Water
{
public:
	Water(CTerrain*,CPlayer*);
	~Water();

	void cleanUp();
	void bindReflectionBuffer();
	void bindRefractionBuffer();

	glm::vec4 getReflectionClipPlane();
	glm::vec4 getRefractionClipPlane();

	int LoadBMPTexture(TCHAR imageResourceId[], GLuint *Texture);
	void RenderWater(GLuint shaderID, GLuint vaoID, GLuint indicesID,glm::mat4 ModelMatrix, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix, GLfloat deltaTime);
	
private:
	CTerrain *pTerrain;
	CPlayer *pPlayer;

	const int SIZE = 1000;
	const int REFLECTION_WIDTH = 1280;
	const int REFLECTION_HEIGHT = 720;
	const int REFRACTION_WIDTH = 1280;
	const int REFRACTION_HEIGHT = 720;

	static constexpr GLfloat WAVE_SPEED = 0.02f;
	GLfloat moveFactor = 0;

	GLuint reflectionFrameBuffer;
	GLuint reflectionTexture;
	GLuint reflectionDepthBuffer;

	GLuint refractionFrameBuffer;
	GLuint refractionTexture;
	GLuint refractionDepthTexture;

	GLuint dudvTexture;
	GLuint normalTexture;

	glm::vec4 reflection_clip_plane;
	glm::vec4 refraction_clip_plane;

};

