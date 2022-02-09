#pragma once

#include <windows.h>									// Standard windows header
#include <stdio.h>										// Standard I/O header 
#include <gl\glew.h>
#include <gl\gl.h>										// Header for OpenGL32 library
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include "../Texture/Tga.h"


//#include "SpritesTexture.h"

#define POINT_COUNT 300

class Sprites {

public:
	Sprites();
	~Sprites();

	static inline float random_float()
	{
		float res;
		unsigned int tmp;
		static unsigned int seed = 0x13371337;

		seed *= 16807;

		tmp = seed ^ (seed >> 4) ^ (seed << 15);

		*((unsigned int *)&res) = (tmp >> 9) | 0x3F800000;

		return (res - 1.0f);
	}

	int LoadBMPTexture(TCHAR imageResourceId[], GLuint *Texture);

	void RenderSprites(GLuint spriteShaderProgramID, GLuint vaoID, glm::mat4 Model, glm::mat4 View, glm::mat4 Projection, GLfloat deltaTime);

private:
	GLuint sprite_texture;
	TGALoader *pTGA;
};
