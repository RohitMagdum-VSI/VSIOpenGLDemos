#pragma once
#include <windows.h>									// Standard windows header
#include <stdio.h>										// Standard I/O header 
#include <gl\glew.h>
#include <gl\gl.h>										// Header for OpenGL32 library
#include <glm\glm.hpp>
#include "WindowSizeHandler.h"

class FrameBufferUtilities
{
public:
	static GLuint createBuffer();
	static GLuint createTextureAttachment(int width, int height);
	static GLuint createDepthTextureAttachment(int width, int height);
	static GLuint createDepthBufferAttachment(int width, int height);
	static void bindBuffer(GLuint frameBuffer, int width, int height);
	static void unbindCurrentBuffer();
};

