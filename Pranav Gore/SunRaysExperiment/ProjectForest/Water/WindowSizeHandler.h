#pragma once

#include <windows.h>									// Standard windows header
#include <stdio.h>										// Standard I/O header 
#include <gl\glew.h>
#include <gl\gl.h>										// Header for OpenGL32 library
#include <glm\glm.hpp>


class WindowSizeHandler
{
public:
	static glm::vec2 getFrameBufferSize();
	static void setFrameBufferSize(int width, int height);

private:
	static glm::vec2 framebufferSize;
};

