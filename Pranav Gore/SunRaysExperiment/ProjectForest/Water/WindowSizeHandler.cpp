#include "WindowSizeHandler.h"


glm::vec2 WindowSizeHandler::framebufferSize = glm::vec2(800, 600);

glm::vec2 WindowSizeHandler::getFrameBufferSize()
{
	return framebufferSize;
}

void WindowSizeHandler::setFrameBufferSize(int width, int height)
{
	framebufferSize = glm::vec2(width, height);
	glViewport(0, 0, width, height);
}
