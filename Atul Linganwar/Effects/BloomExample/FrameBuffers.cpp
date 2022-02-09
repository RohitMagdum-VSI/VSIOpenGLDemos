#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "FrameBuffers.h"

bool InitializeFrameBuffer(FRAMEBUFFER_OBJECT* pFrameBufferObject)
{
	glGenFramebuffers(1, &pFrameBufferObject->uiFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, pFrameBufferObject->uiFBO);

	glGenTextures(1, &pFrameBufferObject->uiColorRenderBuffer);
	glBindTexture(GL_TEXTURE_2D, pFrameBufferObject->uiColorRenderBuffer);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pFrameBufferObject->uiColorRenderBuffer, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenRenderbuffers(1, &pFrameBufferObject->uiDepthRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, pFrameBufferObject->uiDepthRenderBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, OGL_WINDOW_WIDTH, OGL_WINDOW_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pFrameBufferObject->uiDepthRenderBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	pFrameBufferObject->bIsFBOInitialized = true;

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		return false;
	}

	return true;
}

void ResizeFrameBuffer(int iWidth, int iHeight, FRAMEBUFFER_OBJECT* pFrameBufferObject)
{
	if (pFrameBufferObject->bIsFBOInitialized)
	{
		glBindTexture(GL_TEXTURE_2D, pFrameBufferObject->uiColorRenderBuffer);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iWidth, iHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glBindRenderbuffer(GL_RENDERBUFFER, pFrameBufferObject->uiDepthRenderBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, iWidth, iHeight);

		glBindFramebuffer(GL_FRAMEBUFFER, pFrameBufferObject->uiFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pFrameBufferObject->uiColorRenderBuffer, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pFrameBufferObject->uiDepthRenderBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void FreeFrameBuffer(FRAMEBUFFER_OBJECT* pFrameBufferObject)
{

}