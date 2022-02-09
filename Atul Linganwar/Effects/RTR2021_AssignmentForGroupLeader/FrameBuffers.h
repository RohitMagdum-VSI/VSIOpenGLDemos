#pragma once

typedef struct _FRAMEBUFFER_OBJECT
{
	GLuint uiFBO;
	GLuint uiColorRenderBuffer;
	GLuint uiDepthRenderBuffer;

	bool bIsFBOInitialized;

}FRAMEBUFFER_OBJECT;

bool InitializeFrameBuffer(FRAMEBUFFER_OBJECT* pFrameBufferObject);
void ResizeFrameBuffer(int iWidth, int iHeight, FRAMEBUFFER_OBJECT* pFrameBufferObject);
void FreeFrameBuffer(FRAMEBUFFER_OBJECT* pFrameBufferObject);
