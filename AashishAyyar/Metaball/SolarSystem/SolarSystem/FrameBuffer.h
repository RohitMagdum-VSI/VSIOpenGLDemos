#pragma once

typedef struct _FRAME_BUFFER
{
	GLuint fbo;
	GLuint colorTexture;
	GLuint depthTexture;
	BOOLEAN boInitialized;

}FRAME_BUFFER, *P_FRAME_BUFFER;

BOOLEAN InitFrameBuffer(FRAME_BUFFER &FrameBuffer);
void ResizeFrameBuffer(FRAME_BUFFER &FrameBuffer);
void CleanupFrameBuffer(FRAME_BUFFER &FrameBuffer);
