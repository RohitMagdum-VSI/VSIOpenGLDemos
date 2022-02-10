#include "Common.h"

FRAME_BUFFER gFrameBuffer;

extern FLOAT gWindowWidth;
extern FLOAT gWindowHeight;

BOOLEAN InitFrameBuffer(FRAME_BUFFER &FrameBuffer) 
{
	//
	//	Framebuffer
	//
	glGenFramebuffers(1, &FrameBuffer.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, FrameBuffer.fbo);
		
		glGenTextures(1, &FrameBuffer.colorTexture);
		glBindTexture(GL_TEXTURE_2D, FrameBuffer.colorTexture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gWindowWidth, gWindowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FrameBuffer.colorTexture, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenRenderbuffers(1, &FrameBuffer.depthTexture);
		glBindRenderbuffer(GL_RENDERBUFFER, FrameBuffer.depthTexture);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, gWindowWidth, gWindowHeight);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, FrameBuffer.depthTexture);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	FrameBuffer.boInitialized = TRUE;

	return TRUE;
}

void ResizeFrameBuffer(FRAME_BUFFER &FrameBuffer)
{
	if (FrameBuffer.boInitialized) 
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FrameBuffer.fbo);
			
			glBindTexture(GL_TEXTURE_2D, FrameBuffer.colorTexture);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gWindowWidth, gWindowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FrameBuffer.colorTexture, 0);
			glBindTexture(GL_TEXTURE_2D, 0);

			glBindRenderbuffer(GL_RENDERBUFFER, FrameBuffer.depthTexture);
				glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, gWindowWidth, gWindowHeight);
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, FrameBuffer.depthTexture);
			glBindRenderbuffer(GL_RENDERBUFFER, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);	
	}
}

void CleanupFrameBuffer(FRAME_BUFFER &FrameBuffer) 
{
	if (FrameBuffer.boInitialized)
	{
		glDeleteFramebuffers(1, &FrameBuffer.fbo);
		FrameBuffer.fbo = 0;
	}
}
