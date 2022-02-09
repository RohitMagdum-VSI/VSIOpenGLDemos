#pragma once

#define NUM_STARS						1000000

typedef struct _STAR_FIELD
{
	SHADER_OBJECT ShaderObject;

	GLuint uiVAO;
	GLuint uiVBO;

	GLuint uiFBO2D;
	GLuint uiStarFieldFB2DTexture;

	GLuint uiFBO3D;
	GLuint uiStarFieldFB3DTexture;
	GLuint uiStarFieldFB3DTextureDepth;

	GLuint uiStarFieldTexture;

	GLuint uiMVPUniform;
	GLuint uiTextureSamplerUniform;
	GLuint uiTimeUniform;

	bool bIsStarFieldFBOInitialized;

}STAR_FIELD;

bool InitializeStarField(STAR_FIELD* pStarField);
void UnInitializeStarField(STAR_FIELD* pStarField);

void DrawStarField(STAR_FIELD* pStarField);
void DrawStarFieldToFrameBuffer(STAR_FIELD* pStarField);
void DrawStarFieldToFrameBuffer3D(STAR_FIELD* pStarField);
void UpdateStarField();
void ResizeStarFieldFBO2D(STAR_FIELD* pStarField, int iWidth, int iHeight);
void ResizeStarFieldFBO3D(STAR_FIELD* pStarField, int iWidth, int iHeight, int iDepth);

float random_float(void);

