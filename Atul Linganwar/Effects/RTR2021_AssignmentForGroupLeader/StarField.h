#pragma once

#define NUM_STARS						10000000

typedef struct _STAR_FIELD
{
	SHADER_OBJECT ShaderObject;

	GLuint uiVAO;
	GLuint uiVBO;

	GLuint uiFBO;
	GLuint uiStarFieldFBTexture;

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
void UpdateStarField();
void ResizeStarFieldFBO(STAR_FIELD* pStarField, int iWidth, int iHeight);

float random_float(void);

