#pragma once
typedef struct _BASIC_QUAD_RTT_TEXTURE_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiModelMatrixUniform;
	GLuint uiViewMatrixUniform;
	GLuint uiProjectionMatrixUniform;

	GLuint uiTextureSamplerUniform;

}BASIC_QUAD_RTT_TEXTURE_SHADER;

bool InitializeBasicQuadRTTTextureShaderProgram(BASIC_QUAD_RTT_TEXTURE_SHADER* pShaderObj);
void UnInitializeBasicQuadRTTTextureShaderProgram(BASIC_QUAD_RTT_TEXTURE_SHADER* pShaderObj);