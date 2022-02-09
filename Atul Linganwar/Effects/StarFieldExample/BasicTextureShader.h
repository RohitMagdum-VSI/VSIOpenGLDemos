#pragma once

typedef struct _BASIC_TEXTURE_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiModelMatrixUniform;
	GLuint uiViewMatrixUniform;
	GLuint uiProjectionMatrixUniform;

	GLuint uiTextureSamplerUniform;

}BASIC_TEXTURE_SHADER;

bool InitializeBasicTextureShaderProgram(BASIC_TEXTURE_SHADER* pShaderObj);
void UnInitializeBasicTextureShaderProgram(BASIC_TEXTURE_SHADER* pShaderObj);
