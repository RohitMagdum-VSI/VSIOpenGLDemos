#pragma once

typedef struct _BASIC_3DTEXTURE_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiModelMatrixUniform;
	GLuint uiViewMatrixUniform;
	GLuint uiProjectionMatrixUniform;

	GLuint uiTextureSamplerUniform;
	GLuint uiTextureSamplerUniform2D;

}BASIC_3DTEXTURE_SHADER;

bool InitializeBasic3DTextureShaderProgram(BASIC_3DTEXTURE_SHADER* pShaderObj);
void UnInitializeBasic3DTextureShaderProgram(BASIC_3DTEXTURE_SHADER* pShaderObj);
