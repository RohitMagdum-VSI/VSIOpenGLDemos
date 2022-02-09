#pragma once
typedef struct _HDR_SCENE_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiModelMatrixUniform;
	GLuint uiViewMatrixUniform;
	GLuint uiProjectionMatrixUniform;

	GLuint uiLaUniform;
	GLuint uiLdUniform;
	GLuint uiLsUniform;
	GLuint uiLightPositionUniform;

	GLuint uiKaUniform;
	GLuint uiKdUniform;
	GLuint uiKsUniform;
	GLuint uiMaterialShininessUniform;

	GLuint uiBloomThreshMinUniform;
	GLuint uiBloomThreshMaxUniform;

	GLuint uiTextureSamplerUniform;

}HDR_SCENE_SHADER;

bool InitializeHDRSceneShaderProgram(HDR_SCENE_SHADER* pShaderObj);
void UnInitializeHDRSceneShaderProgram(HDR_SCENE_SHADER* pShaderObj);