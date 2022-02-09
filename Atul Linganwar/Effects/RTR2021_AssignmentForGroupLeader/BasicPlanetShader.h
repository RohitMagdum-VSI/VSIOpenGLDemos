#pragma once

typedef struct _BASIC_PLANET_SHADER
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

	GLuint uiTextureSamplerUniform;

}BASIC_PLANET_SHADER;

bool InitializeBasicPlanetShaderProgram(BASIC_PLANET_SHADER* pShaderObj);
void UnInitializeBasicPlanetShaderProgram(BASIC_PLANET_SHADER* pShaderObj);
