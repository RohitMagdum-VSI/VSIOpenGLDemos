#pragma once

typedef struct _BASIC_PLANET_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiMVPUniform;
	GLuint uiTextureSamplerUniform;

}BASIC_PLANET_SHADER;

bool InitializeBasicPlanetShaderProgram(BASIC_PLANET_SHADER* pShaderObj);
void UnInitializeBasicPlanetShaderProgram(BASIC_PLANET_SHADER* pShaderObj);