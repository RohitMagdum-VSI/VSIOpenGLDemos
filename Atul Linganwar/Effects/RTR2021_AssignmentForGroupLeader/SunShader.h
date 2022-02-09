#pragma once

typedef struct _SUN_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiMVMatrix;
	GLuint uiProjectionMatrixUniform;

	GLuint uiOffsetUniform;
	
	GLuint uiColor1Uniform;
	GLuint uiColor2Uniform;

	GLuint uiScaleUniform;

	GLuint uiTextureSamplerUniform;

}SUN_SHADER;

bool InitializeSunShaderProgram(SUN_SHADER* pShaderObj);
void UnInitializeSunShaderProgram(SUN_SHADER* pShaderObj);
