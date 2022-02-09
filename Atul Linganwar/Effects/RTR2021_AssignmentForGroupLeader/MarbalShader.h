#pragma once

typedef struct _MARBAL_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiMVMatrix;
	GLuint uiProjectionMatrixUniform;

	GLuint uiOffsetUniform;

	GLuint uiVeinColorUniform;
	GLuint uiMarbalColorUniform;

	GLuint uiScaleUniform;

	GLuint uiTextureSamplerUniform;

}MARBAL_SHADER;

bool InitializeMarbalShaderProgram(MARBAL_SHADER* pShaderObj);
void UnInitializeMarbalShaderProgram(MARBAL_SHADER* pShaderObj);
