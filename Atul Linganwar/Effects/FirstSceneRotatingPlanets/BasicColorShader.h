#pragma once

typedef struct _BASIC_COLOR_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiMVPUniform;

}BASIC_COLOR_SHADER;

bool InitializeBasicColorShaderProgram(BASIC_COLOR_SHADER* pShaderObj);
void UnInitializeBasicColorShaderProgram(BASIC_COLOR_SHADER* pShaderObj);