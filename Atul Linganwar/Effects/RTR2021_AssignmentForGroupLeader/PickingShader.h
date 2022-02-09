#pragma once

typedef struct _PICKING_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiModelMatrixUniform;
	GLuint uiViewMatrixUniform;
	GLuint uiProjectionMatrixUniform;

	GLuint uiObjectIdUniform;

}PICKING_SHADER;

bool InitializePickingShaderProgram(PICKING_SHADER* pShaderObj);
void UnInitializePickingShaderProgram(PICKING_SHADER* pShaderObj);