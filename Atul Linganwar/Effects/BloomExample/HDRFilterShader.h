#pragma once

typedef struct _HDR_FILTER_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiTextureHDRImageUniform;	

}HDR_FILTER_SHADER;

bool InitializeHDRFilterShaderProgram(HDR_FILTER_SHADER* pShaderObj);
void UnInitializeHDRFilterShaderProgram(HDR_FILTER_SHADER* pShaderObj);

