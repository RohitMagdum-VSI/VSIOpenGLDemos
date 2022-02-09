#pragma once

typedef struct _HDR_RESOLVE_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiTextureHDRImageUniform;
	GLuint uiTextureBloomImageUniform;

	GLuint uiExposureUniform;
	GLuint uiBloomFactorUniform;
	GLuint uiSceneFactorUniform;

}HDR_RESOLVE_SHADER;

bool InitializeHDRResolveShaderProgram(HDR_RESOLVE_SHADER* pShaderObj);
void UnInitializeHDRResolveShaderProgram(HDR_RESOLVE_SHADER* pShaderObj);

