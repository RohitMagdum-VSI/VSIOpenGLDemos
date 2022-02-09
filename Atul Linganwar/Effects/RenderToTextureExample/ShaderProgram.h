#pragma once

enum SHADER_TYPE
{
	COLOR_SHADER = 1,
	TEXTURE_SHADER,
	QUAD_TEXTURE_SHADER,
	NORMAL_SHADER,
};


class ShaderProgram
{
private:
	GLuint m_uiVertexShaderObject;
	GLuint m_uiFragmentShaderObject;
	GLuint m_uiShaderProgramObject;

	GLuint m_uiMVPUniform;
	GLuint m_uiTextureSamplerUniform;

	GLuint m_uiTextureKundali;
	GLuint m_uiTextureStone;

	SHADER_TYPE m_shaderType;

	bool InitializeColorProgram(void);
	void UnInitializeColorProgram(void);

	bool InitializeTextureProgram(void);
	void UnInitializeTextureProgram(void);

	bool InitializeTextureProgramQuad(void);
	void UnInitializeTextureProgramQuad(void);

	int LoadTexture(GLuint* texture, TCHAR imageResourceId[]);

public:
	ShaderProgram(SHADER_TYPE type);
	~ShaderProgram();

	bool Init();
	void DeInit();

	GLuint GetShaderProgramObject() { return m_uiShaderProgramObject; }
	GLuint GetMVPUniform() { return m_uiMVPUniform; }
	GLuint GetTextureSamplerUniform() { return m_uiTextureSamplerUniform; }
	GLuint GetTextureKundali() { return m_uiTextureKundali; }
	GLuint GetTextureStone() { return m_uiTextureStone; }
};