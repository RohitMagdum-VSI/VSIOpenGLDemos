#pragma once

enum VAO_TYPE
{
	VAO_COLOR = 1,
	VAO_TEXTURE,
	VAO_NORMAL,
};

enum SHAPE
{
	QUAD = 1,
	PYRAMID,
	CUBE,
};

class VAO
{
private:
	GLuint m_uiVAO;

	GLuint m_uiVBOPosition;
	GLuint m_uiVBOColor;
	GLuint m_uiVBOTexture;
	GLuint m_uiVBONormal;

	VAO_TYPE m_type;
	SHAPE m_shape;

	void FillVBOPositionData(void);
	void FillVBOColorData(void);
	void FillVBOTextureData(void);
	void FillVBONormalData(void);

public:

	VAO(VAO_TYPE type, SHAPE shape);
	~VAO();

	bool Init();
	void DeInit();

	GLuint GetVAO() { return m_uiVAO; }
};
