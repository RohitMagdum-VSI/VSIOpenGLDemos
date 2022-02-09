#pragma once

struct FontData
{
	GLuint uiTextureId;			// Glyph texture id
	vmath::vec2 vec2Size;		// Glyph size
	vmath::vec2 vec2Bearing;	// Offset from origin to left-top of glyph
	GLuint uiAdvance;			// Horizontal offset to the next glyph
};

bool InitializeFontMap(const char* pchFontFileName);
void RenderFont(GLuint uiShaderProgram, GLuint uiVBO, const char* pszText, vmath::vec4 vec4TextColor, GLfloat x, GLfloat y, GLfloat scale);


//Ref: 
//https://learnopengl.com/In-Practice/Text-Rendering
