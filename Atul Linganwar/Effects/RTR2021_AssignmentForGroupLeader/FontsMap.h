#pragma once
#include <map>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <freetype/fttypes.h>
#include <freetype/fterrors.h>

struct FontData
{
	GLuint uiTextureId;			// Glyph texture id
	vmath::vec2 vec2Size;		// Glyph size
	vmath::vec2 vec2Bearing;	// Offset from origin to left-top of glyph
	GLuint uiAdvance;			// Horizontal offset to the next glyph
};

typedef struct _FONT_MAP
{
	std::map<GLchar, FontData> Fonts;

}FONT_MAP;

typedef struct _FONT_SHADER
{
	SHADER_OBJECT ShaderObject;

	GLuint uiVAO;
	GLuint uiVBO;

	GLuint uiModelMatrixUniform;
	GLuint uiViewMatrixUniform;
	GLuint uiProjectionMatrixUniform;

	GLuint uiTextColorUniform;
	GLuint uiTextureSamplerUniform;

}FONT_SHADER;

bool InitializeFontShaderProgram(FONT_SHADER* pFontShaderObject);
void UnInitializeFontShaderProgram(FONT_SHADER* pFontShaderObject);

bool InitializeFontMap(const char* pchFontFileName, FONT_MAP &FontMap);
void RenderFont(FONT_SHADER*pFontShaderObject, FONT_MAP& FontMap, const char* pszText, vmath::vec4 vec4TextColor, GLfloat x, GLfloat y, GLfloat scale);

//Ref: 
//https://learnopengl.com/In-Practice/Text-Rendering
