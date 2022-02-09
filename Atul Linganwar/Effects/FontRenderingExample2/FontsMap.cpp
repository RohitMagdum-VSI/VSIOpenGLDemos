#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <map>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <freetype/fttypes.h>
#include <freetype/fterrors.h>

#include "../common/vmath.h"
#include "FontsMap.h"

extern FILE* gpFile;

std::map<GLchar, FontData> FontMap;

bool InitializeFontMap(const char* pchFontFileName)
{
	FT_Face ftFace;
	FT_Library ftFontLib;

	FT_Error ftError = 0;

	ftError = FT_Init_FreeType(&ftFontLib);
	if (0 != ftError)
	{
		fprintf(gpFile, "Error while initializing freetype library, Error(%d)\n", (int)ftError);
		return false;
	}

	ftError = FT_New_Face(ftFontLib, pchFontFileName, 0, &ftFace);
	if (0 != ftError)
	{
		fprintf(gpFile, "Error while FT_New_Face(), Error(%d)\n", (int)ftError);
		FT_Done_FreeType(ftFontLib);
		return false;
	}

	FT_Set_Pixel_Sizes(ftFace, 0, 25);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	for (GLubyte ch = 0; ch < 128; ch++)
	{
		ftError = FT_Load_Char(ftFace, ch, FT_LOAD_RENDER);
		if (0 != ftError)
		{
			fprintf(gpFile, "Error while FT_Load_Char(), to load char : %c, Error(%d)\n",ch, (int)ftError);
			FT_Done_Face(ftFace);
			FT_Done_FreeType(ftFontLib);
			return false;
		}

		GLuint uiTextureId = 0;

		glGenTextures(1, &uiTextureId);
		glBindTexture(GL_TEXTURE_2D, uiTextureId);

		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RED,
			ftFace->glyph->bitmap.width,
			ftFace->glyph->bitmap.rows,
			0,
			GL_RED,
			GL_UNSIGNED_BYTE,
			ftFace->glyph->bitmap.buffer
		);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// Fill the font map
		FontData fontData =
		{
			uiTextureId,
			vmath::vec2(ftFace->glyph->bitmap.width, ftFace->glyph->bitmap.rows),
			vmath::vec2(ftFace->glyph->bitmap_left, ftFace->glyph->bitmap_top),
			(GLuint)ftFace->glyph->advance.x
		};

		FontMap.insert(std::pair<GLchar, FontData>(ch, fontData));
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	
	FT_Done_Face(ftFace);
	FT_Done_FreeType(ftFontLib);

	return true;
}

void RenderFont(GLuint uiShaderProgram, GLuint uiVBO, const char* pszText, vmath::vec4 vec4TextColor, GLfloat x, GLfloat y, GLfloat scale)
{
	glUniform4fv(glGetUniformLocation(uiShaderProgram, "u_text_color"), 1, vec4TextColor);
	
	const char* pch;
	for (pch = pszText; *pch; pch++)
	{
		FontData fontData = FontMap[*pch];

		GLfloat xpos = x + fontData.vec2Bearing[0] * scale;
		GLfloat ypos = y - (fontData.vec2Size[1] - fontData.vec2Bearing[1]) * scale;

		GLfloat w = fontData.vec2Size[0] * scale;
		GLfloat h = fontData.vec2Size[1] * scale;

		GLfloat vertices[6][4] =
		{
			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos,     ypos,       0.0, 1.0 },
			{ xpos + w, ypos,       1.0, 1.0 },

			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos + w, ypos,       1.0, 1.0 },
			{ xpos + w, ypos + h,   1.0, 0.0 }
		};

		glBindTexture(GL_TEXTURE_2D, fontData.uiTextureId);
		
		glBindBuffer(GL_ARRAY_BUFFER, uiVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		glDrawArrays(GL_TRIANGLES, 0, 6);
		
		x += (fontData.uiAdvance >> 6) * scale;
	}

	return;
}
