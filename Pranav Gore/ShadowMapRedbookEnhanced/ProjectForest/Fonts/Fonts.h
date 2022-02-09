#pragma once

#include <windows.h>									// Standard windows header
#include <stdio.h>										// Standard I/O header 
#include <gl\glew.h>
#include <gl\gl.h>										// Header for OpenGL32 library
#include <glm\glm.hpp>

#include <map>

//fonts includes
#include <ft2build.h>
#include FT_FREETYPE_H  


class Fonts
{

	struct Character {
		GLuint TextureID;   // ID handle of the glyph texture
		glm::ivec2 Size;    // Size of glyph
		glm::ivec2 Bearing;  // Offset from baseline to left/top of glyph
		GLuint Advance;    // Horizontal offset to advance to next glyph
	};

	std::map<GLchar, Character> Characters;

public:
	Fonts();
	~Fonts();
	void LoadFontTexture(const char* fontFileName);
	void RenderText(GLuint fontShaderID, GLuint fontVBOID, std::string text, GLfloat r, GLfloat g, GLfloat b, GLfloat x, GLfloat y, GLfloat scale);
};

