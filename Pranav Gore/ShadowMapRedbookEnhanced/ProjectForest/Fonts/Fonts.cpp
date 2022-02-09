#include "Fonts.h"



Fonts::Fonts()
{
}


Fonts::~Fonts()
{
}

void Fonts::LoadFontTexture(const char * fontFileName)
{
	//https://learnopengl.com/In-Practice/Text-Rendering
	//Each of these FreeType functions returns a non-zero integer whenever an error occurred.
	// therefor while error checking just check if ...
	//initialize Fonts load
	FT_Library fontLibrary;
	if (FT_Init_FreeType(&fontLibrary))
		MessageBox(NULL, TEXT("Failed to init Free Type Font Library"), TEXT("ERROR"), MB_OK);	// Display an error message

	//load font as face "Data/Fonts/arialbd.ttf"
	FT_Face face;
	if (FT_New_Face(fontLibrary, fontFileName, 0, &face))
		MessageBox(NULL, TEXT("Failed to load ttf"), TEXT("ERROR"), MB_OK);	// Display an error message

	//define font size
	FT_Set_Pixel_Sizes(face, 0, 25); // 0 : pix width, 48: pixel height

									 // Disable byte-alignment restriction
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Load first 128 characters of ASCII set
	for (GLubyte c = 0; c < 128; c++)
	{
		// Load character glyph 
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			MessageBox(NULL, TEXT("ERROR::FREETYTPE: Failed to load Glyph"), TEXT("ERROR"), MB_OK);	// Display an error message
			continue;
		}
		// Generate texture
		GLuint textureID;
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(
			GL_TEXTURE_2D,//target
			0,//level
			GL_RED,//internal format
			face->glyph->bitmap.width,//width
			face->glyph->bitmap.rows,//height
			0,//border
			GL_RED,//format
			GL_UNSIGNED_BYTE,//type
			face->glyph->bitmap.buffer//pixels
		);
		// Set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// Now store character for later use
		Character character = {
			textureID,
			glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
			glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
			(GLuint)face->glyph->advance.x
		};
		Characters.insert(std::pair<GLchar, Character>(c, character));
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	glBindTexture(GL_TEXTURE_2D, 0);

	// Destroy FreeType at this point we have loaded all 128 char
	FT_Done_Face(face);
	FT_Done_FreeType(fontLibrary);
}

void Fonts::RenderText(GLuint fontShaderID,GLuint fontVBOID,std::string text, GLfloat r, GLfloat g, GLfloat b,GLfloat x, GLfloat y, GLfloat scale)
{
	glUniform3f(glGetUniformLocation(fontShaderID, "textColor_u"),r,g,b);
	// Iterate through all characters
	std::string::const_iterator c;
	for (c = text.begin(); c != text.end(); c++)
	{
		Character ch = Characters[*c];

		GLfloat xpos = x + ch.Bearing.x * scale;
		GLfloat ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

		GLfloat w = ch.Size.x * scale;
		GLfloat h = ch.Size.y * scale;
		// Update VBO for each character
		GLfloat vertices[6][4] = {
			{ xpos,     ypos + h,   0.0, 0.0 },
		{ xpos,     ypos,       0.0, 1.0 },
		{ xpos + w, ypos,       1.0, 1.0 },

		{ xpos,     ypos + h,   0.0, 0.0 },
		{ xpos + w, ypos,       1.0, 1.0 },
		{ xpos + w, ypos + h,   1.0, 0.0 }
		};
		// Render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.TextureID);
		// Update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER,fontVBOID);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// Render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
		x += (ch.Advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
	}
}


