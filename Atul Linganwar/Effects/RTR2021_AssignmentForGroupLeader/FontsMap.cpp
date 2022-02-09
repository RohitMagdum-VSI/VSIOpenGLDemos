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
#include "../common/Common.h"
#include "FontsMap.h"

extern FILE* gpFile;
extern GLfloat gWindowWidth;
extern GLfloat gWindowHeight;
extern vmath::mat4 gOrthographicProjectionMatrix;
extern vmath::mat4 gPerspectiveProjectionMatrix;

bool InitializeFontShaderProgram(FONT_SHADER* pFontShaderObject)
{
	pFontShaderObject->ShaderObject.uiVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* pchVertexShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec4 vPosTexCoord;\n" \
		"out vec2 out_texture0_coord;\n" \
		"uniform mat4 u_model_matrix;\n" \
		"uniform mat4 u_view_matrix;\n" \
		"uniform mat4 u_projection_matrix;\n" \
		"void main(void)\n" \
		"{\n" \
		"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(vPosTexCoord.xy, 0.0, 1.0);\n" \
		"out_texture0_coord = vPosTexCoord.zw;\n" \
		"}\n";

	glShaderSource(pFontShaderObject->ShaderObject.uiVertexShaderObject, 1, (const GLchar**)&pchVertexShaderSource, NULL);

	glCompileShader(pFontShaderObject->ShaderObject.uiVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompiledStatus = 0;
	char* szInfoLog = NULL;

	glGetShaderiv(pFontShaderObject->ShaderObject.uiVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(pFontShaderObject->ShaderObject.uiVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(pFontShaderObject->ShaderObject.uiVertexShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fonts VertexShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pFontShaderObject->ShaderObject.uiFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* pchFragmentShaderSource =
		"#version 430 core\n" \
		"\n" \
		"in vec2 out_texture0_coord;\n" \
		"uniform sampler2D u_texture0_sampler;\n" \
		"uniform vec4 u_text_color;\n" \
		"out vec4 FragColor;\n" \
		"void main(void)\n" \
		"{\n" \
		"FragColor = vec4(1.0, 1.0, 1.0, texture2D(u_texture0_sampler, out_texture0_coord).r) * u_text_color;\n" \
		"}\n";

	glShaderSource(pFontShaderObject->ShaderObject.uiFragmentShaderObject, 1, (const GLchar**)&pchFragmentShaderSource, NULL);

	glCompileShader(pFontShaderObject->ShaderObject.uiFragmentShaderObject);
	glGetShaderiv(pFontShaderObject->ShaderObject.uiFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompiledStatus);
	if (iShaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(pFontShaderObject->ShaderObject.uiFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(pFontShaderObject->ShaderObject.uiFragmentShaderObject, iInfoLogLength, &written, szInfoLog);
				fprintf(gpFile, "Fonts FragmentShader compilation log: %s\n", szInfoLog);
				free(szInfoLog);
				return false;
			}
		}
	}

	pFontShaderObject->ShaderObject.uiShaderProgramObject = glCreateProgram();

	glAttachShader(pFontShaderObject->ShaderObject.uiShaderProgramObject, pFontShaderObject->ShaderObject.uiVertexShaderObject);
	glAttachShader(pFontShaderObject->ShaderObject.uiShaderProgramObject, pFontShaderObject->ShaderObject.uiFragmentShaderObject);

	glBindAttribLocation(pFontShaderObject->ShaderObject.uiShaderProgramObject, OGL_ATTRIBUTE_POSITION, "vPosTexCoord");

	glLinkProgram(pFontShaderObject->ShaderObject.uiShaderProgramObject);
	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(pFontShaderObject->ShaderObject.uiShaderProgramObject, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(pFontShaderObject->ShaderObject.uiShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			GLsizei written;
			glGetProgramInfoLog(pFontShaderObject->ShaderObject.uiShaderProgramObject, iInfoLogLength, &written, szInfoLog);
			fprintf(gpFile, "Basic Color Shader Program Link log: %s\n", szInfoLog);
			free(szInfoLog);
			return false;
		}
	}

	pFontShaderObject->uiModelMatrixUniform = glGetUniformLocation(pFontShaderObject->ShaderObject.uiShaderProgramObject, "u_model_matrix");
	pFontShaderObject->uiViewMatrixUniform = glGetUniformLocation(pFontShaderObject->ShaderObject.uiShaderProgramObject, "u_view_matrix");
	pFontShaderObject->uiProjectionMatrixUniform = glGetUniformLocation(pFontShaderObject->ShaderObject.uiShaderProgramObject, "u_projection_matrix");
	pFontShaderObject->uiTextColorUniform = glGetUniformLocation(pFontShaderObject->ShaderObject.uiShaderProgramObject, "u_text_color");
	pFontShaderObject->uiTextureSamplerUniform = glGetUniformLocation(pFontShaderObject->ShaderObject.uiShaderProgramObject, "u_texture0_sampler");

	//
	// VAO
	//
	glGenVertexArrays(1, &pFontShaderObject->uiVAO);
	glBindVertexArray(pFontShaderObject->uiVAO);

	glGenBuffers(1, &pFontShaderObject->uiVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pFontShaderObject->uiVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	return true;
}

bool InitializeFontMap(const char* pchFontFileName, FONT_MAP& FontMap)
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

	FT_Set_Pixel_Sizes(ftFace, 0, 64);

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

		FontMap.Fonts.insert(std::pair<GLchar, FontData>(ch, fontData));
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	
	FT_Done_Face(ftFace);
	FT_Done_FreeType(ftFontLib);

	return true;
}

void RenderFont(FONT_SHADER* pFontShaderObject, FONT_MAP& FontMap, const char* pszText, vmath::vec4 vec4TextColor, GLfloat x, GLfloat y, GLfloat scale)
{
	glDisable(GL_DEPTH_TEST);
	glUseProgram(pFontShaderObject->ShaderObject.uiShaderProgramObject);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	vmath::mat4 ModelMatrix = vmath::mat4::identity();
	vmath::mat4 ViewMatrix = vmath::mat4::identity();

	ModelMatrix = vmath::translate(0.0f, 0.0f, -10.0f);

	glUniformMatrix4fv(pFontShaderObject->uiModelMatrixUniform, 1, GL_FALSE, ModelMatrix);
	glUniformMatrix4fv(pFontShaderObject->uiViewMatrixUniform, 1, GL_FALSE, ViewMatrix);
	glUniformMatrix4fv(pFontShaderObject->uiProjectionMatrixUniform, 1, GL_FALSE, gOrthographicProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glUniform1i(pFontShaderObject->uiTextureSamplerUniform, 0);

	glUniform4fv(pFontShaderObject->uiTextColorUniform, 1, vec4TextColor);
	
	glBindVertexArray(pFontShaderObject->uiVAO);

	const char* pch;
	for (pch = pszText; *pch; pch++)
	{
		FontData fontData = FontMap.Fonts[*pch];

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
		
		glBindBuffer(GL_ARRAY_BUFFER, pFontShaderObject->uiVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		glDrawArrays(GL_TRIANGLES, 0, 6);
		
		x += (fontData.uiAdvance >> 6) * scale;
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindVertexArray(0);

	glDisable(GL_BLEND);

	glUseProgram(0);
	glEnable(GL_DEPTH_TEST);

	return;
}

void UnInitializeFontShaderProgram(FONT_SHADER* pFontShaderObject)
{
	glDetachShader(pFontShaderObject->ShaderObject.uiShaderProgramObject, pFontShaderObject->ShaderObject.uiVertexShaderObject);
	glDetachShader(pFontShaderObject->ShaderObject.uiShaderProgramObject, pFontShaderObject->ShaderObject.uiFragmentShaderObject);

	glDeleteShader(pFontShaderObject->ShaderObject.uiVertexShaderObject);
	pFontShaderObject->ShaderObject.uiVertexShaderObject = 0;

	glDeleteShader(pFontShaderObject->ShaderObject.uiFragmentShaderObject);
	pFontShaderObject->ShaderObject.uiFragmentShaderObject = 0;

	glDeleteProgram(pFontShaderObject->ShaderObject.uiShaderProgramObject);
	pFontShaderObject->ShaderObject.uiShaderProgramObject = 0;
}
