#include "Common.h"

extern mat4 gOrthographicProjectionMatrix;
extern FILE* gpFile = NULL;

BOOLEAN InitializePlanetDetails(P_PLANET_DETAILS pPlanetDetails) 
{
	pPlanetDetails->Shader.vso = glCreateShader(GL_VERTEX_SHADER);
	
	const GLchar* vs =
	{
		"#version 430 core							\n"
		"											\n"
		"in vec4 vPosTexCoord;						\n"
		"out vec2 out_texure0_coord;				\n"
		"											\n"
		"uniform mat4 u_model_matrix;				\n"
		"uniform mat4 u_view_matrix;				\n"
		"uniform mat4 u_projection_matrix;			\n"
		"											\n"
		"void main(void)							\n"
		"{											\n"
		"	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(vPosTexCoord.xy, 0.0, 1.0);	  \n"
		"	out_texture0_coord = vPosTexCoord.zw;	\n"
		"}											\n"
	};

	glShaderSource(pPlanetDetails->Shader.vso, 1, (const GLchar **)&vs, NULL);
	glCompileShader(pPlanetDetails->Shader.vso);
	if (CheckCompileStatus(pPlanetDetails->Shader.vso) == FALSE)
		return FALSE;
	
	pPlanetDetails->Shader.fso = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* fs =
	{
		"#version 430 core				\n"
		"								\n"
		"in vec2 out_texture0_coord;		\n"
		"									\n"
		"uniform sampler2D u_texture0_sampler;	\n"
		"uniform vec4 u_text_color;				\n"
		"										\n"
		"out vec4 FragColor;					\n"
		"										\n"
		"void main(void)						\n"
		"{										\n"
		"	FragColor = vec4(1.0, 1.0, 1.0, texture2D(u_texture_sampler, out_texture0_coord).r) * u_text_color; \n"
		"}										\n"
	};
	glShaderSource(pPlanetDetails->Shader.fso, 1, (const GLchar **)&fs, NULL);
	glCompileShader(pPlanetDetails->Shader.fso);
	if (CheckCompileStatus(pPlanetDetails->Shader.fso) == FALSE)
		return FALSE;

	pPlanetDetails->Shader.renderProgram = glCreateProgram();
	
	glAttachShader(pPlanetDetails->Shader.renderProgram, pPlanetDetails->Shader.vso);
	glAttachShader(pPlanetDetails->Shader.renderProgram, pPlanetDetails->Shader.fso);
	
	glBindAttribLocation(pPlanetDetails->Shader.renderProgram, OGL_ATTRIBUTE_POSITION, "vPosTexCoord");
	glLinkProgram(pPlanetDetails->Shader.renderProgram);
	if (CheckLinkStatus(pPlanetDetails->Shader.renderProgram))
		return FALSE;

	pPlanetDetails->modelMatrixUniform = glGetUniformLocation(pPlanetDetails->Shader.renderProgram, "u_model_matrix");
	pPlanetDetails->viewMatrixUniform = glGetUniformLocation(pPlanetDetails->Shader.renderProgram, "u_view_matrix");
	pPlanetDetails->projectionMatrixUniform = glGetUniformLocation(pPlanetDetails->Shader.renderProgram, "u_projection_matrix");
	pPlanetDetails->textColorUniform = glGetUniformLocation(pPlanetDetails->Shader.renderProgram, "u_text_color");
	pPlanetDetails->textureSamplerUniform = glGetUniformLocation(pPlanetDetails->Shader.renderProgram, "u_texture0_sampler");

	glGenVertexArrays(1, &pPlanetDetails->vaoPlanetDetails);
	glBindVertexArray(pPlanetDetails->vaoPlanetDetails);

	glGenBuffers(1, &pPlanetDetails->vboPlanetDetails);
	glBindBuffer(GL_ARRAY_BUFFER, pPlanetDetails->vboPlanetDetails);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_POSITION);
	glVertexAttribPointer(OGL_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	return TRUE;
}

BOOLEAN InitializeFontTexture(const char *pchFontFileName, FONT_MAP &FontMap) 
{
	FT_Face ftFace;
	FT_Library ftFontLib;
	FT_Error ftError = 0;

	ftError = FT_Init_FreeType(&ftFontLib);
	if (0 != ftError) 
	{
		fprintf(gpFile, "Free type init failed Error : %d\n", (int)ftError);
		return false;
	}
	
	ftError = FT_New_Face(ftFontLib, pchFontFileName, 0, &ftFace);
	if (0 != ftError)
	{
		fprintf(gpFile, "FT_New_Face: failed Error : %d\n", (int)ftError);
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
			fprintf(gpFile, "FT_Load_Char: failed Error : %c: %d\n", ch, (int)ftError);
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

		FONT_DATA FontData =
		{
			uiTextureId,
			vmath::vec2(ftFace->glyph->bitmap.width, ftFace->glyph->bitmap.rows),
			vmath::vec2(ftFace->glyph->bitmap_left, ftFace->glyph->bitmap_top),
			(GLuint)ftFace->glyph->advance.x
		};
		
		FontMap.Fonts.insert(std::pair<GLchar, FONT_DATA>(ch, FontData));
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	
	glBindTexture(GL_TEXTURE_2D, 0);
	
	FT_Done_Face(ftFace);
	FT_Done_FreeType(ftFontLib);

	return true;
}

VOID RenderFont(P_PLANET_DETAILS pPlanetDetails, FONT_MAP &FontMap, const char *pszText, vec4 vec4TextColor, GLfloat x, GLfloat y, GLfloat scale)
{
	glDisable(GL_DEPTH_TEST);
	glUseProgram(pPlanetDetails->Shader.renderProgram);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	mat4 modelMatrix = mat4::identity();
	mat4 viewMatrix = mat4::identity();

	modelMatrix = translate(0.0f, 0.0f, -10.0f);

	glUniformMatrix4fv(pPlanetDetails->modelMatrixUniform, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(pPlanetDetails->viewMatrixUniform, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(pPlanetDetails->projectionMatrixUniform, 1, GL_FALSE, gOrthographicProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glUniform1i(pPlanetDetails->textureSamplerUniform, 0);

	glUniform4fv(pPlanetDetails->textColorUniform, 1, vec4TextColor);
	glBindVertexArray(pPlanetDetails->vaoPlanetDetails);

	const char* pch;
	for (pch = pszText; *pch; pch++) 
	{
		FONT_DATA fontData = FontMap.Fonts[*pch];
		
		GLfloat xpos = x + fontData.vec2Bearing[0] * scale;
		GLfloat ypos = y - (fontData.vec2Size[1] - fontData.vec2Bearing[1]) * scale;
		
		GLfloat w = fontData.vec2Size[0] * scale;
		GLfloat h = fontData.vec2Size[1] * scale;

		GLfloat vertices[6][4] = 
		{
			{xpos,		ypos + h,	0.0, 0.0},
			{xpos,		ypos,		0.0, 1.0},
			{xpos + w,	ypos,		0.0, 1.0},
			
			{xpos,		ypos + h,	0.0, 0.0},
			{xpos + w,	ypos,		0.0, 1.0},
			{xpos + w,	ypos + h,	1.0, 0.0},
		};

		glBindTexture(GL_TEXTURE_2D, fontData.uiTextureId);

		glBindBuffer(GL_ARRAY_BUFFER, pPlanetDetails->vboPlanetDetails);
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
