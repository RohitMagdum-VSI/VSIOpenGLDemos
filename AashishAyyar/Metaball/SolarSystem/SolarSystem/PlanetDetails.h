#pragma once

#include <map>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <freetype/fttypes.h>
#include <freetype/fterrors.h>

typedef struct _FONT_DATA 
{
	GLuint uiTextureId;
	vmath::vec2 vec2Size;
	vmath::vec2 vec2Bearing;
	GLuint uiAdvance;

}FONT_DATA, *P_FONT_DATA;

typedef struct _FONT_MAP 
{
	std::map<GLchar, FONT_DATA> Fonts;
}FONT_MAP, *P_FONT_MAP;

typedef struct _PLANET_DETAILS 
{
	SHADER Shader;

	GLuint vaoPlanetDetails;
	GLuint vboPlanetDetails;

	// Uniforms
	GLuint modelMatrixUniform;
	GLuint viewMatrixUniform;
	GLuint projectionMatrixUniform;

	GLuint textColorUniform;
	GLuint textureSamplerUniform;
}PLANET_DETAILS, *P_PLANET_DETAILS;


