#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl/glew.h>
#include<gl/GL.h>

#include "../common/vmath.h"
#include "VAOsPicking.h"
#include "ShaderProgramPicking.h"
#include "PickingExample.h"
#include "DrawPlanets.h"

void DrawSunPicking(ShaderProgram& program, VAO& vao)
{
	glUseProgram(program.GetShaderProgramObject());

	//TRIANGLE
	vmath::mat4 modelViewMatrix = vmath::mat4(1.0f);
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4(1.0f);

	modelViewMatrix = vmath::translate(-2.5f, 2.5f, -10.0f);

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(program.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);
	glUniform1i(program.GetObjectIndexUniform(), (GLint)PLANETS::SUN);

	glBindVertexArray(vao.GetVAO());
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);
}

void DrawMercuryPicking(ShaderProgram& program, VAO& vao)
{
	glUseProgram(program.GetShaderProgramObject());

	//TRIANGLE
	vmath::mat4 modelViewMatrix = vmath::mat4(1.0f);
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4(1.0f);

	modelViewMatrix = vmath::translate(2.5f, 2.5f, -10.0f);

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(program.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);
	glUniform1i(program.GetObjectIndexUniform(), (GLint)PLANETS::MERCURY);

	glBindVertexArray(vao.GetVAO());
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);
}

void DrawVenusPicking(ShaderProgram& program, VAO& vao)
{
	glUseProgram(program.GetShaderProgramObject());

	//TRIANGLE
	vmath::mat4 modelViewMatrix = vmath::mat4(1.0f);
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4(1.0f);

	modelViewMatrix = vmath::translate(2.5f, -2.5f, -10.0f);

	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(program.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);
	glUniform1i(program.GetObjectIndexUniform(), (GLint)PLANETS::VENUS);

	glBindVertexArray(vao.GetVAO());
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

	glUseProgram(0);
}

void DrawSun(ShaderProgram& program, VAO& vao)
{
	glUseProgram(program.GetShaderProgramObject());

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(-2.5f, 2.5f, -10.0f);
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(program.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, program.GetTextureSmiley());
	glUniform1i(program.GetTextureSamplerUniform(), 0);

	glBindVertexArray(vao.GetVAO());

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glBindVertexArray(0);

	glUseProgram(0);
}

void DrawMercury(ShaderProgram& program, VAO& vao)
{
	glUseProgram(program.GetShaderProgramObject());

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(2.5f, 2.5f, -10.0f);
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(program.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, program.GetTextureKundali());
	glUniform1i(program.GetTextureSamplerUniform(), 0);

	glBindVertexArray(vao.GetVAO());

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glBindVertexArray(0);

	glUseProgram(0);
}

void DrawVenus(ShaderProgram& program, VAO& vao)
{
	glUseProgram(program.GetShaderProgramObject());

	vmath::mat4 modelViewMatrix = vmath::mat4::identity();
	vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

	modelViewMatrix = vmath::translate(2.5f, -2.5f, -10.0f);
	modelViewProjectionMatrix = gPerspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(program.GetMVPUniform(), 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, program.GetTextureStone());
	glUniform1i(program.GetTextureSamplerUniform(), 0);

	glBindVertexArray(vao.GetVAO());

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glBindVertexArray(0);

	glUseProgram(0);
}