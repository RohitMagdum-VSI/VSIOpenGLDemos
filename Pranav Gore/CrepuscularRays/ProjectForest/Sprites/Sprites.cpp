#include "Sprites.h"

Sprites::Sprites()
{
	
	pTGA = new TGALoader();

	//sprite_texture = pTGA->LoadTGATexture(filename);
}

Sprites::~Sprites()
{

}

int Sprites::LoadBMPTexture(TCHAR imageResourceId[], GLuint *Texture)
{
	GLfloat Anisotropy = 16.0f;

	// variable declarations
	HBITMAP hBitmap;
	BITMAP bmp;
	int iStatus = FALSE;

	// code
	glGenTextures(1, Texture);// 1 image
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	if (hBitmap)// if bitmap exists ( means hBitmap is not null )
	{
		iStatus = TRUE;
		GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);// set 1 rather than default 4, for better performance
		glBindTexture(GL_TEXTURE_2D, *Texture);// bind texture
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

		if (GLEW_EXT_texture_filter_anisotropic)
		{
			glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &Anisotropy);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, Anisotropy);
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);

		// Create mipmaps for this texture for better image quality
		glGenerateMipmap(GL_TEXTURE_2D);
		DeleteObject(hBitmap);// delete unwanted bitmap handle
	}
	return(iStatus);
}

void Sprites::RenderSprites(GLuint spriteShaderProgramID, GLuint vaoID,glm::mat4 Model, glm::mat4 View, glm::mat4 Projection,GLfloat deltaTime)
{
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);

	glUseProgram(spriteShaderProgramID);

		
	glUniformMatrix4fv(spriteShaderProgramID, 1, GL_FALSE, glm::value_ptr(Projection));
	glBindTexture(GL_TEXTURE_2D, sprite_texture);

	glm::mat4 modelMatrix = glm::translate(glm::mat4(1.0f),glm::vec3(0.0f, 0.0f, -2.0f));
	glm::mat4 translate = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -6));
	GLfloat dt =  fmod(deltaTime, 1.0f);
	glm::mat4 rotate = glm::rotate(glm::mat4(1.0), dt / 6 * 360.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	rotate = glm::rotate(rotate, dt / 8 * 720.0f, glm::vec3(0.0f, 0.0f, 1.0f));
	Model *= rotate * translate * modelMatrix * View;

	glUniformMatrix4fv(glGetUniformLocation(spriteShaderProgramID,"model_matrix"), 1, GL_FALSE, glm::value_ptr(Model));
	// *** bind vao ***
	glBindVertexArray(vaoID);

	// *** draw, either by glDrawTriangles() or glDrawArrays() or glDrawElements()
	glDrawArrays(GL_POINTS, 0, POINT_COUNT); // 3 (each with its x,y,z ) vertices in triangleVertices array

											 // *** unbind vao ***
	glBindVertexArray(0);

	glUseProgram(0);
	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);


}
