#include "Water.h"
#include "WaterTexture.h"


Water::Water(CTerrain*pTerrain, CPlayer*pPlayer)
{
	
	this->pTerrain = pTerrain;
	this->pPlayer = pPlayer;

	LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_DUDV_WATER), &dudvTexture);
	LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_NORMAL_WATER), &normalTexture);

	glm::vec2 size = WindowSizeHandler::getFrameBufferSize();

	reflectionFrameBuffer = FrameBufferUtilities::createBuffer();
	reflectionTexture = FrameBufferUtilities::createTextureAttachment(REFLECTION_WIDTH, REFLECTION_HEIGHT);
	reflectionDepthBuffer = FrameBufferUtilities::createDepthBufferAttachment(REFLECTION_WIDTH, REFLECTION_HEIGHT);
	
	FrameBufferUtilities::unbindCurrentBuffer();

	refractionFrameBuffer = FrameBufferUtilities::createBuffer();
	refractionTexture = FrameBufferUtilities::createTextureAttachment(REFRACTION_WIDTH, REFRACTION_HEIGHT);
	refractionDepthTexture = FrameBufferUtilities::createDepthTextureAttachment(REFRACTION_WIDTH, REFRACTION_HEIGHT);
	FrameBufferUtilities::unbindCurrentBuffer();

	double waterheight = 0;
	reflection_clip_plane = glm::vec4{ 0, 1, 0, -waterheight + 0.05 };
	refraction_clip_plane = glm::vec4{ 0, 8, 0, -waterheight };
}


Water::~Water()
{
	cleanUp();
}


void Water::cleanUp()
{
	glDeleteFramebuffers(1, &reflectionFrameBuffer);
	glDeleteTextures(1, &reflectionTexture);
	glDeleteRenderbuffers(1, &reflectionDepthBuffer);
	glDeleteFramebuffers(1, &refractionFrameBuffer);
	glDeleteTextures(1, &refractionTexture);
	glDeleteTextures(1, &refractionDepthTexture);
	
}

void Water::bindReflectionBuffer()
{
	FrameBufferUtilities::bindBuffer(reflectionFrameBuffer, REFLECTION_WIDTH, REFLECTION_HEIGHT);
}

void Water::bindRefractionBuffer()
{
	FrameBufferUtilities::bindBuffer(refractionFrameBuffer, REFRACTION_WIDTH, REFRACTION_HEIGHT);
}



glm::vec4 Water::getReflectionClipPlane()
{
	return reflection_clip_plane;
}

glm::vec4 Water::getRefractionClipPlane()
{
	return refraction_clip_plane;
}

int Water::LoadBMPTexture(TCHAR imageResourceId[], GLuint *Texture)
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

void Water::RenderWater(GLuint ShaderProgramObjectWater, GLuint vaoID, GLuint indicesID, glm::mat4 ModelMatrix, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix,GLfloat deltaTime)
{	
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(pTerrain->GetTerrainWidthHeight().x , 1, pTerrain->GetTerrainWidthHeight().y));
	//glm::vec3 playerPos = pPlayer->GetPlayerPosition();
	//glm::mat4 translate = glm::translate(glm::mat4(1.0f), glm::vec3(1, WATER_HEIGHT, 1));
	
	//ModelMatrix *= translate * scale;
	ModelMatrix *= scale;
	//ModelMatrix *=  scale;

	glUseProgram(ShaderProgramObjectWater);

	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectWater, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectWater, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectWater, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(ProjectionMatrix));

	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "lightPosition[0]"),300000.0f, 100000.0f, -300000.0f);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "lightColor[0]"), 0.5, 0.5, 0.5);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "lightAttenuation[0]"), 1,0,0);

	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "lightPosition[1]"), 300000.0f, 500000.0f, -300000.0f);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "lightColor[1]"), 0.5, 0.5, 0.5);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "lightAttenuation[1]"), 1, 0, 0);

	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "reflectionTexture"), 0);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "refractionTexture"), 1);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "depthTexture"), 2);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "dudvTexture"), 3);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "normalTexture"), 4);
	moveFactor += WAVE_SPEED * deltaTime;
	moveFactor = fmod(moveFactor, 1.0);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectWater, "time"), moveFactor);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "fogColor"), 0.45, 0.60, 0.70);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectWater, "fogDencity"), 0.004);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectWater, "gamma"), 1.3f);
	/*
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectWater, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectWater, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectWater, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(ProjectionMatrix));

	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "u_cameraPosition"), pPlayer->GetPlayerPosition().x, pPlayer->GetPlayerPosition().y, pPlayer->GetPlayerPosition().z);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "u_lightPosition"), 0, 3000, -5000);
	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "u_lightColor"), 1.0f, 1.0f, 1.0f);

	glUniform3f(glGetUniformLocation(ShaderProgramObjectWater, "u_skyColor"), 0.5444f, 0.62f, 0.69f);


	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "u_texture0_reflection_sampler"), 0);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "u_texture1_refraction_sampler"), 1);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "u_texture2_refractionDepth_sampler"), 2);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "u_texture3_dudv_sampler"), 3);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectWater, "u_texture4_normal_sampler"), 4);

	moveFactor += WAVE_SPEED * deltaTime;
	moveFactor = fmod(moveFactor, 1.0);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectWater, "u_moveFactor"), moveFactor);
	*/
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBindVertexArray(vaoID);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, reflectionTexture);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, refractionTexture);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, refractionDepthTexture);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, dudvTexture);
	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, normalTexture);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesID);
	glDrawElements(GL_TRIANGLES,6, GL_UNSIGNED_INT, 0);

	glBindVertexArray(0);

	glDisable(GL_BLEND);

	glUseProgram(0);
}
