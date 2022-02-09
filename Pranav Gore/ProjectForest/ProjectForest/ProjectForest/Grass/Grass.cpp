#include "Grass.h"

CGrass::CGrass()
{
	ModelMatrices = new glm::vec4[MAX_GRASS_BLADES * 4];
	ModelMatricesUpdateChunk = &ModelMatrices[MAX_GRASS_BLADES * 4];

	bIsGrassDrawn = GL_FALSE;

	WaveValue = 0.0f;

	GLM_RAND_FIX = 0.6f;

	GrassMinHeight = 0.4f;
	GrassMaxHeight = 2.0f;

	GrassMinBrightness = 0.5f;
	GrassMaxBrightness = 0.9f;
	BrightnessRange = GrassMaxBrightness - GrassMinBrightness;

	BrightnessValues = new GLfloat[MAX_GRASS_BLADES];
	BrightnessValuesPtr = BrightnessValues;

	SidePtr = &ModelMatrices[MAX_GRASS_BLADES * 0];
	UpPtr = &ModelMatrices[MAX_GRASS_BLADES * 1];
	ForwardPtr = &ModelMatrices[MAX_GRASS_BLADES * 2];
	PosPtr = &ModelMatrices[MAX_GRASS_BLADES * 3];

	pCPlayer = NULL;
	pCTerrain = NULL;

	NumBladesPerUpdate = 2000;
	ResetPointerSteps = MAX_GRASS_BLADES / NumBladesPerUpdate;
}

CGrass::~CGrass()
{

}

static void *InvokeWrapLoop(void *arg)
{

}

void BeginWrapThread(void)
{

}

void CGrass::InitGrass(CPlayer *pCPlayer, CTerrain *pCTerrain, GLuint ShaderObject, GLuint ShaderObjectNight)
{
	this->pCPlayer = pCPlayer;
	this->pCTerrain = pCTerrain;
	this->ShaderProgramObjectGrass = ShaderObject;
	this->ShaderProgramObjectGrassNight = ShaderObjectNight;

	WaveValue = 0.0f;

	int iNumVertices = 3;
	glm::vec3 Vertices[] = {
		glm::vec3(-0.01f, 0.0f, 0.0f),
		glm::vec3(0.01f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.09f)
	};

	glm::vec4 Colors[] = {
		glm::vec4(0.25f, 0.37f, 0.12f, 1.33f) * 1.125f,
		glm::vec4(0.15f, 0.35f, 0.09f, 1.33f) * 1.125f,
		glm::vec4(0.45f, 0.65f, 0.22f, 1.33f) * 1.375f
	};

	glm::vec3 GrassNormal = glm::triangleNormal(Vertices[0], Vertices[1], Vertices[2]);

	//glm::vec3 Normals[] = {
	//	GrassNormal,
	//	GrassNormal,
	//	GrassNormal
	//};

	glm::vec3 Normals[] = {
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
	};

	glGenVertexArrays(1, &VAOGrass);
	glBindVertexArray(VAOGrass);

	// for grass position
	glGenBuffers(1, &VBOGrassPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOGrassPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * iNumVertices, Vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// for grass color
	glGenBuffers(1, &VBOGrassColor);
	glBindBuffer(GL_ARRAY_BUFFER, VBOGrassColor);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * iNumVertices, Colors, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// for grass normal
	glGenBuffers(1, &VBOGrassNormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBOGrassNormal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * iNumVertices, Normals, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// for grass brightness
	glGenBuffers(1, &VBOBrightness);
	glBindBuffer(GL_ARRAY_BUFFER, VBOBrightness);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * MAX_GRASS_BLADES, NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_BRIGHTNESS, 1, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_BRIGHTNESS);
	glVertexAttribDivisor(OGL_ATTRIBUTE_BRIGHTNESS, 1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// for grass instanced matrices
	//glGenBuffers(1, &gVBOModelMatrices);
	//glBindBuffer(GL_ARRAY_BUFFER, gVBOModelMatrices);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * MAX_GRASS_BLADES * 4, NULL, GL_DYNAMIC_DRAW);

	/*glEnableVertexAttribArray(OGL_ATTRIBUTE_INSTANCE_MATRIX);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_INSTANCE_MATRIX + 1);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_INSTANCE_MATRIX + 2);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_INSTANCE_MATRIX + 3);


	glVertexAttribPointer(OGL_ATTRIBUTE_INSTANCE_MATRIX, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(sizeof(glm::vec4) * MAX_GRASS_BLADES * 0));
	glVertexAttribPointer(OGL_ATTRIBUTE_INSTANCE_MATRIX + 1, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(sizeof(glm::vec4) * MAX_GRASS_BLADES * 1));
	glVertexAttribPointer(OGL_ATTRIBUTE_INSTANCE_MATRIX + 2, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(sizeof(glm::vec4) * MAX_GRASS_BLADES * 2));
	glVertexAttribPointer(OGL_ATTRIBUTE_INSTANCE_MATRIX + 3, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(sizeof(glm::vec4) * MAX_GRASS_BLADES * 3));

	glVertexAttribDivisor(OGL_ATTRIBUTE_INSTANCE_MATRIX, 1);
	glVertexAttribDivisor(OGL_ATTRIBUTE_INSTANCE_MATRIX + 1, 1);
	glVertexAttribDivisor(OGL_ATTRIBUTE_INSTANCE_MATRIX + 2, 1);
	glVertexAttribDivisor(OGL_ATTRIBUTE_INSTANCE_MATRIX + 3, 1);*/

	glGenBuffers(1, &ModelMatrixBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * MAX_GRASS_BLADES, NULL, GL_DYNAMIC_DRAW);

	for (int i = 0; i < 4; i++)
	{
		glVertexAttribPointer(OGL_ATTRIBUTE_INSTANCE_MATRIX + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void *)(sizeof(glm::vec4) * i));
		glEnableVertexAttribArray(OGL_ATTRIBUTE_INSTANCE_MATRIX + i);
		glVertexAttribDivisor(OGL_ATTRIBUTE_INSTANCE_MATRIX + i, 1);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);



	if (bIsGrassDrawn == GL_FALSE)
	{
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		glm::mat4 *Matrices = (glm::mat4 *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (int i = 0; i < MAX_GRASS_BLADES; i++)
		{
			RandomPosition = glm::vec3(glm::linearRand(-GRASS_RADIUS, GRASS_RADIUS + GLM_RAND_FIX), 0.0f, glm::linearRand(-GRASS_RADIUS, GRASS_RADIUS + GLM_RAND_FIX));
			RandomPosition = RandomPosition + pCPlayer->GetPlayerPosition();
			RandomPosition.y = pCTerrain->GetHeightAt(RandomPosition);

			PerlinFactor = (1.0f + SimplePerlin((GLint)RandomPosition.x, (GLint)RandomPosition.z)) / 2.0f;
			*BrightnessValuesPtr++ = GrassMinBrightness + (BrightnessRange * PerlinFactor);

			//strViewMatrix = glm::to_string(RandomPosition);
			//fprintf_s(Essential.gpFile, "RandomPosition:%s\n", strViewMatrix.c_str());

			//TranslationMatrixGrass = glm::translate(glm::mat4(1.0f), RandomPosition);
			//Matrices[i] = Matrices[i] * TranslationMatrixGrass;

			//RotationMatrxGrass = glm::rotate(glm::mat4(1.0f), (float)rand(), glm::vec3(0.0f, 1.0f, 0.0f));
			//Matrices[i] = Matrices[i] * RotationMatrxGrass;

			////ScaleMatrixGrass = glm::scale(glm::mat4(1.0f), glm::vec3(glm::vec3(glm::linearRand(1.0f, 3.5f)), glm::linearRand(GrassMinHeight, GrassMaxHeight), 1.0f));
			//ScaleMatrixGrass = glm::scale(Temp, glm::vec3(glm::linearRand(1.0f, 2.0f), glm::linearRand(GrassMinHeight, GrassMaxHeight), 1.0f));
			//Matrices[i] = Matrices[i] * ScaleMatrixGrass;

			//Matrices[i] = glm::mat4(1.0f);
			Matrices[i] = glm::translate(Temp, RandomPosition) * glm::rotate(Temp, (GLfloat)rand(), glm::vec3(0.3f, 1.0f, 0.3f)) * glm::scale(Temp, glm::vec3(glm::linearRand(1.0f, 2.0f), glm::linearRand(GrassMinHeight, GrassMaxHeight), 1.0f));
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		bIsGrassDrawn = GL_TRUE;
	}

	glBindVertexArray(0);

	//PlaceGrass();
}

GLuint CGrass::GetShaderObject(int type)
{
	if (type == GRASS_DAY)
		return(this->ShaderProgramObjectGrass);
	else
		return(this->ShaderProgramObjectGrassNight);
}

float CGrass::SimplePerlin(int x, int y)
{
	int n;

	n = x + y * 57;
	n = (n << 13) ^ n;
	return(1.0f - ((n * ((n * n * 15731) + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);
}

void CGrass::PlaceGrass(CPlayer *pCPlayer, GLuint ShaderObject)
{
	// build a list of grass objects
	for (int i = 0; i < MAX_GRASS_BLADES; i++)
	{
		RandomPosition = glm::vec3(glm::linearRand(-GRASS_RADIUS, GRASS_RADIUS + GLM_RAND_FIX), 0.0f, glm::linearRand(-GRASS_RADIUS, GRASS_RADIUS + GLM_RAND_FIX));

		PerlinFactor = (1.0f + SimplePerlin((GLint)RandomPosition.x, (GLint)RandomPosition.z)) / 2.0f;
		*BrightnessValuesPtr++ = GrassMinBrightness + (BrightnessRange * PerlinFactor);

		RandomPosition = RandomPosition + pCPlayer->GetPlayerPosition();
		//RandomPosition = RandomPosition + pCCamera->GetCameraPosition();
		RandomPosition.y = pCTerrain->GetHeightAt(RandomPosition);

		Temp = glm::mat4(1.0f);
		Temp = glm::translate(Temp, RandomPosition);
		Temp = glm::scale(Temp, glm::vec3(1.0f, glm::linearRand(GrassMinHeight, GrassMaxHeight), 1.0f));
		Temp = glm::rotate(Temp, (float)rand(), glm::vec3(0.0f, 1.0f, 0.0f));

		*SidePtr++ = Temp[0];
		*UpPtr++ = Temp[1];
		*ForwardPtr++ = Temp[2];
		*PosPtr = Temp[3];
	}

	glBindVertexArray(VAOGrass);

	//glBindBuffer(GL_ARRAY_BUFFER, gVBOBrightness);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * MAX_GRASS_BLADES, pBrightnessValues);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, VBOModelMatrices);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec4) * MAX_GRASS_BLADES * 4, ModelMatrices);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	delete[] BrightnessValues;
}

void CGrass::UpdateGrassLoop(void)
{
	glm::vec4 *PositionPtr;

	glm::vec4 PlayerPos;
	glm::vec4 Pos;
	glm::vec4 Diff;
	bool IsGrassShifted = false;
	int i;

	if (pCPlayer->GetIsMoving())
	{
		PositionPtr = &ModelMatrices[MAX_GRASS_BLADES * 3];

		PlayerPos = glm::vec4(pCPlayer->GetPlayerPosition(), 1.0f);
		PlayerPos.y = 0.0f;

		for (i = 0; i < MAX_GRASS_BLADES; i++)
		{
			Pos = *PositionPtr;
			Pos.y = 0.0f;
			Diff = PlayerPos - Pos;
			IsGrassShifted = false;

			if (Diff.x >(GRASS_RADIUS + 1))
			{
				Pos.x += GRASS_RADIUS * 2.0f;
				IsGrassShifted = true;
			}
			else if (Diff.x < -(GRASS_RADIUS + 1))
			{
				Pos.x -= GRASS_RADIUS * 2.0f;
				IsGrassShifted = true;
			}

			if (Diff.z >(GRASS_RADIUS + 1))
			{
				Pos.z += GRASS_RADIUS * 2.0f;
				IsGrassShifted = true;
			}
			else if (Diff.z < -(GRASS_RADIUS + 1))
			{
				Pos.z -= GRASS_RADIUS * 2.0f;
				IsGrassShifted = true;
			}

			if (IsGrassShifted)
			{
				//Pos.y = GetHeightAt(Pos);

				*PositionPtr = Pos;
			}
			PositionPtr++;
		}
	}
}

void CGrass::ControlGrassWaving(GLfloat dt)
{
	float fWaveSpeed = 1.0f * dt;

	WaveValue += fWaveSpeed;
}

void CGrass::DeletGrass(void)
{
	delete[] ModelMatrices;
}


void CGrass::RenderGrass(glm::mat4 ModelMatrix, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix)
{
	glm::mat4 TranslationMatrixGrass = glm::mat4(1.0f);
	glm::mat4 RotationMatrxGrass = glm::mat4(1.0f);
	glm::mat4 ScaleMatrixGrass = glm::mat4(1.0f);

	//PlaceGrass();

	glUseProgram(ShaderProgramObjectGrass);

	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectGrass, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(ProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectGrass, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	//glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectGrass, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	glUniform1f(glGetUniformLocation(ShaderProgramObjectGrass, "WaveTime"), WaveValue);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectGrass, "GrassAreaRadius"), GRASS_RADIUS);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectGrass, "WaveStrength"), 0.025f);

	glBindVertexArray(VAOGrass);

	//glBindBuffer(GL_ARRAY_BUFFER, gVBOModelMatrices);
	//glBufferSubData(GL_ARRAY_BUFFER, (sizeof(glm::vec4) * MAX_GRASS_BLADES * 3) + (sizeof(glm::vec4) * giUpdateChunkIndex * iNumBladesPerUpdate), sizeof(glm::vec4) * iNumBladesPerUpdate, gpModelMatrices);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * MAX_GRASS_BLADES * 4, gpModelMatrices, GL_DYNAMIC_DRAW);

	

	glBindBuffer(GL_ARRAY_BUFFER, VBOBrightness);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * MAX_GRASS_BLADES, BrightnessValues);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//gpModelMatrixUpdateChunk += iNumBladesPerUpdate;
	//giUpdateChunkIndex++;

	//if (giUpdateChunkIndex >= iResetPointerSteps)
	//{
	//	gpModelMatrixUpdateChunk = &gpModelMatrices[MAX_GRASS_BLADES * 3];
	//	giUpdateChunkIndex = 0;
	//}

	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, MAX_GRASS_BLADES);
	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);

	glBindVertexArray(0);

	glUseProgram(0);
}

void CGrass::RenderGrassNight(glm::mat4 ModelMatrix, glm::mat4 ViewMatrix, glm::mat4 ProjectionMatrix)
{
	glm::mat4 TranslationMatrixGrass = glm::mat4(1.0f);
	glm::mat4 RotationMatrxGrass = glm::mat4(1.0f);
	glm::mat4 ScaleMatrixGrass = glm::mat4(1.0f);

	//PlaceGrass();

	glUseProgram(ShaderProgramObjectGrassNight);

	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectGrassNight, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(ProjectionMatrix));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectGrassNight, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(ViewMatrix));
	//glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObjectGrass, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(ModelMatrix));

	glUniform1f(glGetUniformLocation(ShaderProgramObjectGrassNight, "WaveTime"), WaveValue);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectGrassNight, "GrassAreaRadius"), GRASS_RADIUS);
	glUniform1f(glGetUniformLocation(ShaderProgramObjectGrassNight, "WaveStrength"), 0.025f);

	glBindVertexArray(VAOGrass);

	//glBindBuffer(GL_ARRAY_BUFFER, gVBOModelMatrices);
	//glBufferSubData(GL_ARRAY_BUFFER, (sizeof(glm::vec4) * MAX_GRASS_BLADES * 3) + (sizeof(glm::vec4) * giUpdateChunkIndex * iNumBladesPerUpdate), sizeof(glm::vec4) * iNumBladesPerUpdate, gpModelMatrices);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * MAX_GRASS_BLADES * 4, gpModelMatrices, GL_DYNAMIC_DRAW);

	if (bIsGrassDrawn == GL_FALSE)
	{
		glBindBuffer(GL_ARRAY_BUFFER, ModelMatrixBuffer);
		glm::mat4 *Matrices = (glm::mat4 *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (int i = 0; i < MAX_GRASS_BLADES; i++)
		{
			RandomPosition = glm::vec3(glm::linearRand(-GRASS_RADIUS, GRASS_RADIUS + GLM_RAND_FIX), 0.0f, glm::linearRand(-GRASS_RADIUS, GRASS_RADIUS + GLM_RAND_FIX));
			RandomPosition = RandomPosition + pCPlayer->GetPlayerPosition();
			RandomPosition.y = pCTerrain->GetHeightAt(RandomPosition);

			PerlinFactor = (1.0f + SimplePerlin((GLint)RandomPosition.x, (GLint)RandomPosition.z)) / 2.0f;
			*BrightnessValuesPtr++ = GrassMinBrightness + (BrightnessRange * PerlinFactor);

			//strViewMatrix = glm::to_string(RandomPosition);
			//fprintf_s(Essential.gpFile, "RandomPosition:%s\n", strViewMatrix.c_str());

			/*TranslationMatrixGrass = glm::translate(glm::mat4(1.0f), RandomPosition);
			Matrices[i] = Matrices[i] * TranslationMatrixGrass;

			RotationMatrxGrass = glm::rotate(glm::mat4(1.0f), (float)rand(), glm::vec3(0.0f, 1.0f, 0.0f));
			Matrices[i] = Matrices[i] * RotationMatrxGrass;

			ScaleMatrixGrass = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, glm::linearRand(fGrassMinHeight, fGrassMaxHeight), 1.0f));
			Matrices[i] = Matrices[i] * ScaleMatrixGrass;*/

			Matrices[i] = glm::mat4(1.0f);
			Matrices[i] = glm::translate(Temp, RandomPosition) * glm::rotate(Temp, (float)rand(), glm::vec3(0.0f, 1.0f, 0.0f)) * glm::scale(Temp, glm::vec3(1.0f, glm::linearRand(GrassMinHeight, GrassMaxHeight), 1.0f));
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		bIsGrassDrawn = GL_TRUE;
	}

	glBindBuffer(GL_ARRAY_BUFFER, VBOBrightness);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * MAX_GRASS_BLADES, BrightnessValues);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//gpModelMatrixUpdateChunk += iNumBladesPerUpdate;
	//giUpdateChunkIndex++;

	//if (giUpdateChunkIndex >= iResetPointerSteps)
	//{
	//	gpModelMatrixUpdateChunk = &gpModelMatrices[MAX_GRASS_BLADES * 3];
	//	giUpdateChunkIndex = 0;
	//}

	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, MAX_GRASS_BLADES);
	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);

	glBindVertexArray(0);

	glUseProgram(0);
}

