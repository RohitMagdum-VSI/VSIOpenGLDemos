#include "Terrain.h"

// Enable multitexture blending accross terrain
#ifndef ENABLE_MULTITEXTURE
#define ENABLE_MULTITEXTURE 1
#endif // !ENABLE_MULTITEXTURE

// Enable the blend constant based on slope of terrain
#ifndef ENABLE_SLOPE_BASED_BLEND
#define ENABLE_SLOPE_BASED_BLEND 1
#endif

namespace fs = std::filesystem;

CTerrain::CTerrain()
{
	if (fopen_s(&TerrainLogFile, filepath, "w") != 0)
	{
		MessageBox(NULL, TEXT("Failed to create the log file. Exiting now."), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf_s(TerrainLogFile, "Log file created successfully.\n");
	}

	pCCommon = new CCommon();

	HeightScale = 10.0f;
	BlockScale = 5.0f;
}

CTerrain::~CTerrain()
{
	if (stoneTextureID)
	{
		glDeleteTextures(1, &stoneTextureID);
		stoneTextureID = 0;
	}

	if (bgTextureID)
	{
		glDeleteTextures(1, &bgTextureID);
		bgTextureID = 0;
	}

	if (rTextureID)
	{
		glDeleteTextures(1, &rTextureID);
		rTextureID = 0;
	}

	if (gTextureID)
	{
		glDeleteTextures(1, &gTextureID);
		gTextureID = 0;
	}
	
	if (bTextureID)
	{
		glDeleteTextures(1, &bTextureID);
		bTextureID = 0;
	}

	if (blendMapTextureID)
	{
		glDeleteTextures(1, &blendMapTextureID);
		blendMapTextureID = 0;
	}
	
	if (VBOTerrainPosition)
	{
		glDeleteBuffers(1, &VBOTerrainPosition);
		VBOTerrainPosition = 0;
	}

	if (VBOTerrainColor)
	{
		glDeleteBuffers(1, &VBOTerrainColor);
		VBOTerrainColor = 0;
	}

	if (VBOTerrainNormal)
	{
		glDeleteBuffers(1, &VBOTerrainNormal);
		VBOTerrainNormal = 0;
	}

	if (VBOTerrainTexCoord)
	{
		glDeleteBuffers(1, &VBOTerrainTexCoord);
		VBOTerrainTexCoord = 0;
	}

	if (VBOTerrainIndex)
	{
		glDeleteBuffers(1, &VBOTerrainIndex);
		VBOTerrainIndex = 0;
	}

	if (VAOTerrainMultitextured)
	{
		glDeleteVertexArrays(1, &VAOTerrainMultitextured);
		VAOTerrainMultitextured = 0;
	}

	if (ShaderProgramObjectTerrrain)
	{
		glDeleteProgram(ShaderProgramObjectTerrrain);
		ShaderProgramObjectTerrrain = 0;
	}

	if (ShaderProgramObjectTerrrainNight)
	{
		glDeleteProgram(ShaderProgramObjectTerrrainNight);
		ShaderProgramObjectTerrrainNight = 0;
	}


	if (TerrainLogFile)
	{
		fopen_s(&TerrainLogFile, filepath, "a+");
		fprintf_s(TerrainLogFile, "Log file closed successfully.\n");
		fclose(TerrainLogFile);
		TerrainLogFile = NULL;
	}

	if (pCCommon != NULL)
	{
		delete[] pCCommon;
	}
}

bool CTerrain::LoadHeightMap(const std::string &fileName, unsigned char bitsPerPixel, unsigned int width, unsigned int height)
{
	if (!fs::exists(fileName))
	{
		fprintf_s(TerrainLogFile, "Could not find the file %s\n", fileName.c_str());
		return false;
	}

	//read file using ifs
	std::ifstream ifs;

	//open file
	ifs.open(fileName, std::ifstream::binary);

	//check if it is opened or not
	if (ifs.fail())
	{
		fprintf_s(TerrainLogFile, "Failed to open %s\n", fileName.c_str());
		return false;
	}

	const unsigned int bytesPerPixel = bitsPerPixel / 8;
	const unsigned int expectedFileSize = (bytesPerPixel * width * height);
	const unsigned int fileSize = GetFileLength(ifs);

	//check for expected file size if differ from actual file size
	if (expectedFileSize != fileSize)
	{
		fprintf_s(TerrainLogFile, "Expected file size [\"%d\" bytes] differ from actual file size [\"%d\"bytes]\n", expectedFileSize, fileSize);
		return false;
	}

	unsigned char* heightmap = new unsigned char[fileSize];
	//read file into heightmap char array
	ifs.read((char*)heightmap, fileSize);

	if (ifs.fail())
	{
		fprintf_s(TerrainLogFile, "Error occured while reading from HeightMap file: %s\n", fileName.c_str());
		ifs.close();
		delete[] heightmap;
		return false;
	}
	//close file
	ifs.close();

	unsigned int numVertices = width * height;
	//resize all buffer vector as per size of width and height
	TerrainPositionBuffer.resize(numVertices);
	TerrainColorBuffer.resize(numVertices);
	TerrainNormalBuffer.resize(numVertices);
	TerrainTangentBuffer.resize(numVertices);
	TerrainTexCoordBuffer.resize(numVertices);

	HeightMapDimensions = glm::uvec2(width, height);

	//size of terrain in world units
	float terrainWidth = (width - 1) * BlockScale;
	float terrainHeight = (height - 1)* BlockScale;

	//half terrain 
	float halfTerrainWidth = terrainWidth * 0.5f;
	float halfTerrainHeight = terrainHeight * 0.5f;

	for (unsigned int j = 0; j < height; ++j)
	{
		for (unsigned int i = 0; i < width; ++i)
		{
			unsigned int index = (j * width) + i;

			assert(index* bytesPerPixel < fileSize);

			float heightValue = GetHeightValue(&heightmap[index* bytesPerPixel], bytesPerPixel);

			float S = (i / (float)(width - 1));
			float T = (j / (float)(height - 1));

			float X = (S * terrainWidth) - halfTerrainWidth;
			float Y = heightValue * HeightScale;
			float Z = (T * terrainHeight) - halfTerrainHeight;

			//Blend 3 terxures depending on the height of the terrain
			float tex0Contribution = 1.0f - GetPercentage(heightValue, 0.0f, 0.75f);
			float tex2Contribution = 1.0f - GetPercentage(heightValue, 0.75f, 1.0f);

			//fill buffers with values
			TerrainNormalBuffer[index] = glm::vec3(0.0f);
			TerrainTangentBuffer[index] = glm::vec3(0.0f);
			TerrainPositionBuffer[index] = glm::vec3(X, Y, Z);
#if ENABLE_MULTITEXTURE
			TerrainColorBuffer[index] = glm::vec4(tex0Contribution, tex0Contribution, tex0Contribution, tex2Contribution);
#else
			TerrainColorBuffer[index] = glm::vec4(1.0f);
#endif
			TerrainTexCoordBuffer[index] = glm::vec2(S, T);

		}
	}

	fprintf_s(TerrainLogFile, "Terrian has been loaded successfully.\n");
	delete[] heightmap;


	GenerateIndexBuffer();
	GenerateNormals();
	return true;
}

float CTerrain::GetHeightAt(const glm::vec3& position)
{
	float height = -FLT_MAX;

	//check for terrain dimenstions are valid
	if (HeightMapDimensions.x < 2 || HeightMapDimensions.y < 2)
		return height;

	// width and height of the terrain in world units
	float terrainWidth = (HeightMapDimensions.x - 1) * BlockScale;
	float terrainHeight = (HeightMapDimensions.y - 1)* BlockScale;
	float halfWidth = terrainWidth * 0.5f;
	float halfHeight = terrainHeight * 0.5f;

	//multiply position by inverse of the terrain matrix to get the position in terrain local space
	glm::vec3 terrainPosition = glm::vec3(glm::vec4(position, 1.0f));
	glm::vec3 invBlockScale(1.0f / BlockScale, 0.0f, 1.0f / BlockScale);

	//calculate offset and scale to get vertex indices
	glm::vec3 offset(halfWidth, 0.0f, halfHeight);

	//get the 4 vertices that make up the triangle we were OVER
	glm::vec3 vertexIndices = (terrainPosition + offset) * invBlockScale;

	GLuint u0 = (GLuint)floorf(vertexIndices.x);
	GLuint u1 = u0 + 1;
	GLuint v0 = (GLuint)floorf(vertexIndices.z);
	GLuint v1 = v0 + 1;

	if (u0 >= 0 && u1 < HeightMapDimensions.x && v0 >= 0 && v1 < HeightMapDimensions.y)
	{
		glm::vec3 p00 = TerrainPositionBuffer[(v0 * HeightMapDimensions.x) + u0]; //Top Left
		glm::vec3 p10 = TerrainPositionBuffer[(v0 * HeightMapDimensions.x) + u1]; //Top right
		glm::vec3 p01 = TerrainPositionBuffer[(v1 * HeightMapDimensions.x) + u0]; // bottom left
		glm::vec3 p11 = TerrainPositionBuffer[(v1 * HeightMapDimensions.x) + u1]; // bottom right

																			   // which triangle are we over?
		float percentU = vertexIndices.x - u0;
		float percentV = vertexIndices.z - v0;

		glm::vec3 dU, dV;
		if (percentU > percentV)
		{
			//top triangle
			dU = p10 - p00;
			dV = p11 - p10;
		}
		else
		{
			//Bottom triangle
			dU = p11 - p01;
			dV = p01 - p00;
		}

		glm::vec3 heightPos = p00 + (dU * percentU) + (dV * percentV);

		//convert back to world space by multiplying by terrains world matrix
		heightPos = glm::vec3(glm::vec4(heightPos, 1));
		height = heightPos.y;
	}

	return height;
}

void CTerrain::GenerateIndexBuffer()
{
	if (HeightMapDimensions.x < 2 || HeightMapDimensions.y < 2)
	{
		// Terrain hasn't been loaded, or is of an incorrect size
		return;
	}

	const unsigned int terrainWidth = HeightMapDimensions.x;
	const unsigned int terrainHeight = HeightMapDimensions.y;

	// Two triangles for every quad of terrain mesh
	const unsigned int numTriangles = (terrainWidth - 1) * (terrainHeight - 1) * 2;

	// 3 indices for each triangle in terrain mesh
	TerrainIndexBuffer.resize(numTriangles * 3);

	unsigned int index = 0; // index for index buffer
	for (unsigned int j = 0; j < (terrainHeight - 1); ++j)
	{
		for (unsigned int i = 0; i < (terrainWidth - 1); ++i)
		{
			int vertexIndex = (j * terrainWidth) + i;
			//Top Triangle T0
			TerrainIndexBuffer[index++] = vertexIndex;						//V0
			TerrainIndexBuffer[index++] = vertexIndex + terrainWidth + 1;	//V3
			TerrainIndexBuffer[index++] = vertexIndex + 1;					//V1
																		//Bottom Triangle T1
			TerrainIndexBuffer[index++] = vertexIndex;						//V0
			TerrainIndexBuffer[index++] = vertexIndex + terrainWidth;		//V2
			TerrainIndexBuffer[index++] = vertexIndex + terrainWidth + 1;	//V3

		}
	}
}

void CTerrain::GenerateNormals()
{
	glm::vec3 v0 = glm::vec3(0.0f);
	glm::vec3 v1 = glm::vec3(0.0f);
	glm::vec3 v2 = glm::vec3(0.0f);

	glm::vec3 normal = glm::vec3(0.0f);

	glm::vec3 Edge1 = glm::vec3(0.0f);
	glm::vec3 Edge2 = glm::vec3(0.0f);

	glm::vec2 deltaUV1 = glm::vec2(0.0f);
	glm::vec2 deltaUV2 = glm::vec2(0.0f);

	glm::vec2 v0_uv = glm::vec2(0.0f);
	glm::vec2 v1_uv = glm::vec2(0.0f);
	glm::vec2 v2_uv = glm::vec2(0.0f);

	glm::vec3 Tangent = glm::vec3(0.0f);

	GLfloat f = 0.0f;

	for (unsigned int i = 0; i < TerrainIndexBuffer.size(); i += 3)
	{
		v0 = TerrainPositionBuffer[TerrainIndexBuffer[i + 0]];
		v1 = TerrainPositionBuffer[TerrainIndexBuffer[i + 1]];
		v2 = TerrainPositionBuffer[TerrainIndexBuffer[i + 2]];

		normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

		TerrainNormalBuffer[TerrainIndexBuffer[i + 0]] += normal;
		TerrainNormalBuffer[TerrainIndexBuffer[i + 1]] += normal;
		TerrainNormalBuffer[TerrainIndexBuffer[i + 2]] += normal;

		// now calculate the tangent
		v0_uv = TerrainTexCoordBuffer[TerrainIndexBuffer[i + 0]];
		v1_uv = TerrainTexCoordBuffer[TerrainIndexBuffer[i + 1]];
		v2_uv = TerrainTexCoordBuffer[TerrainIndexBuffer[i + 2]];

		// calculate tangent vector

		Edge1 = v1 - v0;
		Edge2 = v2 - v0;

		deltaUV1 = v1_uv - v0_uv;
		deltaUV2 = v2_uv - v0_uv;

		f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

		Tangent.x = f * (deltaUV2.y * Edge1.x - deltaUV1.y *  Edge2.x);
		Tangent.y = f * (deltaUV2.y * Edge1.y - deltaUV1.y *  Edge2.y);
		Tangent.z = f * (deltaUV2.y * Edge1.z - deltaUV1.y *  Edge2.z);
		Tangent = glm::normalize(Tangent);

		TerrainTangentBuffer[TerrainIndexBuffer[i + 0]] += Tangent;
		TerrainTangentBuffer[TerrainIndexBuffer[i + 1]] += Tangent;
		TerrainTangentBuffer[TerrainIndexBuffer[i + 2]] += Tangent;
	}
	const glm::vec3 UP(0.0f, 1.0f, 0.0f);

	unsigned int nSize = TerrainNormalBuffer.size();

	for (unsigned int i = 0; i < nSize; ++i)
	{
		TerrainNormalBuffer[i] = glm::normalize(TerrainNormalBuffer[i]);
#if ENABLE_SLOPE_BASED_BLEND
		float fTexture0Contribution = glm::clamp(glm::dot(TerrainNormalBuffer[i], UP) - 0.1f, 0.0f, 1.0f);
		TerrainColorBuffer[i] = glm::vec4(fTexture0Contribution, fTexture0Contribution, fTexture0Contribution, TerrainColorBuffer[i].w);
#endif
	}

	unsigned int nSizeTangent = TerrainTangentBuffer.size();

	for (unsigned int i = 0; i < nSizeTangent; ++i)
	{
		TerrainTangentBuffer[i] = glm::normalize(TerrainTangentBuffer[i]);
	}
}


int CTerrain::GetFileLength(std::istream &file)
{
	int pos = file.tellg();
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	//restore position of get pointer
	file.seekg(pos);
	return length;
}

float CTerrain::GetHeightValue(const unsigned char *data, unsigned char numBytes)
{
	switch (numBytes)
	{
	case 1: // 1byte
		return (unsigned char)(data[0]) / (float)0xff;
		break;
	case 2: // 2byte
		return (unsigned short)(data[1] << 8 | data[0]) / (float)0xffff;
		break;
	case 3: // 4byte
		return (unsigned int)(data[3] << 24 | data[2] << 16 | data[1] << 8 || data[0]) / (float)0xffffffff;
		break;
	default:
		assert(false); // Height field with non std pixel value
		break;
	}
	return 0.0f;
}

float CTerrain::GetPercentage(float value, const float min, const float max)
{
	value = glm::clamp(value, min, max);
	return (value - min) / (max - min);
}

int CTerrain::LoadBMPTexture(TCHAR imageResourceId[], GLuint *Texture)
{
	GLfloat Anisotropy = 0.0f;

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


void CTerrain::InitializeTerrain(GLuint ShaderObjectDay, GLuint ShaderObjectNight)
{
	this->ShaderProgramObjectTerrrain = ShaderObjectDay;
	this->ShaderProgramObjectTerrrainNight = ShaderObjectNight;
	
	stoneTextureID = tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/blackrock-albedo.tga");
	//if (LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_STONE), &stoneTextureID))
	//{
	//	fprintf_s(TerrainLogFile, "IDBITMAP_STONE loaded successfully\n");
	//}

	//bgTextureID = tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/ground_1024_v2_Q3.tga");
	
	bgTextureID = tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/mixedmoss-albedo2.tga");
	
	/*if (LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_BACKGROUND), &bgTextureID))
	{
		fprintf_s(TerrainLogFile, "IDBITMAP_BACKGROUND loaded successfully\n");
	}*/

	NormalMapID - tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/mixedmoss-normal2.tga");

	rTextureID = tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/dry-dirt2-albedo.tga");
	
	/*if (LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_R), &rTextureID))
	{
		fprintf_s(TerrainLogFile, "IDBITMAP_R loaded successfully\n");
	}*/
	
	gTextureID = tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/ground_1024_v2_Q3.tga");

	//if (LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_G), &gTextureID))
	//{
	//	fprintf_s(TerrainLogFile, "IDBITMAP_G loaded successfully\n");
	//}

	bTextureID = tgaLoader.LoadTGATexture("Resources/ImageTextures/TGA/octostoneAlbedo.tga");
	

	if (LoadBMPTexture(MAKEINTRESOURCE(IDBITMAP_BLENDMAP), &blendMapTextureID))
	{
		fprintf_s(TerrainLogFile, "IDBITMAP_BLENDMAP loaded successfully\n");
	}

	fclose(TerrainLogFile);
	
	glGenVertexArrays(1, &VAOTerrainMultitextured);
	glBindVertexArray(VAOTerrainMultitextured);

	// VBO Position
	glGenBuffers(1, &VBOTerrainPosition);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTerrainPosition);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * TerrainPositionBuffer.size(), &TerrainPositionBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_VERTEX, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Color
	glGenBuffers(1, &VBOTerrainColor);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTerrainColor);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * TerrainColorBuffer.size(), &TerrainColorBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_COLOR, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Normal
	glGenBuffers(1, &VBOTerrainNormal);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTerrainNormal);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * TerrainNormalBuffer.size(), &TerrainNormalBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO Normal
	glGenBuffers(1, &VBOTerrainTangent);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTerrainTangent);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * TerrainTangentBuffer.size(), &TerrainTangentBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TANGENT, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TANGENT);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// VBO TEXTURE0
	glGenBuffers(1, &VBOTerrainTexCoord);
	glBindBuffer(GL_ARRAY_BUFFER, VBOTerrainTexCoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * TerrainTexCoordBuffer.size(), &TerrainTexCoordBuffer[0], GL_STATIC_DRAW);
	glVertexAttribPointer(OGL_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(OGL_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glGenBuffers(1, &VBOTerrainIndex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOTerrainIndex);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * TerrainIndexBuffer.size(), &TerrainIndexBuffer[0], GL_STATIC_DRAW);

	//fopen_s(&TerrainLogFile, filepath, "a+");

	//fprintf_s(TerrainLogFile, "Printing Position Buffer\n");
	//for (int i = 0; i < TerrainPositionBuffer.size(); i++)
	//{
	//	fprintf_s(TerrainLogFile, "%s\n", glm::to_string(TerrainPositionBuffer[i]).c_str());
	//}

	//fprintf_s(TerrainLogFile, "\nPrinting Normal Buffer\n");
	//for (int i = 0; i < TerrainNormalBuffer.size(); i++)
	//{
	//	fprintf_s(TerrainLogFile, "%s\n", glm::to_string(TerrainNormalBuffer[i]).c_str());
	//}
	//
	//fprintf_s(TerrainLogFile, "\nPrinting Texture Buffer\n");
	//for (int i = 0; i < TerrainTexCoordBuffer.size(); i++)
	//{
	//	fprintf_s(TerrainLogFile, "%s\n", glm::to_string(TerrainTexCoordBuffer[i]).c_str());
	//}

	//fprintf_s(TerrainLogFile, "\nPrinting Index Buffer\n");
	//for (int i = 0; i < TerrainIndexBuffer.size(); i += 3)
	//{
	//	fprintf_s(TerrainLogFile, "%d\t%d\t%d\n", TerrainIndexBuffer[i], TerrainIndexBuffer[i + 1], TerrainIndexBuffer[i + 2]);
	//}

	//fclose(TerrainLogFile);
}

void CTerrain::RenderTerrain(glm::mat4 Model, glm::mat4 View, glm::mat4 Projection, glm::vec4 clipPlane)
{
	glUseProgram(ShaderProgramObjectTerrrain);

	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(Model));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(View));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(Projection));

	//clip plane
	glUniform4f(glGetUniformLocation(ShaderProgramObjectTerrrain, "clipPlane"), clipPlane.x, clipPlane.y, clipPlane.z, clipPlane.w);


	//bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bgTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_texture0_bg_sampler"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, rTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_texture1_r_sampler"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_texture2_g_sampler"), 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, bTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_texture3_b_sampler"), 3);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, blendMapTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_texture4_blendmap_sampler"), 4);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, NormalMapID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrain, "u_texture5_normalmap_sampler"), 5);

	glBindVertexArray(VAOTerrainMultitextured);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOTerrainIndex);
	glDrawElements(GL_TRIANGLES, TerrainIndexBuffer.size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	

	glUseProgram(0);
}

void CTerrain::RenderTerrainNight(glm::mat4 Model, glm::mat4 View, glm::mat4 Projection,  glm::vec4 clipPlane)
{
	glUseProgram(ShaderProgramObjectTerrrainNight);

	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_model_matrix"), 1, GL_FALSE, glm::value_ptr(Model));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_view_matrix"), 1, GL_FALSE, glm::value_ptr(View));
	glUniformMatrix4fv(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_projection_matrix"), 1, GL_FALSE, glm::value_ptr(Projection));

	//clip plane
	glUniform4f(glGetUniformLocation(ShaderProgramObjectTerrrain, "clipPlane"), clipPlane.x, clipPlane.y, clipPlane.z, clipPlane.w);


	//bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bgTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_texture0_bg_sampler"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, rTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_texture1_r_sampler"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_texture2_g_sampler"), 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, bTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_texture3_b_sampler"), 3);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, blendMapTextureID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_texture4_blendmap_sampler"), 4);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, NormalMapID);
	glUniform1i(glGetUniformLocation(ShaderProgramObjectTerrrainNight, "u_texture5_normalmap_sampler"), 5);

	glBindVertexArray(VAOTerrainMultitextured);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOTerrainIndex);
	glDrawElements(GL_TRIANGLES, TerrainIndexBuffer.size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	glUseProgram(0);
}

GLuint CTerrain::GetShaderObject(GLuint Time)
{
	if (Time == 0)
		return(this->ShaderProgramObjectTerrrain);
	else //(Time == 1)
		return(this->ShaderProgramObjectTerrrainNight);
}

glm::uvec2 CTerrain::GetTerrainWidthHeight()
{
	return glm::uvec2((HeightMapDimensions.x - 1) * BlockScale, (HeightMapDimensions.y - 1) * BlockScale);
}

void CTerrain::DeleteTerrain(void)
{
	if (TerrainLogFile)
	{
		fopen_s(&TerrainLogFile, filepath, "a+");
		fprintf_s(TerrainLogFile, "TerrainLogFile closing");
		fclose(TerrainLogFile);
		TerrainLogFile = NULL;
	}
}
