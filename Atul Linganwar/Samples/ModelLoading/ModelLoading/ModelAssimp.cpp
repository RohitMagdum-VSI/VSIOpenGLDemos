#include <Windows.h>
#include <windowsx.h>
#include <stdio.h>
#include <stdlib.h>

#include <gl/glew.h>
#include <gl/GL.h>

#include <vector>

#include "../common/vmath.h"
#include "Main.h"

// Assimp
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "ModelAssimp.h"

#define SET_FLOAT4(f,a,b,c,d) { f[0] = a; f[1] = b, f[2] = c; f[3] = d; }
#define COLOR4_TO_FLOAT4(c,f) { f[0] = c.r; f[1] = c.g; f[2] = c.b; f[3] = c.a; }

extern FILE* gpFile;

bool ModelAssimp::LoadModel(PCHAR pchFileName, PMODEL_DATA pModelData)
{
	if (NULL == pModelData)
	{
		return false;
	}

	pModelData->pAiScene = aiImportFile(pchFileName, aiProcessPreset_TargetRealtime_MaxQuality);
	if (!pModelData->pAiScene)
	{
		fprintf(gpFile, "Error while aiImportFile(%s)\n", pchFileName);
		return false;
	}

	bool bRet = LoadNodeData(pModelData, pModelData->pAiScene->mRootNode);
	if (false == bRet)
	{
		fprintf(gpFile, "Error while LoadNodeData(rootNode).\n");
		if (pModelData->pAiScene)
		{
			aiReleaseImport(pModelData->pAiScene);
			pModelData->pAiScene = NULL;
		}
		return false;
	}

	if (pModelData->pAiScene)
	{
		aiReleaseImport(pModelData->pAiScene);
		pModelData->pAiScene = NULL;
	}

	return true;
}

void ModelAssimp::FreeModelData(PMODEL_DATA pModelData)
{
	if (NULL == pModelData)
	{
		return;
	}

	if (pModelData->vec3Vertices.size())
	{
		pModelData->vec3Vertices.clear();
	}

	if (pModelData->vec3Colors.size())
	{
		pModelData->vec3Colors.clear();
	}

	if (pModelData->vec3Normals.size())
	{
		pModelData->vec3Normals.clear();
	}

	if (pModelData->vec2Textures.size())
	{
		pModelData->vec2Textures.clear();
	}

	if (pModelData->vec3Tangents.size())
	{
		pModelData->vec3Tangents.clear();
	}

	if (pModelData->vec3BiTangents.size())
	{
		pModelData->vec3BiTangents.clear();
	}

	if (NULL != pModelData->pAiScene)
	{
		aiReleaseImport(pModelData->pAiScene);
		pModelData->pAiScene = NULL;
	}

	return;
}

bool ModelAssimp::LoadNodeData(PMODEL_DATA pModelData, const struct aiNode* pAiNode)
{
	bool bRet = false;
	vmath::vec3 vec3Temp;

	for (int iMesh = 0; iMesh < pAiNode->mNumMeshes; ++iMesh)
	{
		const struct aiMesh* pAiMesh = pModelData->pAiScene->mMeshes[pAiNode->mMeshes[iMesh]];

		bRet = LoadMaterialData(pModelData, pModelData->pAiScene->mMaterials[pAiMesh->mMaterialIndex]);
		if (false == bRet)
		{
			fprintf(gpFile, "Error while LoadMaterialData[iMesh = %d]\n", iMesh);
			return false;
		}

		for (int iFace = 0; iFace < pAiMesh->mNumFaces; ++iFace)
		{
			const struct aiFace Face = pAiMesh->mFaces[iFace];

			for (int iIndice = 0; iIndice < Face.mNumIndices; iIndice++)
			{
				int index = Face.mIndices[iIndice];

				// Vertices
				if (pAiMesh->mVertices)
				{
					vec3Temp = vmath::vec3(pAiMesh->mVertices[index].x,
						pAiMesh->mVertices[index].y,
						pAiMesh->mVertices[index].z);

					pModelData->vec3Vertices.push_back(vec3Temp);
				}

				// Vertex Colors
				if (pAiMesh->HasVertexColors(0))
				{
					vec3Temp = vmath::vec3(pAiMesh->mColors[0][index].r,
						pAiMesh->mColors[0][index].g,
						pAiMesh->mColors[0][index].b);

					pModelData->vec3Colors.push_back(vec3Temp);
				}

				// Normals
				if (pAiMesh->HasNormals())
				{
					vec3Temp = vmath::vec3(pAiMesh->mNormals[index].x,
						pAiMesh->mNormals[index].y,
						pAiMesh->mNormals[index].z);

					pModelData->vec3Normals.push_back(vec3Temp);
				}

				// Tangents and BiTangents
				if (pAiMesh->HasTangentsAndBitangents())
				{
					vec3Temp = vmath::vec3(pAiMesh->mTangents[index].x,
						pAiMesh->mTangents[index].y,
						pAiMesh->mTangents[index].z);

					pModelData->vec3Tangents.push_back(vec3Temp);

					vec3Temp = vmath::vec3(pAiMesh->mBitangents[index].x,
						pAiMesh->mBitangents[index].y,
						pAiMesh->mBitangents[index].z);

					pModelData->vec3BiTangents.push_back(vec3Temp);

				}
			}
		}
	}

	for (int iChildren = 0; iChildren < pAiNode->mNumChildren; ++iChildren)
	{
		bRet = LoadNodeData(pModelData, pAiNode->mChildren[iChildren]);
		if (false == bRet)
		{
			fprintf(gpFile, "Error while LoadNodeData(iChildren = %d]\n", iChildren);
			return false;
		}
	}

	return true;
}

bool ModelAssimp::LoadMaterialData(PMODEL_DATA pModelData, const struct aiMaterial* pAiMaterial)
{
	C_STRUCT aiColor4D Diffuse;
	C_STRUCT aiColor4D Specular;
	C_STRUCT aiColor4D Ambient;
	C_STRUCT aiColor4D Emission;
	ai_real Shininess, Strength;
	unsigned int max = 1;

	SET_FLOAT4(pModelData->vec4LightDiffuse, 0.8f, 0.8f, 0.8f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(pAiMaterial, AI_MATKEY_COLOR_DIFFUSE, &Diffuse))
	{
		COLOR4_TO_FLOAT4(Diffuse, pModelData->vec4LightDiffuse);
	}

	SET_FLOAT4(pModelData->vec4LightSpecular, 0.0f, 0.0f, 0.0f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(pAiMaterial, AI_MATKEY_COLOR_SPECULAR, &Specular))
	{
		COLOR4_TO_FLOAT4(Specular, pModelData->vec4LightSpecular);
	}

	SET_FLOAT4(pModelData->vec4LightAmbient, 0.2f, 0.2f, 0.2f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(pAiMaterial, AI_MATKEY_COLOR_DIFFUSE, &Ambient))
	{
		COLOR4_TO_FLOAT4(Ambient, pModelData->vec4LightAmbient);
	}

	SET_FLOAT4(pModelData->vec4LightEmission, 0.0f, 0.0f, 0.0f, 1.0f);
	if (AI_SUCCESS == aiGetMaterialColor(pAiMaterial, AI_MATKEY_COLOR_DIFFUSE, &Emission))
	{
		COLOR4_TO_FLOAT4(Emission, pModelData->vec4LightEmission);
	}

	if (AI_SUCCESS == aiGetMaterialFloatArray(pAiMaterial, AI_MATKEY_SHININESS, &Shininess, &max))
	{
		pModelData->fShininess = Shininess;
		max = 1;
		if (AI_SUCCESS == aiGetMaterialFloatArray(pAiMaterial, AI_MATKEY_SHININESS_STRENGTH, &Strength, &max))
			pModelData->fStrength = Strength;
		else
			pModelData->fStrength = 0.0f;
	}
	else
	{
		pModelData->fShininess = 0.0f;
		pModelData->fStrength = 0.0f;
	}

	return true;
}