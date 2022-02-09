#pragma once

typedef struct _MODEL_DATA
{
	const struct aiScene* pAiScene;

	std::vector<vmath::vec3> vec3Vertices;
	std::vector<vmath::vec3> vec3Colors;
	std::vector<vmath::vec3> vec3Normals;
	std::vector<vmath::vec2> vec2Textures;

	std::vector<vmath::vec3> vec3Tangents;
	std::vector<vmath::vec3> vec3BiTangents;

	vmath::vec4 vec4LightAmbient;
	vmath::vec4 vec4LightDiffuse;
	vmath::vec4 vec4LightSpecular;
	vmath::vec4 vec4LightEmission;

	GLfloat fShininess;
	GLfloat fStrength;

}MODEL_DATA, *PMODEL_DATA;

class ModelAssimp
{
public:

	ModelAssimp(){}
	~ModelAssimp(){}

	bool LoadModel(PCHAR pchFileName, PMODEL_DATA pModeData);
	void FreeModelData(PMODEL_DATA pModelData);

private:

	bool LoadNodeData(PMODEL_DATA pModelData, const struct aiNode* pAiNode);
	bool LoadMaterialData(PMODEL_DATA pModelData, const struct aiMaterial* pAiMaterial);
};