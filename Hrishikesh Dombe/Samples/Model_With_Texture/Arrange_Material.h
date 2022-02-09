#include<Windows.h>
#include<vector>

void Rearrange_Material_Data(std::vector<Mesh_Data> &mesh, std::vector<material>&mat)
{
	//This loop is for MeshData
	for (int i = 0; i < mesh.size(); i++)
	{
		//This is for Material Data
		for (int j = 0; j < mat.size(); j++)
		{
			if (strcmp(mesh[i].usemtl, mat[j].material_name) == S_EQUAL)
			{
				mesh[i].material_index = j;
			}
		}
	}
}
