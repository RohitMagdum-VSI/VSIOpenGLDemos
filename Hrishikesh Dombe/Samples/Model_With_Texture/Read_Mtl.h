#define STB_IMAGE_IMPLEMENTATION
#include<Windows.h>
#include<stdio.h>
#include<gl/GL.h>
#include<vector>
#include"stb_image.h"

#pragma comment(lib,"user32.lib")

#define BUFFER_SIZE 256
#define S_EQUAL 0
#define NR_KA 3
#define NR_KD 3
#define NR_KS 3


FILE *gpMtlFile;
FILE *gpLog_File;
char line_mtl[BUFFER_SIZE];
int counter_newmtl = 0;
int counter_Ka = 0;
int counter_Kd = 0;
int counter_Ks = 0;
int counter_Ns = 0;
int counter_Ni = 0;
int counter_d = 0;
int counter_illum = 0;
int counter_map_Kd = 0;
int counter_map_Ka = 0;
int counter_map_Ks = 0;
int counter_map_Bump = 0;
//int iterator=0;

struct material {
	char material_name[255];
	float Ns;
	float Ka[4];
	float Kd[4];
	float Ks[4];
	float Ni;
	float d;
	float illum;
	char map_Kd[255];
	char map_Ka[255];
	char map_Ks[255];
	char map_Bump[255];
	bool ismap_Kd = false;
	bool ismap_Ka = false;
	bool ismap_Ks = false;
	bool ismap_Bump = false;
	GLuint gTexture;

	material()
	{
		ismap_Kd = false;
		ismap_Ka = false;
		ismap_Ks = false;
		ismap_Bump = false;
	}
};

struct material material_temp;

char str_material[256];

bool gbFirstTime = false;

void LoadMaterialData(char *filename, std::vector<material>&mat)
{
	int LoadGLTextures(GLuint *, char *);
	void InitTextures(std::vector<material>&);

	MessageBox(NULL, TEXT("In LoadMaterialData()"), TEXT("MSG"), MB_OK);
	gpMtlFile = fopen(filename, "r");
	//gpMtlFile = fopen("Cube.mtl", "r");
	if (!gpMtlFile)
	{
		MessageBox(NULL, TEXT("Cannot Open MTL File"), TEXT("ERROR"), MB_OK);
		exit(EXIT_FAILURE);
	}

	//gpLog_File = fopen("MTL_Log.txt", "w");

	char *first_token = NULL;
	char *token = NULL;
	char *sep_space = " ";
	gbFirstTime = false;
	ZeroMemory(&material_temp, sizeof(material));
	mat.clear();

	while (fgets(line_mtl, BUFFER_SIZE, gpMtlFile) != NULL)
	{
		first_token = strtok(line_mtl, sep_space);
		if (strcmp(first_token, "newmtl") == S_EQUAL)
		{
			if (gbFirstTime == true)
			{
				mat.push_back(material_temp);
				ZeroMemory(&material_temp, sizeof(material));
			}
			token = strtok(NULL, sep_space);
			strcpy(material_temp.material_name, token);
			counter_newmtl++;
			gbFirstTime = true;
		}
		else if (strcmp(first_token, "Ns") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			material_temp.Ns = atof(token);
			counter_Ns++;
		}
		else if (strcmp(first_token, "Ka") == S_EQUAL)
		{
			for (int i = 0; i < NR_KA; i++)
			{
				token = strtok(NULL, sep_space);
				material_temp.Ka[i] = atof(token);
			}
			counter_Ka++;
		}
		else if (strcmp(first_token, "Kd") == S_EQUAL)
		{
			for (int i = 0; i < NR_KD; i++)
			{
				token = strtok(NULL, sep_space);
				material_temp.Kd[i] = atof(token);
			}
			counter_Kd++;
		}
		else if (strcmp(first_token, "Ks") == S_EQUAL)
		{
			for (int i = 0; i < NR_KS; i++)
			{
				token = strtok(NULL, sep_space);
				material_temp.Ks[i] = atof(token);
			}
			counter_Ks++;
		}
		else if (strcmp(first_token, "Ni") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			material_temp.Ni = atof(token);
			counter_Ni++;
		}
		else if (strcmp(first_token, "d") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			material_temp.d = atof(token);
			counter_d++;
		}
		else if (strcmp(first_token, "illum") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			material_temp.illum = atof(token);
			counter_illum++;
		}
		else if (strcmp(first_token, "map_Kd") == S_EQUAL)
		{
			int size;
			token = strtok(NULL, sep_space);
			size = strlen(token);

			memcpy(material_temp.map_Kd, token, size - 1);
			//strcpy(mat[counter_newmtl-1].map_Kd,token);
			material_temp.ismap_Kd = true;
			/*int result = LoadGLTextures(&mat[counter_newmtl - 1].gTexture, mat[counter_newmtl - 1].map_Kd);
			if (result == FALSE)
			{
			sprintf(str_material, "Cannot Load Image :%s.", mat[counter_newmtl - 1].map_Kd);
			MessageBox(NULL, str_material, TEXT("Error"), MB_OK);
			}*/
			counter_map_Kd++;
		}
		else if (strcmp(first_token, "map_Ka") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			strcpy(material_temp.map_Ka, token);
			material_temp.ismap_Ka = true;
			counter_map_Ka++;
		}
		else if (strcmp(first_token, "map_Ks") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			strcpy(material_temp.map_Ks, token);
			material_temp.ismap_Ks = true;
			counter_map_Ks++;
		}
		else if (strcmp(first_token, "map_Bump") == S_EQUAL)
		{
			token = strtok(NULL, sep_space);
			strcpy(material_temp.map_Bump, token);
			material_temp.ismap_Bump = true;
			counter_map_Bump++;
		}
		//iterator++;
	}

	mat.push_back(material_temp);
	ZeroMemory(&material_temp, sizeof(material));

	/*fprintf(gpLog_File, "********************************************\n");

	for (int i = 0; i <= counter_newmtl - 1; i++)
	{
		fprintf(gpLog_File, "Newmtl %s  \n", mat[i].material_name);
		fprintf(gpLog_File, "Ns %f  \n", mat[i].Ns);
		for (int j = 0; j < NR_KA; j++)
		{
			fprintf(gpLog_File, "Ka %f  \t", mat[i].Ka[j]);
		}
		fprintf(gpLog_File, "\n");
		for (int j = 0; j < NR_KD; j++)
		{
			fprintf(gpLog_File, "Kd %f  \t", mat[i].Kd[j]);
		}
		fprintf(gpLog_File, "\n");
		for (int j = 0; j < NR_KS; j++)
		{
			fprintf(gpLog_File, "Ks %f  \t", mat[i].Ks[j]);
		}
		fprintf(gpLog_File, "\n");
		fprintf(gpLog_File, "Ni %f  \n", mat[i].Ni);
		fprintf(gpLog_File, "d %f  \n", mat[i].d);
		fprintf(gpLog_File, "illum %f  \n", mat[i].illum);

		if (mat[i].ismap_Kd == true)
			fprintf(gpLog_File, "map_Kd %s", mat[i].map_Kd);
		if (mat[i].ismap_Ka == true)
			fprintf(gpLog_File, "map_Ka %s", mat[i].map_Ka);
		if (mat[i].ismap_Ks == true)
			fprintf(gpLog_File, "map_Ks %s", mat[i].map_Ks);
		if (mat[i].ismap_Bump == true)
			fprintf(gpLog_File, "map_Bump %s", mat[i].map_Bump);

		fprintf(gpLog_File, "\n\n");
	}


	fprintf(gpLog_File, "newmtl Count : %d\n", counter_newmtl);
	fprintf(gpLog_File, "Ka Count : %d\n", counter_Ka);
	fprintf(gpLog_File, "Kd Count : %d\n", counter_Kd);
	fprintf(gpLog_File, "Ks Count : %d\n", counter_Ks);
	fprintf(gpLog_File, "Ni Count : %d\n", counter_Ni);
	fprintf(gpLog_File, "Ns Count : %d\n", counter_Ns);
	fprintf(gpLog_File, "d Count : %d\n", counter_d);
	fprintf(gpLog_File, "map_Kd Count : %d\n", counter_map_Kd);
	fprintf(gpLog_File, "map_Ka Count : %d\n", counter_map_Ka);
	fprintf(gpLog_File, "map_Ks Count : %d\n", counter_map_Ks);
	fprintf(gpLog_File, "map_Bump Count : %d\n", counter_map_Bump);

	fclose(gpLog_File);*/
	fclose(gpMtlFile);
	InitTextures(mat);
	MessageBox(NULL, TEXT("END"), TEXT("MSG"), MB_OK);
	//return(0);
}

void InitTextures(std::vector<material>&mat)
{
	int LoadGLTextures(GLuint *, char *);
	for (int i = 0; i < mat.size(); i++)
	{
		if (mat[i].ismap_Kd == true)
		{
			int result = LoadGLTextures(&mat[i].gTexture, mat[i].map_Kd);
			if (result == FALSE)
			{
				sprintf(str_material, "Cannot Load Image :%s.", mat[i].map_Kd);
				MessageBox(NULL, str_material, TEXT("Error"), MB_OK);
			}
			else if (result == TRUE)
			{
				//MessageBox(NULL, TEXT("Texture Loaded Successfully"), TEXT("MSG"), MB_OK);
			}
		}
	}
}

int LoadGLTextures(GLuint *texture, char *filename)
{
	//	HBITMAP hBitmap;
	//BITMAP bmp;
	int iStatus = FALSE;
	int width, height, nrComponents;
	unsigned char *image = NULL;

	glGenTextures(1, texture);
	//	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
	image = stbi_load(filename, &width, &height, &nrComponents, 0);
	if (image)
	{
		GLenum format;
		if (nrComponents == 1)
			format = GL_RED;
		else if (nrComponents == 3)
			format = GL_RGB;
		else if (nrComponents == 4)
			format = GL_RGBA;

		iStatus = TRUE;
		//GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, image);

		glGenerateMipmap(GL_TEXTURE_2D);

		//DeleteObject(hBitmap);
		stbi_image_free(image);
	}
	return(iStatus);
}
