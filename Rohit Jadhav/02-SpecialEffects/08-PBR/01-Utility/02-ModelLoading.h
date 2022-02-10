
//Model Loading
struct VecFloat {

	float *pData;
	int iSize;
};


#define RRJ_SUCCESS 1
#define RRJ_ERROR 0

#pragma pack(1)

typedef struct _MODEL{

	FILE *gpFile_Model;
	char szModelName[128];

	GLuint vao;
	GLuint vbo_pos;
	GLuint vbo_tex;
	GLuint vbo_nor;

	GLuint texture;

	struct VecFloat *pVecFloat_Model_Vertices;
	struct VecFloat *pVecFloat_Model_Normals;
	struct VecFloat *pVecFloat_Model_Texcoord;

	struct VecFloat *pVecFloat_Model_Sorted_Vertices;
	struct VecFloat *pVecFloat_Model_Sorted_Normals;
	struct VecFloat *pVecFloat_Model_Sorted_Texcoord;

}MODEL;


struct VecFloat* CreateVecFloat(void) {

	struct VecFloat *pTemp = NULL;

	pTemp = (struct VecFloat*)malloc(sizeof(struct VecFloat));
	if (pTemp == NULL) {
		printf("ERROR: CreateVecInt(): Malloc() Failed!\n");
		exit(0);
	}

	memset((void*)pTemp, 0, sizeof(struct VecFloat));

	return(pTemp);
}


int PushBackVecFloat(struct VecFloat *pVec, float data) {

	pVec->pData = (float*)realloc(pVec->pData, sizeof(struct VecFloat) * (pVec->iSize + 1));

	assert(pVec->pData);

	pVec->iSize = pVec->iSize + 1;
	pVec->pData[pVec->iSize - 1] = data;
	//fprintf(gpFile_RRJ, "iSize: %d   iData: %f\n", pVec->iSize, pVec->pData[pVec->iSize - 1]);

	return(RRJ_SUCCESS);
}


void ShowVecFloat(struct VecFloat *pVec) {

	for (int i = 0; i < pVec->iSize; i++)
		fprintf(gpFile, "P[%d]: %f\t", i, pVec->pData[i]);
}


int DestroyVecFloat(struct VecFloat *pVec) {


	free(pVec->pData);
	pVec->pData = NULL;
	pVec->iSize = 0;
	free(pVec);
	pVec = NULL;

	return(RRJ_SUCCESS);
}


MODEL* initialize_ModelWithFileName(const char* name){

	MODEL *m = (MODEL*)malloc(sizeof(MODEL) * 1);
	if(m == NULL){
		fprintf(gpFile, "initialize_ModelWithFileName(): Malloc() failed\n");
		return(NULL);
	}

	strcpy(m->szModelName, name);
	m->vao = 0;
	m->vbo_pos = 0;
	m->vbo_nor = 0;
	m->vbo_tex = 0;

	m->texture = 0;

	m->pVecFloat_Model_Vertices = NULL;
	m->pVecFloat_Model_Normals = NULL;
	m->pVecFloat_Model_Texcoord = NULL;

	m->pVecFloat_Model_Sorted_Vertices = NULL;
	m->pVecFloat_Model_Sorted_Normals = NULL;
	m->pVecFloat_Model_Sorted_Texcoord = NULL;


	return(m);
}


void LoadModel(MODEL *m) {

	char buffer[1024];
	char *firstToken = NULL;
	char *My_Strtok(char*, char);
	const char *space = " ";
	char *cContext = NULL;


	fopen_s(&m->gpFile_Model, m->szModelName, "r");
	if (m->gpFile_Model == NULL) {
		fprintf(gpFile, "ERROR Model: Model File fopen() Failed!!\n");
		exit(0);
	}
	else
		fprintf(gpFile, "Model : %s Open Succesfully\n", m->szModelName);


	m->pVecFloat_Model_Vertices = CreateVecFloat();
	m->pVecFloat_Model_Normals = CreateVecFloat();
	m->pVecFloat_Model_Texcoord = CreateVecFloat();


	m->pVecFloat_Model_Sorted_Vertices = CreateVecFloat();
	m->pVecFloat_Model_Sorted_Normals = CreateVecFloat();
	m->pVecFloat_Model_Sorted_Texcoord = CreateVecFloat();


	while (fgets(buffer, 1024, m->gpFile_Model) != NULL) {

		firstToken = strtok_s(buffer, space, &cContext);

		if (strcmp(firstToken, "v") == 0) {
			//Vertices
			float x, y, z;
			x = (float)atof(strtok_s(NULL, space, &cContext));
			y = (float)atof(strtok_s(NULL, space, &cContext));
			z = (float)atof(strtok_s(NULL, space, &cContext));

			// fprintf(gpFile, "%f/%f/%f\n", x, y, z);

			PushBackVecFloat(m->pVecFloat_Model_Vertices, x);
			//fprintf(gpFile_Vertices, "\n\nSrt: %f\n", pVecFloat_Model_Vertices->pData[0]);
			PushBackVecFloat(m->pVecFloat_Model_Vertices, y);
			PushBackVecFloat(m->pVecFloat_Model_Vertices, z);

		}
		else if (strcmp(firstToken, "vt") == 0) {
			//Texture

			float u, v;
			u = (float)atof(strtok_s(NULL, space, &cContext));
			v = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf(gpFile_TexCoord, "%f/%f\n", u, v);
			PushBackVecFloat(m->pVecFloat_Model_Texcoord, u);
			PushBackVecFloat(m->pVecFloat_Model_Texcoord, v);
		}
		else if (strcmp(firstToken, "vn") == 0) {
			//Normals

			float x, y, z;
			x = (float)atof(strtok_s(NULL, space, &cContext));
			y = (float)atof(strtok_s(NULL, space, &cContext));
			z = (float)atof(strtok_s(NULL, space, &cContext));

			//fprintf(gpFile_Normals, "%f/%f/%f\n", x, y, z);

			PushBackVecFloat(m->pVecFloat_Model_Normals, x);
			PushBackVecFloat(m->pVecFloat_Model_Normals, y);
			PushBackVecFloat(m->pVecFloat_Model_Normals, z);

		}
		else if (strcmp(firstToken, "f") == 0) {
			//Faces


			for (int i = 0; i < 3; i++) {

				char *faces = strtok_s(NULL, space, &cContext);
				int v, vt, vn;
				v = atoi(My_Strtok(faces, '/')) - 1;
				vt = atoi(My_Strtok(faces, '/')) - 1;
				vn = atoi(My_Strtok(faces, '/')) - 1;

				float x, y, z;

				//Sorted Vertices
				x = m->pVecFloat_Model_Vertices->pData[(v * 3) + 0];
				y = m->pVecFloat_Model_Vertices->pData[(v * 3) + 1];
				z = m->pVecFloat_Model_Vertices->pData[(v * 3) + 2];

				PushBackVecFloat(m->pVecFloat_Model_Sorted_Vertices, x);
				PushBackVecFloat(m->pVecFloat_Model_Sorted_Vertices, y);
				PushBackVecFloat(m->pVecFloat_Model_Sorted_Vertices, z);


				//Sorted Normals
				x = m->pVecFloat_Model_Normals->pData[(vn * 3) + 0];
				y = m->pVecFloat_Model_Normals->pData[(vn * 3) + 1];
				z = m->pVecFloat_Model_Normals->pData[(vn * 3) + 2];

				PushBackVecFloat(m->pVecFloat_Model_Sorted_Normals, x);
				PushBackVecFloat(m->pVecFloat_Model_Sorted_Normals, y);
				PushBackVecFloat(m->pVecFloat_Model_Sorted_Normals, z);


				//Sorted Texcoord;
				x = m->pVecFloat_Model_Texcoord->pData[(vt * 2) + 0];
				y = m->pVecFloat_Model_Texcoord->pData[(vt * 2) + 1];
				z = 0.0f;

				PushBackVecFloat(m->pVecFloat_Model_Sorted_Texcoord, x);
				PushBackVecFloat(m->pVecFloat_Model_Sorted_Texcoord, y);

			}
		}

	}





	/********** Model Vao **********/
	glGenVertexArrays(1, &(m->vao));
	glBindVertexArray(m->vao);

	/********** Position **********/
	glGenBuffers(1, &(m->vbo_pos));
	glBindBuffer(GL_ARRAY_BUFFER, m->vbo_pos);
	glBufferData(GL_ARRAY_BUFFER,
		m->pVecFloat_Model_Sorted_Vertices->iSize * sizeof(float),
		m->pVecFloat_Model_Sorted_Vertices->pData,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Texture **********/
	glGenBuffers(1, &(m->vbo_tex));
	glBindBuffer(GL_ARRAY_BUFFER, m->vbo_tex);
	glBufferData(GL_ARRAY_BUFFER, 
		sizeof(float) * m->pVecFloat_Model_Sorted_Texcoord->iSize,
		m->pVecFloat_Model_Sorted_Texcoord->pData,
		GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0,
		2,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	/********** Normals **********/
	glGenBuffers(1, &(m->vbo_nor));
	glBindBuffer(GL_ARRAY_BUFFER, (m->vbo_nor));
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(float) * m->pVecFloat_Model_Sorted_Normals->iSize,
		m->pVecFloat_Model_Sorted_Normals->pData,
		GL_STATIC_DRAW);

	glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL,
		3,
		GL_FLOAT,
		GL_FALSE,
		0, NULL);

	glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	if(m->gpFile_Model){
		fclose(m->gpFile_Model);
		m->gpFile_Model = NULL;
	}


	fprintf(gpFile, "LoadModel: %s model LoadModel() Done\n", m->szModelName);

}



void uninitialize_Model(MODEL *m){


	if (m->pVecFloat_Model_Sorted_Texcoord) {
		DestroyVecFloat(m->pVecFloat_Model_Sorted_Texcoord);
		m->pVecFloat_Model_Sorted_Texcoord = NULL;
	}

	if (m->pVecFloat_Model_Sorted_Normals) {
		DestroyVecFloat(m->pVecFloat_Model_Sorted_Normals);
		m->pVecFloat_Model_Sorted_Normals = NULL;
	}


	if (m->pVecFloat_Model_Sorted_Vertices) {
		DestroyVecFloat(m->pVecFloat_Model_Sorted_Vertices);
		m->pVecFloat_Model_Sorted_Vertices = NULL;
	}


	if (m->pVecFloat_Model_Normals) {
		DestroyVecFloat(m->pVecFloat_Model_Normals);
		m->pVecFloat_Model_Normals = NULL;
	}

	if (m->pVecFloat_Model_Texcoord) {
		DestroyVecFloat(m->pVecFloat_Model_Texcoord);
		m->pVecFloat_Model_Texcoord = NULL;
	}

	if (m->pVecFloat_Model_Vertices) {
		DestroyVecFloat(m->pVecFloat_Model_Vertices);
		m->pVecFloat_Model_Vertices = NULL;
	}


	if(m->texture){
		glDeleteTextures(1, &(m->texture));
		m->texture = 0;
	}


	if (m->vbo_nor) {
		glDeleteBuffers(1, &(m->vbo_nor));
		m->vbo_nor = 0;
	}


	if(m->vbo_tex){
		glDeleteBuffers(1, &(m->vbo_tex));
		m->vbo_tex = 0;
	}

	if (m->vbo_pos) {
		glDeleteBuffers(1, &(m->vbo_pos));
		m->vbo_pos = 0;
	}

	if (m->vao) {
		glDeleteVertexArrays(1, &(m->vao));
		m->vao = 0;
	}

	if(m){
		free(m);
		m = NULL;
	}

}


void display_Model(MODEL *m){


	glBindVertexArray(m->vao);

		glDrawArrays(GL_TRIANGLES, 0, (m->pVecFloat_Model_Sorted_Vertices->iSize) / 3);

	glBindVertexArray(0);

}

char gBuffer[128];

char* My_Strtok(char* str, char delimiter) {

	static int  i = 0;
	int  j = 0;
	char c;


	while ((c = str[i]) != delimiter && c != '\0') {
		gBuffer[j] = c;
		j = j + 1;
		i = i + 1;
	}

	gBuffer[j] = '\0';


	if (c == '\0') {
		i = 0;
	}
	else
		i = i + 1;


	return(gBuffer);
}
