
///////////////////////   SPHERE  ///////////////////////
//For Sphere
GLuint Shape_Vao_Sphere;
GLuint Shape_Vbo_Sphere_Pos;
GLuint Shape_Vbo_Sphere_Normals;
GLuint Shape_Vbo_Sphere_Texcoord;
GLuint Shape_Vbo_Sphere_Index;


#define STACK 30
#define SLICES 30
#define TOTAL_TRIANGLE (((STACK - 1) * 2) + (STACK - 1) * 2 * (STACK - 1 - 2)) 

GLfloat sphere_Pos[STACK * SLICES * 3];
GLfloat sphere_Nor[STACK * SLICES * 3];
GLfloat sphere_Texcoord[STACK * SLICES * 2];
GLuint sphere_Indicies[TOTAL_TRIANGLE * 3];
GLint gNumOfElements = 0;


void init_sphere(void){

	/********** Pyramid Vertices Information **********/
	void MakeMySphere(GLfloat radius, GLint stackCount, GLint sliceCount, GLfloat *pos, GLfloat *nor, GLfloat *tex, GLuint* index);

	MakeMySphere(1.0f, STACK - 1, SLICES - 1, sphere_Pos, sphere_Nor, sphere_Texcoord, sphere_Indicies);


	/********** Creating Vertex Array Object **********/
	glGenVertexArrays(1, &Shape_Vao_Sphere);
	glBindVertexArray(Shape_Vao_Sphere);

		/********** Creating Vertex Buffer Object Position *********/
		glGenBuffers(1, &Shape_Vbo_Sphere_Pos);
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Sphere_Pos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Pos), sphere_Pos, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Creating Vertex Buffer Object Normal *********/
		glGenBuffers(1, &Shape_Vbo_Sphere_Normals);
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Sphere_Normals);
		glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Nor), sphere_Nor, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Creating Vertex Buffer Object Texcoord *********/
		glGenBuffers(1, &Shape_Vbo_Sphere_Texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Sphere_Texcoord);
		glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_Texcoord), sphere_Texcoord, GL_STATIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Creating Vertex Buffer Object Index *********/
		glGenBuffers(1, &Shape_Vbo_Sphere_Index);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Shape_Vbo_Sphere_Index);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_Indicies), sphere_Indicies, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	
	glBindVertexArray(0);

	fprintf(gpFile, "05-Shapes: Sphere Init Done\n");
}


void uninit_sphere(void){
	if (Shape_Vbo_Sphere_Index) {
		glDeleteBuffers(1, &Shape_Vbo_Sphere_Index);
		Shape_Vbo_Sphere_Index = 0;
	}

	if (Shape_Vbo_Sphere_Normals) {
		glDeleteBuffers(1, &Shape_Vbo_Sphere_Normals);
		Shape_Vbo_Sphere_Normals = 0;
	}

	if (Shape_Vbo_Sphere_Texcoord) {
		glDeleteBuffers(1, &Shape_Vbo_Sphere_Texcoord);
		Shape_Vbo_Sphere_Texcoord = 0;
	}


	if (Shape_Vbo_Sphere_Pos) {
		glDeleteBuffers(1, &Shape_Vbo_Sphere_Pos);
		Shape_Vbo_Sphere_Pos = 0;
	}

	if (Shape_Vao_Sphere) {
		glDeleteVertexArrays(1, &Shape_Vao_Sphere);
		Shape_Vao_Sphere = 0;
	}

	fprintf(gpFile, "05-Shapes: Sphere UnInit Done\n");

}


void draw_Sphere(void){

	glBindVertexArray(Shape_Vao_Sphere);
		// glDrawArrays(GL_POINTS, 0, STACK * SLICES);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Shape_Vbo_Sphere_Index);
		glDrawElements(GL_TRIANGLES, gNumOfElements, GL_UNSIGNED_INT, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

}



void MakeMySphere(GLfloat radius, GLint stackCount, GLint sliceCount, GLfloat *pos, GLfloat *nor, GLfloat *tex, GLuint *indices){

	const GLfloat Pi = 3.14159265359f;

	GLfloat x, y, z, xy;
	GLfloat nx, ny, nz;
	GLfloat s, t;
	GLfloat invLen = 1.0f / radius;

	GLfloat sliceStep = 2.0f * Pi / (sliceCount);
	GLfloat stackStep = Pi / (stackCount);

	GLfloat sliceAngle;
	GLfloat stackAngle;


	for(int i = 0; i <= stackCount; i++){

		stackAngle = Pi / 2.0f - (i * stackStep);	

		xy = radius * cos(stackAngle);
		z = radius * sin(stackAngle);

		for(int j = 0; j <= sliceCount; j++){

			sliceAngle = j * sliceStep;

			x = xy * cos(sliceAngle);
			y = xy * sin(sliceAngle);

			pos[(i * (sliceCount + 1) * 3) + (j * 3) + 0] = x;
			pos[(i * (sliceCount + 1) * 3) + (j * 3) + 1] = y;
			pos[(i * (sliceCount + 1) * 3) + (j * 3) + 2] = z;


			nx = x * invLen;
			ny = y * invLen;
			nz = z * invLen;

			nor[(i * (sliceCount + 1) * 3) + (j * 3) + 0] = nx;
			nor[(i * (sliceCount + 1) * 3) + (j * 3) + 1] = ny;
			nor[(i * (sliceCount + 1) * 3) + (j * 3) + 2] = nz;

			s = ((GLfloat) j / (sliceCount));
			t = ((GLfloat) i / (stackCount));

			tex[(i * (sliceCount + 1) * 2) + (j * 2) + 0] = s;
			tex[(i * (sliceCount + 1) * 2) + (j * 2) + 1] = t;
		}

	}



	GLuint k1, k2;
	GLint index = 0;

	for(int i = 0; i < stackCount; i++){

		k1 = i * (sliceCount + 1);
		k2 = k1 + (sliceCount + 1);

		for(int j = 0; j < sliceCount; j++, k1++, k2++){

			if(i != 0){

				

				indices[index + 0] = k1;
				indices[index + 1] = k2;
				indices[index + 2] = k1 + 1;

				index = index + 3;

				// fprintf(gpFile, "1 : %d, %d\n", i, j);

			}

			if(i != (stackCount - 1)){

				indices[index + 0] = k1 + 1;
				indices[index + 1] = k2;
				indices[index + 2] = k2 + 1;

				index = index + 3;
				// fprintf(gpFile, "2 : %d, %d\n", i, j);

			}

		}

	}

	gNumOfElements = index;

	fprintf(gpFile, "%d\n", index);



}



///////////////////////   RECT  ///////////////////////
//For Rect
GLuint Shape_Vao_Rect;
GLuint Shape_Vbo_Rect_Pos;
GLuint Shape_Vbo_Rect_Texcoord;
GLuint Shape_Vbo_Rect_Normal;



void init_rect(void){

	
	GLfloat Rect_Texcoord[] = {
		//Front
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
	};


	/********** Creating Vertex Array Object **********/
	glGenVertexArrays(1, &Shape_Vao_Rect);
	glBindVertexArray(Shape_Vao_Rect);

		/********** Creating Vertex Buffer Object Position *********/
		glGenBuffers(1, &Shape_Vbo_Rect_Pos);
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Pos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * 3, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		/********** Creating Vertex Buffer Object Texcoord *********/
		glGenBuffers(1, &Shape_Vbo_Rect_Texcoord);
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Texcoord);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Texcoord), Rect_Texcoord, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_TEXCOORD0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_TEXCOORD0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


		/********** Creating Vertex Buffer Object Normal *********/
		glGenBuffers(1, &Shape_Vbo_Rect_Normal);
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Normal);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * 3, NULL, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(AMC_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(AMC_ATTRIBUTE_NORMAL);
		glBindBuffer(GL_ARRAY_BUFFER, 0);


	
	glBindVertexArray(0);

	fprintf(gpFile, "05-Shapes: Rect Init Done\n");

}


void uninit_rect(void){
	if (Shape_Vbo_Rect_Texcoord) {
		glDeleteBuffers(1, &Shape_Vbo_Rect_Texcoord);
		Shape_Vbo_Rect_Texcoord = 0;
	}

	if (Shape_Vbo_Rect_Normal) {
		glDeleteBuffers(1, &Shape_Vbo_Rect_Normal);
		Shape_Vbo_Rect_Normal = 0;
	}

	if (Shape_Vbo_Rect_Pos) {
		glDeleteBuffers(1, &Shape_Vbo_Rect_Pos);
		Shape_Vbo_Rect_Pos = 0;
	}

	if (Shape_Vao_Rect) {
		glDeleteVertexArrays(1, &Shape_Vao_Rect);
		Shape_Vao_Rect = 0;
	}

	fprintf(gpFile, "05-Shapes: Rect UnInit Done\n");

}


void draw_Rect(GLint flag){

	glBindVertexArray(Shape_Vao_Rect);

	if(flag == 0){

		// Vertical Rectangle

		GLfloat Rect_Position[] = {
			1.0f, 1.0f, 0.0f,
			-1.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
		};


		GLfloat Rect_Normal[] = {
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f,
		};

		// Position
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Pos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Position), Rect_Position, GL_DYNAMIC_DRAW);	
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Normal
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Normal);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Normal), Rect_Normal, GL_DYNAMIC_DRAW);	
		glBindBuffer(GL_ARRAY_BUFFER, 0);


	}
	else if(flag == 1){

		// Horizontal Rectangle

		GLfloat Rect_Position[] = {
			1.0f, 0.0f, -1.0f,
			-1.0f, 0.0f, -1.0f,
			-1.0f, 0.0f, 1.0f,
			1.0f, 0.0f, 1.0f,
		};


		GLfloat Rect_Normal[] = {
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
		};

		// Position
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Pos);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Position), Rect_Position, GL_DYNAMIC_DRAW);	
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Normal
		glBindBuffer(GL_ARRAY_BUFFER, Shape_Vbo_Rect_Normal);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Rect_Normal), Rect_Normal, GL_DYNAMIC_DRAW);	
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

			
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);
}




///////////////////////////////////////////////////////
void initialize_Shapes(void){

	init_sphere();
	init_rect();
}

void uninitialize_Shape(void){

	uninit_sphere();
	uninit_rect();
}
