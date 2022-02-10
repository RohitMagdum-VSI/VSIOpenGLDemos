
extern FILE *RRJ_gpFile;

// extern vec3 cameraPosition = {0.0f, 10.0f, 0.0f};
// extern vec3 cameraFront = {0.0f, 0.0f, -1.0f};
// extern vec3 cameraUp = {0.0f, 1.0f, 0.0f};
// extern vec3 cameraLookingAt = {0.0f, 0.0f, 0.0f};

typedef struct _CAMERA{

	vec3 cameraPosition;
	vec3 cameraFront;
	vec3 cameraUp;
	vec3 cameraLookingAt;

}CAMERA;


//For Camera
extern CAMERA c;



GLfloat yawAngle = -90.0f;
GLfloat pitchAngle = 0.0f;
GLfloat gLastX = 1366.0f / 2.0f;
GLfloat gLastY = 768.0f / 2.0f;
GLfloat gSensitivity = 0.25F;

GLfloat c_PI = 3.1415926535f;

GLfloat degToRad(GLfloat deg){

	return(deg * (c_PI / 180.0f));
}

void initialize_Camera(CAMERA &c){

	c.cameraPosition = vec3(0.0f, 30.0f, 0.0f);
	c.cameraFront = vec3(0.0f, 0.0f, -1.0f);
	c.cameraUp = vec3(0.0f, 1.0f, 0.0f);
	c.cameraLookingAt = vec3(0.0f, 0.0f, 0.0f);
}


void moveForward(CAMERA &c, GLfloat speed){

	vec3 temp;

	temp = c.cameraFront * vec3(speed);
	c.cameraPosition = c.cameraPosition + temp;
}

void moveBackward(CAMERA &c, GLfloat speed){

	vec3 temp;

	temp = c.cameraFront * vec3(speed);
	c.cameraPosition = c.cameraPosition - temp;

}
	

void moveForwardStright(CAMERA &c, GLfloat speed){

	vec3 temp;

	temp = c.cameraFront * vec3(speed, 1.0f, speed);
	c.cameraPosition = c.cameraPosition + vec3(temp[0], 0.0f, temp[2]);
}	

void moveBackwardStright(CAMERA &c, GLfloat speed){

	vec3 temp;

	temp = c.cameraFront * vec3(speed, 1.0f, speed);
	c.cameraPosition = c.cameraPosition - vec3(temp[0], 0.0f, temp[2]);
}

void moveLeft(CAMERA &c, GLfloat speed){

	vec3 crossProduct;

	crossProduct = vmath::cross(c.cameraFront, c.cameraUp);
	vmath::normalize(crossProduct);

	crossProduct = crossProduct * vec3(speed);
	c.cameraPosition = c.cameraPosition - crossProduct;

}

void moveRight(CAMERA &c, GLfloat speed){
	
	vec3 crossProduct;

	crossProduct = vmath::cross(c.cameraFront, c.cameraUp);
	vmath::normalize(crossProduct);

	crossProduct = crossProduct * vec3(speed);
	c.cameraPosition = c.cameraPosition + crossProduct;
}
	
void setCameraFrontUsingAngle(CAMERA &c, GLfloat pitch, GLfloat yawn){


	GLfloat x = cos(degToRad(yawn)) * cos(degToRad(pitch));
	GLfloat y = sin(degToRad(pitch));
	GLfloat z = sin(degToRad(yawn)) * cos(degToRad(pitch));

	c.cameraFront = vmath::normalize(vec3(x, y, z));

}


void cameraRotation(CAMERA &c, GLfloat offsetX, GLfloat offsetY){


	

	GLfloat xoffset = offsetX - gLastX;
	GLfloat yoffset = gLastY - offsetY;

	// fprintf(RRJ_gpFile, "%f/ %f/ %f/ %f : %f %f\n", offsetX, offsetY, gLastX, gLastY, xoffset, yoffset);

	gLastX = offsetX;
	gLastY = offsetY;

	xoffset *= gSensitivity;
	yoffset *= gSensitivity;

	yawAngle += yoffset;
	pitchAngle += xoffset;

	if(pitchAngle > 89.0)
		pitchAngle = 89.0;

	if(pitchAngle < -89.0)
		pitchAngle = -89.0;


	//fprintf(RRJ_gpFile, "02-Camera: %f/ %f -> %f/ %f\n", pitchAngle, yawAngle, offsetX, offsetY);

	setCameraFrontUsingAngle(c, pitchAngle, yawAngle);
}



