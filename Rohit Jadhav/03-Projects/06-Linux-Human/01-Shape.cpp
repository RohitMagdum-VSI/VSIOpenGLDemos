#include<stdio.h>
#include<math.h>


int CreateSphere_RRJ(float fRadius, int stack, int slices, float pos[], float tex[], float nor[], unsigned short element[]){


	float lat = 0.0f * (3.1415f / 180.0f);
	float lon = 0.0f * (3.1415f / 180.0f);

	float latFactor = 180.0f * (3.1415f / 180.0f) / stack;
	float lonFactor = 360.0f * (3.1415f / 180.0f) / slices;

	float u = 0.0f;
	float v = 1.0f;

	for(int i = 0; i <= stack; i++){

		//printf("Lat: %f\n", lat);

		lon = 0.0f * (3.1415f / 180.0f);

		for(int j = 0; j < slices; j++){

			//printf("\tLon: %f\n", lon);
			pos[(i * (slices + 0) * 3) + (j * 3) + 0] = fRadius * sin(lat) * cos(lon);
			pos[(i * (slices + 0) * 3) + (j * 3) + 1] = fRadius * sin(lat) * sin(lon);
			pos[(i * (slices + 0) * 3) + (j * 3) + 2] = fRadius * cos(lat);

			tex[(i * (slices + 0) * 3) + (j * 3) + 0] = u + j * (1.0f / (slices - 1));
			tex[(i * (slices + 0) * 3) + (j * 3) + 1] = v - i * (1.0f / stack);

			nor[(i * (slices + 0) * 3) + (j * 3) + 0] = fRadius * sin(lat) * cos(lon);
			nor[(i * (slices + 0) * 3) + (j * 3) + 1] = fRadius * sin(lat) * sin(lon);
			nor[(i * (slices + 0) * 3) + (j * 3) + 2] = fRadius * cos(lat);

			lon = lon + lonFactor;

			//printf("Tex: %f/%f ", tex[(i * (slices + 0) * 3) + (j * 3) + 0], tex[(i * (slices + 0) * 3) + (j * 3) + 1]);
		}
		//printf("\n");
		lat = lat - latFactor;
	}
	

	int index = 0;
	unsigned short topLeft;
	unsigned short topRight;
	unsigned short bottomLeft;
	unsigned short bottomRight;

	for(int i = 0; i < stack; i++){
		for(int j = 0; j < slices; j++){

			if(j < slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = topLeft + 1;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = bottomLeft + 1;

			}
			else if(j == slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = (i) * (slices) + 0;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = (i + 1) * slices + 0;
			}


			element[index + 0] = topLeft;
			element[index + 1] = bottomLeft;
			element[index + 2] = topRight;

			element[index + 3] = topRight;
			element[index + 4] = bottomLeft;
			element[index + 5] = bottomRight;

			index = index + 6;

		}	
	}

	return(index);
}



int CreateSemiSphere_RRJ(float fRadius, int stack, int slices, float pos[], float tex[], float nor[], unsigned short element[]){


	float lat = 0.0f * (3.1415f / 180.0f);
	float lon = 0.0f * (3.1415f / 180.0f);

	float latFactor = 90.0f * (3.1415f / 180.0f) / stack;
	float lonFactor = 360.0f * (3.1415f / 180.0f) / slices;

	float u = 0.0f;
	float v = 1.0f;

	for(int i = 0; i <= stack; i++){

		//printf("Lat: %f\n", lat);

		lon = 0.0f * (3.1415f / 180.0f);

		for(int j = 0; j < slices; j++){

			//printf("\tLon: %f\n", lon);

			if(i == stack){
				pos[(i * (slices + 0) * 3) + (j * 3) + 0] = 0.0f;
				pos[(i * (slices + 0) * 3) + (j * 3) + 1] = 0.0f;
				pos[(i * (slices + 0) * 3) + (j * 3) + 2] = cos(lat);

				/*** NOTE ***/
				/* We take 1.0f * cos(lat) istated of 0.0f * cos(lat)
				   because we need value of z as previous element so that is have
				   same depth 
				*/

				tex[(i * (slices + 0) * 3) + (j * 3) + 0] = u + j * (1.0f / (slices - 1));
				tex[(i * (slices + 0) * 3) + (j * 3) + 1] = v - i * (1.0f / stack);

				nor[(i * (slices + 0) * 3) + (j * 3) + 0] = 0.0f;
				nor[(i * (slices + 0) * 3) + (j * 3) + 1] = 0.0f;
				nor[(i * (slices + 0) * 3) + (j * 3) + 2] = -1.0f * cos(lat);
			}
			else{
				pos[(i * (slices + 0) * 3) + (j * 3) + 0] = fRadius * sin(lat) * cos(lon);
				pos[(i * (slices + 0) * 3) + (j * 3) + 1] = fRadius * sin(lat) * sin(lon);
				pos[(i * (slices + 0) * 3) + (j * 3) + 2] = fRadius * cos(lat);

				tex[(i * (slices + 0) * 3) + (j * 3) + 0] = u + j * (1.0f / (slices - 1));
				tex[(i * (slices + 0) * 3) + (j * 3) + 1] = v - i * (1.0f / stack);

				nor[(i * (slices + 0) * 3) + (j * 3) + 0] = fRadius * sin(lat) * cos(lon);
				nor[(i * (slices + 0) * 3) + (j * 3) + 1] = fRadius * sin(lat) * sin(lon);
				nor[(i * (slices + 0) * 3) + (j * 3) + 2] = fRadius * cos(lat);


			}

			lon = lon + lonFactor;
		}
		if(i < (slices - 1))
			lat = lat + latFactor;
	}
	

	int index = 0;
	unsigned short topLeft;
	unsigned short topRight;
	unsigned short bottomLeft;
	unsigned short bottomRight;

	for(int i = 0; i < stack; i++){
		for(int j = 0; j < slices; j++){

			if(j < slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = topLeft + 1;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = bottomLeft + 1;

			}
			else if(j == slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = (i) * (slices) + 0;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = (i + 1) * slices + 0;
			}


			element[index + 0] = topLeft;
			element[index + 1] = bottomLeft;
			element[index + 2] = topRight;

			element[index + 3] = topRight;
			element[index + 4] = bottomLeft;
			element[index + 5] = bottomRight;

			index = index + 6;

		}	
	}

	return(index);
}


int CreateCylinder_RRJ(float fRadius, float length, int slices,float pos[], float nor[], unsigned short element[]){

	float theta = 0.0f * (3.1415 / 180.0f);
	float thetaFac = 360.0f * (3.1415f / 180.0f) / slices;

	float tempR = fRadius;

	for(int i = 0; i < 4; i++){

		theta = 0.0f * (3.1415 / 180.0f);

		if(i == 0 || i == 3)
			tempR = 0.0f;
		else
			tempR = fRadius;

		for(int  j = 0; j < slices; j++){

			pos[(i * slices * 3) + (j * 3) + 0] = tempR * cos(theta);

			if(i == 0 || i == 1)
				pos[(i * slices * 3) + (j * 3) + 1] = length / 2.0f;
			else if(i == 2 || i == 3)
				pos[(i * slices * 3) + (j * 3) + 1] = -length / 2.0f;

			pos[(i * slices * 3) + (j * 3) + 2] = tempR * sin(theta);

			nor[(i * slices * 3) + (j * 3) + 0] = pos[(i * slices * 3) + (j * 3) + 0];
			nor[(i * slices * 3) + (j * 3) + 1] = pos[(i * slices * 3) + (j * 3) + 1];
			nor[(i * slices * 3) + (j * 3) + 2] = pos[(i * slices * 3) + (j * 3) + 2];

			theta = theta + thetaFac;
		}	
	}



	int index = 0;
	unsigned short topLeft;
	unsigned short topRight;
	unsigned short bottomLeft;
	unsigned short bottomRight;

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < slices; j++){

			if(j < slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = topLeft + 1;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = bottomLeft + 1;

			}
			else if(j == slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = (i) * (slices) + 0;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = (i + 1) * slices + 0;
			}


			element[index + 0] = topLeft;
			element[index + 1] = bottomLeft;
			element[index + 2] = topRight;

			element[index + 3] = topRight;
			element[index + 4] = bottomLeft;
			element[index + 5] = bottomRight;

			index = index + 6;

		}	
	}

	return(index);

}




int CreateFrustum_RRJ(float r1, float r2, float len, int slices, float pos[], float nor[], unsigned short element[]){

	float theta = 0.0f * (3.1415 / 180.0f);
	float thetaFac = 360.0f * (3.1415f / 180.0f) / slices;

	float tempR = 0.0f;

	for(int i = 0; i < 4; i++){

		theta = 0.0f * (3.1415 / 180.0f);

		if(i == 0 || i == 3)
			tempR = 0.0f;
		else if(i == 1)
			tempR = r1;
		else if(i == 2)
			tempR = r2;

		for(int  j = 0; j < slices; j++){

			if(i == 0 || i == 1){
				pos[(i * slices * 3) + (j * 3) + 1] = len / 2.0f;
			}
			else{
				pos[(i * slices * 3) + (j * 3) + 1] = -len / 2.0f;	
			}

			pos[(i * slices * 3) + (j * 3) + 0] = tempR * cos(theta);
			pos[(i * slices * 3) + (j * 3) + 2] = tempR * sin(theta);

			nor[(i * slices * 3) + (j * 3) + 0] = pos[(i * slices * 3) + (j * 3) + 0];
			nor[(i * slices * 3) + (j * 3) + 1] = pos[(i * slices * 3) + (j * 3) + 1];
			nor[(i * slices * 3) + (j * 3) + 2] = pos[(i * slices * 3) + (j * 3) + 2];

			theta = theta + thetaFac;
		}	
	}



	int index = 0;
	unsigned short topLeft;
	unsigned short topRight;
	unsigned short bottomLeft;
	unsigned short bottomRight;

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < slices; j++){

			if(j < slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = topLeft + 1;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = bottomLeft + 1;

			}
			else if(j == slices - 1){
				topLeft = (i) * (slices) + j;
				topRight = (i) * (slices) + 0;
				bottomLeft = (i + 1) * slices + j;
				bottomRight = (i + 1) * slices + 0;
			}


			element[index + 0] = topLeft;
			element[index + 1] = bottomLeft;
			element[index + 2] = topRight;

			element[index + 3] = topRight;
			element[index + 4] = bottomLeft;
			element[index + 5] = bottomRight;

			index = index + 6;

		}	
	}

	return(index);
}