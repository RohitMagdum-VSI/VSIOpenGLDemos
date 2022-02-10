

vec4 my_gluProject(vec4 pos, mat4 mvMatrix, mat4 pMatrix, vec4 viewPort){

	vec4 matMul(mat4, vec4);

	vec4 worldPos;
	worldPos = matMul(mvMatrix, pos);

	// fprintf(gpFile, "world: %f, %f, %f, %f\n",
	// 	worldPos[0],
	// 	worldPos[1],
	// 	worldPos[2],
	// 	worldPos[3]);



	vec4 clipSpaceCoord;
	clipSpaceCoord = matMul(pMatrix, worldPos);


	// fprintf(gpFile, "clipPos : %f, %f, %f, %f\n", 
	// 	clipSpaceCoord[0],
	// 	clipSpaceCoord[1],
	// 	clipSpaceCoord[2],
	// 	clipSpaceCoord[3]);


	if(clipSpaceCoord[3] == 0){
		fprintf(gpFile, "my_gluProject: clipSpaceCoord[3] == 0\n");
		return(vec4(0.0f));
	}

	clipSpaceCoord[0] /= clipSpaceCoord[3];
	clipSpaceCoord[1] /= clipSpaceCoord[3];
	clipSpaceCoord[2] /= clipSpaceCoord[3];

	// fprintf(gpFile, "NDC : %f, %f, %f, %f\n", 
	// 	clipSpaceCoord[0],
	// 	clipSpaceCoord[1],
	// 	clipSpaceCoord[2],
	// 	clipSpaceCoord[3]);


	clipSpaceCoord[0] = clipSpaceCoord[0] * 0.5f + 0.5f;
	clipSpaceCoord[1] = clipSpaceCoord[1] * 0.5f + 0.5f;
	clipSpaceCoord[2] = clipSpaceCoord[2] * 0.5f + 0.5f;

	// fprintf(gpFile, "0-1 : %f, %f, %f, %f\n", 
	// 	clipSpaceCoord[0],
	// 	clipSpaceCoord[1],
	// 	clipSpaceCoord[2],
	// 	clipSpaceCoord[3]);

	clipSpaceCoord[0] = clipSpaceCoord[0] * viewPort[2] + viewPort[0];
	clipSpaceCoord[1] = clipSpaceCoord[1] * viewPort[3] + viewPort[1];

	// fprintf(gpFile, "WndCoord : %f, %f, %f, %f\n", 
	// 	clipSpaceCoord[0],
	// 	clipSpaceCoord[1],
	// 	clipSpaceCoord[2],
	// 	clipSpaceCoord[3]);


	// for(int i = 0; i < 4; i++){
		
	// 	for(int j = 0; j < 4; j++){
	// 		fprintf(gpFile, "%f ", mvMatrix[i][j]);
	// 	}
		
	// 	fprintf(gpFile, "\n");
	// }


	return(clipSpaceCoord);


	// for(int i = 0; i < 4; i++)
	// 	fprintf(gpFile, "%f\n", worldPos[i]);


	

}	


vec4 matMul(mat4 mvMatrix, vec4 pos){

	vec4 ans = vec4(0.0f);

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){

			ans[i] += (mvMatrix[j][i] * pos[j]);

			// fprintf(gpFile, "ans[%d] : %f : %f : %f\n", i, ans[i], mvMatrix[j][i] * pos[j], mvMatrix[j][i]);

		}
	}

	return(ans);
}