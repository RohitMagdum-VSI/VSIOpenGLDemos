var canvas;
var gl;

//For Uniform
var mvpUniform_CubeMap;
var samplerUniform_CubeMap;

//For Projection
var perspectiveProjectionMatrix_Cube;
	
var vertexShaderObject_CubeMap;
var fragmentShaderObject_CubeMap;	
var shaderProgramObject_CubeMap;

//For Cube
var vao_Cube_CubeMap;
var vbo_Cube_Position_CubeMap;
var vbo_Cube_Texcoord_CubeMap;
var angle_Cube = 360.0;
var texture_CubeMap;



function initialize_CubeMap(){

	
	vertexShaderObject_CubeMap = gl.createShader(gl.VERTEX_SHADER);
	var vertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec2 vTex;" +
		"uniform mat4 u_mvp_matrix;" +
		"out vec3 outTex;" +
		"void main() {" +
			"gl_Position = u_mvp_matrix * vPosition;" +
			"outTex = vPosition.xyz;" +
			"outTex.y = -outTex.y;" +
		"}";

	gl.shaderSource(vertexShaderObject_CubeMap, vertexShaderSourceCode);
	gl.compileShader(vertexShaderObject_CubeMap);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_CubeMap, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_CubeMap);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_CubeMap = gl.createShader(gl.FRAGMENT_SHADER);
	var fragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"precision highp samplerCube;" +
		"in vec3 outTex;" +

		"uniform samplerCube u_sampler;" +
		"out vec4 FragColor;" +

		"void main(){" +
			"FragColor = texture(u_sampler, outTex);" +
			// "FragColor = vec4(1.0f);" +
		"}";

	gl.shaderSource(fragmentShaderObject_CubeMap, fragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject_CubeMap);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_CubeMap, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_CubeMap);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_CubeMap = gl.createProgram();

	gl.attachShader(shaderProgramObject_CubeMap, vertexShaderObject_CubeMap);
	gl.attachShader(shaderProgramObject_CubeMap, fragmentShaderObject_CubeMap);

	gl.bindAttribLocation(shaderProgramObject_CubeMap, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_CubeMap, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");


	gl.linkProgram(shaderProgramObject_CubeMap);

	
	if(!gl.getProgramParameter(shaderProgramObject_CubeMap, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject_CubeMap);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform_CubeMap = gl.getUniformLocation(shaderProgramObject_CubeMap, "u_mvp_matrix");
	samplerUniform_CubeMap = gl.getUniformLocation(shaderProgramObject_CubeMap, "u_sampler");






	LoadCubeMap();
	


	var cube_Position = new Float32Array([
						1.0, 1.0, -1.0,
						-1.0, 1.0, -1.0,
						-1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						//Bottom
						1.0, -1.0, -1.0,
						-1.0, -1.0, -1.0,
						-1.0, -1.0, 1.0,
						1.0, -1.0, 1.0,
						//Front
						1.0, 1.0, 1.0,
						-1.0, 1.0, 1.0,
						-1.0, -1.0, 1.0,
						1.0, -1.0, 1.0,
						//Back
						1.0, 1.0, -1.0,
						-1.0, 1.0, -1.0,
						-1.0, -1.0, -1.0,
						1.0, -1.0, -1.0,
						//Right
						1.0, 1.0, -1.0,
						1.0, 1.0, 1.0,
						1.0, -1.0, 1.0,
						1.0, -1.0, -1.0,
						//Left
						-1.0, 1.0, 1.0, 
						-1.0, 1.0, -1.0, 
						-1.0, -1.0, -1.0, 
						-1.0, -1.0, 1.0,
					]);

	
	var cube_Texcoord = new Float32Array([
						//Top
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Back
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Face
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Back
						1.0, 1.0, 
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Right
						1.0, 1.0,
						0.0, 1.0,
						0.0, 0.0,
						1.0, 0.0,
						//Left
						1.0, 1.0,
						0.0, 1.0, 
						0.0, 0.0,
						1.0, 0.0,
					]);




	/********** Cube **********/
	vao_Cube_CubeMap = gl.createVertexArray();
	gl.bindVertexArray(vao_Cube_CubeMap);

		/********** Position **********/
		vbo_Cube_Position_CubeMap = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Cube_Position_CubeMap);
		gl.bufferData(gl.ARRAY_BUFFER, cube_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Texcoord ***********/
		vbo_Cube_Texcoord_CubeMap = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Cube_Texcoord_CubeMap);
		gl.bufferData(gl.ARRAY_BUFFER, cube_Texcoord, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, 
							2,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.disable(gl.CULL_FACE);
	
	gl.clearDepth(1.0);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix_Cube = mat4.create();
}

function LoadCubeMap(){


	var left, right, top, bottom, front, back;


	/********** Cube **********/
	texture_CubeMap = gl.createTexture();
	// gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);


	// gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	// gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	// gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	// gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	// gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);


	

	right = new Image();
	right.src = "05-CubeMap/skybox/right.jpg";
	right.onload = function(){
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, right);
	}


	left = new Image();
	left.src = "05-CubeMap/skybox/left.jpg";
	left.onload = function(){
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, left);
	}

	top = new Image();
	top.src = "05-CubeMap/skybox/top.jpg";
	top.onload = function(){
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, top);
	}

	bottom = new Image();
	bottom.src = "05-CubeMap/skybox/bottom.jpg";
	bottom.onload = function(){
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, bottom);
	}


	front = new Image();
	front.src = "05-CubeMap/skybox/front.jpg";
	front.onload = function(){
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, front);
	}

	back = new Image();
	back.src = "05-CubeMap/skybox/back.jpg";
	back.onload = function(){
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, back);
	}

	gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);

	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

	// gl.generateMipmap(gl.TEXTURE_CUBE_MAP);

	gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);




}



function uninitialize(){

	if(texture_CubeMap){
		gl.deleteTexture(texture_CubeMap);
		texture_CubeMap = 0;
	}

	if(vbo_Cube_Texcoord_CubeMap){
		gl.deleteBuffer(vbo_Cube_Texcoord_CubeMap);
		vbo_Cube_Texcoord_CubeMap = 0;
	}

	if(vbo_Cube_Position_CubeMap){
		gl.deleteBuffer(vbo_Cube_Position_CubeMap);
		vbo_Cube_Position_CubeMap = 0;
	}

	if(vao_Cube_CubeMap){
		gl.deleteVertexArray(vao_Cube_CubeMap);
		vao_Cube_CubeMap = 0;
	}



	if(shaderProgramObject_CubeMap){

		gl.useProgram(shaderProgramObject_CubeMap);

			if(fragmentShaderObject_CubeMap){
				gl.detachShader(shaderProgramObject_CubeMap, fragmentShaderObject_CubeMap);
				gl.deleteShader(fragmentShaderObject_CubeMap);
				fragmentShaderObject_CubeMap = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject_CubeMap){
				gl.detachShader(shaderProgramObject_CubeMap, vertexShaderObject_CubeMap);
				gl.deleteShader(vertexShaderObject_CubeMap);
				vertexShaderObject_CubeMap = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_CubeMap);
		shaderProgramObject_CubeMap = 0;
	}
}


function draw_CubeMap(){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();


	//gl.clearColor(0.0, 0.0, 0.5, 1.0);
	//gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject_CubeMap);


		
		/********** Cube ***********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);
		mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, 0.0]);
		mat4.scale(modelViewMatrix, modelViewMatrix, [2048.0, 1024.0, 2048.0]);
		
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(180.0));

		// vec3.add(gCameraLookingAt, gCameraPosition, gCameraFront);
		// mat4.lookAt(global_viewMatrix, gCameraPosition, gCameraLookingAt, gCameraUp);
		

		mat4.perspective(perspectiveProjectionMatrix_Cube, 
					45.0,
					parseFloat(canvas.width) / parseFloat(canvas.height),
					0.1,
					4000.0);

		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix_Cube, global_viewMatrix);
		mat4.multiply(modelViewProjectionMatrix, modelViewProjectionMatrix, modelViewMatrix);
		
		gl.uniformMatrix4fv(mvpUniform_CubeMap, false, modelViewProjectionMatrix);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_CUBE_MAP, texture_CubeMap);
		gl.uniform1i(samplerUniform_CubeMap, 0);

		gl.bindVertexArray(vao_Cube_CubeMap);

			gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 4, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 8, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 12, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 16, 4);
			gl.drawArrays(gl.TRIANGLE_FAN, 20, 4);

		gl.bindVertexArray(null);


	gl.useProgram(null);

	update();

}

function update(){

	angle_Cube -= 0.10;


}