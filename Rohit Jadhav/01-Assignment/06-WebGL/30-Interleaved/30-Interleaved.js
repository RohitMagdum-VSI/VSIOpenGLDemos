var canvas_RRJ = null;
var gl_RRJ = null;
var bIsFullScreen_RRJ = false;
var canvas_original_width_RRJ = 0;
var canvas_original_height_RRJ = 0;



const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3,
};


//For Starting Animation we need requestAnimationFrame()

var requestAnimationFrame_RRJ = 
	window.requestAnimationFrame || 
	window.webkitRequestAnimationFrame ||
	window.mozRequestAnimationFrame || 
	window.oRequestAnimationFrame || 
	window.msRequestAnimationFrame || 
	null;

//For Stoping Animation we need cancelAnimationFrame()

var cancelAnimationFrame_RRJ = 
	window.cancelAnimationFrame || 
	window.webkitCancelRequestAnimationFrame || window.webkitCancelAnimationFrame || 
	window.mozCancelRequestAnimationFrame || window.mozCancelAnimationFrame ||
	window.oCancelRequestAnimationFrame || window.oCancelAnimationFrame ||
	window.msCancelRequestAnimationFrame || window.msCancelAnimationFrame ||
	null;


//For Shader
var vertexShaderObject_RRJ;
var fragmentShaderObject_RRJ;
var shaderProgramObject_RRJ;

//For Uniform
var modelMatrix_Uniform_RRJ;
var viewMatrix_Uniform_RRJ;
var projectionMatrix_Uniform_RRJ;


//For Projection Matrix;
var perspectiveProjectionMatrix_RRJ;



//For Sphere
var vao_Cube_RRJ;
var vbo_Cube_Position_RRJ;
var vbo_Cube_Normal_RRJ;
var angle_Cube_RRJ = 0.0;


//For Light Uniform
var la_Uniform_RRJ;
var ld_Uniform_RRJ;
var ls_Uniform_RRJ;
var lightPosition_Uniform_RRJ;
var LKeyPress_Uniform_RRJ;


var ka_Uniform_RRJ;
var kd_Uniform_RRJ;
var ks_Uniform_RRJ;
var shininess_Uniform_RRJ;



//For Lights
var lightAmbient_RRJ = [0.250, 0.250, 0.250];
var lightDiffuse_RRJ =[1.0, 1.0, 1.0];
var lightSpecular_RRJ = [1.0, 1.0, 1.0];
var lightPosition_RRJ = [100.0, 100.0, 100.0, 1.0];
var bLights_RRJ = false;


//For Material
var materialAmbient_RRJ = [0.250, 0.250, 0.250];
var materialDiffuse_RRJ = [1.0, 1.0, 1.0];
var materialSpecular_RRJ = [1.0, 1.0, 1.0];
var materialShininess_RRJ = 128.0;


//For Texture
var samplerUniform_RRJ;
var textureMarble_RRJ;



function main(){

	canvas_RRJ = document.getElementById("30-Interleaved-RRJ");
	if(!canvas_RRJ){
		console.log("Obtaining Canvas Failed!!\n");
		return;
	}
	else
		console.log("Canvas Obtained!!\n");

	canvas_original_width_RRJ = canvas_RRJ.width;
	canvas_original_height_RRJ = canvas_RRJ.height;

	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", resize, false);

	initialize();



	resize();
	draw();
}

function toggleFullScreen(){

	var fullscreen_element_RRJ = 
		document.fullscreenElement ||
		document.webkitFullscreenElement ||
		document.mozFullScreenElement ||
		document.msFullscreenElement || 
		document.oFullscreenElement ||
		null;


	if(fullscreen_element_RRJ == null){

		if(canvas_RRJ.requestFullscreen)
			canvas_RRJ.requestFullscreen();
		else if(canvas_RRJ.webkitRequestFullscreen)
			canvas_RRJ.webkitRequestFullscreen();
		else if(canvas_RRJ.mozRequestFullScreen)
			canvas_RRJ.mozRequestFullScreen();
		else if(canvas_RRJ.msRequestFullscreen)
			canvas_RRJ.msRequestFullscreen();
		
		bIsFullScreen_RRJ = true;
	}
	else{
		if(document.exitFullscreen)
			document.exitFullscreen();
		else if(document.webkitExitFullscreen)
			document.webkitExitFullscreen();
		else if(document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if(document.msExitFullscreen)
			document.msExitFullscreen();

		bIsFullScreen_RRJ = false;
	}
}


function keyDown(event){

	switch(event.keyCode){
		case 27:
			uninitialize();
			window.close();
			break;

		//L 
		case 76:
			if(bLights_RRJ == false)
				bLights_RRJ = true;
			else
				bLights_RRJ = false;

			break;


		//F
		case 70:
			toggleFullScreen();
			break;

	}


}

function mouseDown(){

}



function initialize(){

	gl_RRJ = canvas_RRJ.getContext("webgl2");
	if(gl_RRJ == null){
		console.log("Obtaining Context Failed!!\n");
		return;
	}
	else 
		console.log("Context Obtained!!\n");


	gl_RRJ.viewportWidth = canvas_RRJ.width;
	gl_RRJ.viewportHeight = canvas_RRJ.height;


	vertexShaderObject_RRJ = gl_RRJ.createShader(gl_RRJ.VERTEX_SHADER);
	var vertexShaderSourceCode_RRJ = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec3 vNormal;" +
		"in vec4 vColor;" +
		"in vec2 vTex;" +
		"out vec4 outColor;" +
		"out vec2 outTex;" + 


		"uniform vec4 u_light_position;" +
 
		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_projection_matrix;" +

		"out vec3 outViewer;" + 
		"out vec3 outLightDirection;" +
		"out vec3 outNormal;" + 		

		"void main() {" +

				


			"vec3 normalizeNormals;" + 
			"normalizeNormals = vNormal;" +
			"normalizeNormals = normalize(normalizeNormals);" +


			"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +

			"outLightDirection = vec3(u_light_position - eyeCoordinate);" +

			"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
			"outNormal = vec3(normalMatrix * normalizeNormals);" +

			"outViewer = vec3(-eyeCoordinate.xyz);" +
				
			
			"outColor = vColor;" +
			"outTex = vTex;" +
			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
			
		"}";

	gl_RRJ.shaderSource(vertexShaderObject_RRJ, vertexShaderSourceCode_RRJ);
	gl_RRJ.compileShader(vertexShaderObject_RRJ);

	var shaderCompileStatus_RRJ = gl_RRJ.getShaderParameter(vertexShaderObject_RRJ, gl_RRJ.COMPILE_STATUS);

	if(shaderCompileStatus_RRJ == false){
		var error = gl_RRJ.getShaderInfoLog(vertexShaderObject_RRJ);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_RRJ = gl_RRJ.createShader(gl_RRJ.FRAGMENT_SHADER);
	var fragmentShaderSourceCode_RRJ = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +

		"in vec3 outLightDirection;" +
		"in vec3 outNormal;" +
		"in vec3 outViewer;" +

		"in vec4 outColor;" +
		"in vec2 outTex;" +


		"uniform vec3 u_la;" +
		"uniform vec3 u_ld;" +
		"uniform vec3 u_ls;" +

		
		"uniform vec3 u_ka;" +
		"uniform vec3 u_kd;" +
		"uniform vec3 u_ks;" +
		"uniform float u_shininess;" +

		"uniform int u_LKey;" +

		"uniform sampler2D u_sampler;" +


		"out vec4 FragColor;" +
		"void main(){" +

			"vec3 PhongLight;" +

			"if(u_LKey == 1){" +

				"vec3 normalizeLightDirection = normalize(outLightDirection);" +
				"vec3 normalizeNormalVector = normalize(outNormal);" +
				"float S_Dot_N = max(dot(normalizeLightDirection, normalizeNormalVector), 0.0);" +

				"vec3 normalizeViewer = normalize(outViewer);" +
				"vec3 reflection = reflect(-normalizeLightDirection, normalizeNormalVector);" +
				"float R_Dot_V = max(dot(reflection, normalizeViewer), 0.0);" +

				"vec3 ambient = u_la * u_ka;" +
				"vec3 diffuse = u_ld * u_kd * S_Dot_N;" +
				"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
				"PhongLight = ambient + diffuse + specular;" +

			"}" +
			"else {" + 
				"PhongLight = vec3(1.0, 1.0, 1.0);" +
			"}" +

			"vec4 tex = texture(u_sampler, outTex);" +
			"vec4 light = vec4(PhongLight, 1.0);" +
			
			"FragColor = tex * outColor * light;" +
		"}";

	gl_RRJ.shaderSource(fragmentShaderObject_RRJ, fragmentShaderSourceCode_RRJ);
	gl_RRJ.compileShader(fragmentShaderObject_RRJ);

	shaderCompileStatus_RRJ = gl_RRJ.getShaderParameter(fragmentShaderObject_RRJ, gl_RRJ.COMPILE_STATUS);

	if(shaderCompileStatus_RRJ == false){
		var error = gl_RRJ.getShaderInfoLog(fragmentShaderObject_RRJ);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_RRJ = gl_RRJ.createProgram();

	gl_RRJ.attachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	gl_RRJ.attachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	gl_RRJ.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl_RRJ.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");
	gl_RRJ.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");
	gl_RRJ.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTex");


	gl_RRJ.linkProgram(shaderProgramObject_RRJ);

	
	if(!gl_RRJ.getProgramParameter(shaderProgramObject_RRJ, gl_RRJ.LINK_STATUS)){
		var error = gl_RRJ.getProgramInfoLog(shaderProgramObject_RRJ);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	modelMatrix_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
	viewMatrix_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
	projectionMatrix_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");

	la_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_la");
	ld_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_ld");
	ls_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_ls");
	lightPosition_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_light_position");
	

	ka_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_ka");
	kd_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_kd");
	ks_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_ks");
	shininess_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_shininess");

	LKeyPress_Uniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_LKey");

	samplerUniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_sampler");




	/********** Sphere Position and Normal **********/


	/********** Cube **********/
	textureMarble_RRJ = gl_RRJ.createTexture();
	textureMarble_RRJ.image = new Image();
	textureMarble_RRJ.image.src = "marble.png";
	textureMarble_RRJ.image.onload = function(){
		gl_RRJ.bindTexture(gl_RRJ.TEXTURE_2D, textureMarble_RRJ);
		gl_RRJ.pixelStorei(gl_RRJ.UNPACK_FLIP_Y_WEBGL, true);
		gl_RRJ.texParameteri(gl_RRJ.TEXTURE_2D, gl_RRJ.TEXTURE_MAG_FILTER, gl_RRJ.LINEAR);
		gl_RRJ.texParameteri(gl_RRJ.TEXTURE_2D, gl_RRJ.TEXTURE_MIN_FILTER, gl_RRJ.LINEAR);
		gl_RRJ.texImage2D(gl_RRJ.TEXTURE_2D, 0, gl_RRJ.RGBA, gl_RRJ.RGBA, gl_RRJ.UNSIGNED_BYTE, textureMarble_RRJ.image);
		gl_RRJ.bindTexture(gl_RRJ.TEXTURE_2D, null);
	}



	var cube_Position_RRJ = new Float32Array([


						//vPosition 	//vColor 		//vNormal 		//vTex
						1.0, 1.0, -1.0,	1.0, 0.0, 0.0,	0.0, 1.0, 0.0,	1.0, 1.0,	   
						-1.0, 1.0, -1.0,	1.0, 0.0, 0.0,	0.0, 1.0, 0.0,	0.0, 1.0, 	 
						-1.0, 1.0, 1.0,	1.0, 0.0, 0.0,	0.0, 1.0, 0.0,	0.0, 0.0,	
						1.0, 1.0, 1.0,	1.0, 0.0, 0.0,	0.0, 1.0, 0.0,	1.0, 0.0,
						//Bottom
						1.0, -1.0, -1.0,	0.0, 1.0, 0.0,	0.0, -1.0, 0.0,	1.0, 1.0,
						-1.0, -1.0, -1.0,	0.0, 1.0, 0.0,	0.0, -1.0, 0.0, 	0.0, 1.0,	
						-1.0, -1.0, 1.0,	0.0, 1.0, 0.0,	0.0, -1.0, 0.0,	0.0, 0.0,
						1.0, -1.0, 1.0,	0.0, 1.0, 0.0,	0.0, -1.0, 0.0,	1.0, 0.0,
						//Front
						1.0, 1.0, 1.0,	0.0, 0.0, 1.0,	0.0, 0.0, 1.0,	1.0, 1.0,
						-1.0, 1.0, 1.0,	0.0, 0.0, 1.0,	0.0, 0.0, 1.0,	0.0, 1.0,
						-1.0, -1.0, 1.0,	0.0, 0.0, 1.0,	0.0, 0.0, 1.0,	0.0, 0.0,
						1.0, -1.0, 1.0,	0.0, 0.0, 1.0,	0.0, 0.0, 1.0,	1.0, 0.0,
						//Back
						1.0, 1.0, -1.0,	1.0, 1.0, 0.0,	0.0, 0.0, -1.0,	1.0, 1.0,
						-1.0, 1.0, -1.0,	1.0, 1.0, 0.0,	0.0, 0.0, -1.0,	0.0, 1.0,
						-1.0, -1.0, -1.0,	1.0, 1.0, 0.0,	0.0, 0.0, -1.0,	0.0, 0.0,
						1.0, -1.0, -1.0,	1.0, 1.0, 0.0,	0.0, 0.0, -1.0,	1.0, 0.0,
						//Right
						1.0, 1.0, -1.0,	0.0, 1.0, 1.0,	1.0, 0.0, 0.0,	1.0, 1.0,
						1.0, 1.0, 1.0,	0.0, 1.0, 1.0,	1.0, 0.0, 0.0,	0.0, 1.0,
						1.0, -1.0, 1.0,	0.0, 1.0, 1.0,	1.0, 0.0, 0.0,	0.0, 0.0,
						1.0, -1.0, -1.0,	0.0, 1.0, 1.0,	1.0, 0.0, 0.0,	1.0, 0.0,
						//Left
						-1.0, 1.0, 1.0,	1.0, 0.0, 1.0,	-1.0, 0.0, 0.0,	1.0, 1.0,
						-1.0, 1.0, -1.0,	1.0, 0.0, 1.0,	-1.0, 0.0, 0.0,	0.0, 1.0,
						-1.0, -1.0, -1.0,	1.0, 0.0, 1.0,	-1.0, 0.0, 0.0, 	0.0, 0.0,
						-1.0, -1.0, 1.0,	1.0, 0.0, 1.0,	-1.0, 0.0, 0.0,	1.0, 0.0,
					]);

	




	/********** Cube **********/
	vao_Cube_RRJ = gl_RRJ.createVertexArray();
	gl_RRJ.bindVertexArray(vao_Cube_RRJ);

		/********** Position **********/
		vbo_Cube_Position = gl_RRJ.createBuffer();
		gl_RRJ.bindBuffer(gl_RRJ.ARRAY_BUFFER, vbo_Cube_Position);
		gl_RRJ.bufferData(gl_RRJ.ARRAY_BUFFER, cube_Position_RRJ, gl_RRJ.STATIC_DRAW);
		gl_RRJ.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl_RRJ.FLOAT,
							false,
							11 * 4, 0 * 4);
		
		gl_RRJ.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl_RRJ.FLOAT,
							false,
							11 * 4, 3 * 4);
		
		gl_RRJ.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL,
							3,
							gl_RRJ.FLOAT,
							false,
							11 * 4, 6 * 4);
		
		gl_RRJ.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0,
							2,
							gl_RRJ.FLOAT,
							false,
							11 * 4,  9 * 4);
		
		gl_RRJ.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl_RRJ.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl_RRJ.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl_RRJ.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_TEXCOORD0);
		gl_RRJ.bindBuffer(gl_RRJ.ARRAY_BUFFER, null);


	
	gl_RRJ.bindVertexArray(null);

	
	gl_RRJ.enable(gl_RRJ.DEPTH_TEST);
	gl_RRJ.depthFunc(gl_RRJ.LEQUAL);

	gl_RRJ.disable(gl_RRJ.CULL_FACE);
	
	gl_RRJ.clearDepth(1.0);

	gl_RRJ.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix_RRJ = mat4.create();
}



function uninitialize(){



	if(vbo_Cube_Position){
		gl_RRJ.deleteBuffer(vbo_Cube_Position);
		vbo_Cube_Position = 0;
	}

	if(vao_Cube_RRJ){
		gl_RRJ.deleteVertexArray(vao_Cube_RRJ);
		vao_Cube_RRJ = 0;
	}


	if(shaderProgramObject_RRJ){

		gl_RRJ.useProgram(shaderProgramObject_RRJ);

			if(fragmentShaderObject_RRJ){
				gl_RRJ.detachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
				gl_RRJ.deleteShader(fragmentShaderObject_RRJ);
				fragmentShaderObject_RRJ = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject_RRJ){
				gl_RRJ.detachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
				gl_RRJ.deleteShader(vertexShaderObject_RRJ);
				vertexShaderObject_RRJ = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl_RRJ.useProgram(null);
		gl_RRJ.deleteProgram(shaderProgramObject_RRJ);
		shaderProgramObject_RRJ = 0;
	}
}

function resize(){

	if(bIsFullScreen_RRJ == true){
		canvas_RRJ.width = window.innerWidth;
		canvas_RRJ.height = window.innerHeight;
	}
	else{
		canvas_RRJ.width = canvas_original_width_RRJ;
		canvas_RRJ.height = canvas_original_height_RRJ;
	}

	gl_RRJ.viewport(0, 0, canvas_RRJ.width, canvas_RRJ.height);

	 mat4.identity(perspectiveProjectionMatrix_RRJ);
	 mat4.perspective(perspectiveProjectionMatrix_RRJ, 
	 				45.0,
	 				parseFloat(canvas_RRJ.width) / parseFloat(canvas_RRJ.height),
	 				0.1,
	 				100.0);
}



function draw(){

	var modelMatrix_RRJ = mat4.create();
	var viewMatrix_RRJ = mat4.create();

	gl_RRJ.clear(gl_RRJ.COLOR_BUFFER_BIT | gl_RRJ.DEPTH_BUFFER_BIT);


	gl_RRJ.useProgram(shaderProgramObject_RRJ);



		
		/********** Cube ***********/
		mat4.identity(modelMatrix_RRJ);
		mat4.identity(viewMatrix_RRJ);
		mat4.translate(modelMatrix_RRJ, modelMatrix_RRJ, [0.0, 0.0, -5.0]);
		mat4.rotateX(modelMatrix_RRJ, modelMatrix_RRJ, degToRad(angle_Cube_RRJ));
		mat4.rotateY(modelMatrix_RRJ, modelMatrix_RRJ, degToRad(angle_Cube_RRJ));
		mat4.rotateZ(modelMatrix_RRJ, modelMatrix_RRJ, degToRad(angle_Cube_RRJ));
		
		gl_RRJ.uniformMatrix4fv(modelMatrix_Uniform_RRJ, false, modelMatrix_RRJ);
		gl_RRJ.uniformMatrix4fv(viewMatrix_Uniform_RRJ, false, viewMatrix_RRJ);
		gl_RRJ.uniformMatrix4fv(projectionMatrix_Uniform_RRJ, false, perspectiveProjectionMatrix_RRJ);



		if(bLights_RRJ == true){
			gl_RRJ.uniform1i(LKeyPress_Uniform_RRJ, 1);

			gl_RRJ.uniform3fv(la_Uniform_RRJ, lightAmbient_RRJ);
			gl_RRJ.uniform3fv(ld_Uniform_RRJ, lightDiffuse_RRJ);
			gl_RRJ.uniform3fv(ls_Uniform_RRJ, lightSpecular_RRJ);
			gl_RRJ.uniform4fv(lightPosition_Uniform_RRJ, lightPosition_RRJ);

			gl_RRJ.uniform3fv(ka_Uniform_RRJ, materialAmbient_RRJ);
			gl_RRJ.uniform3fv(kd_Uniform_RRJ, materialDiffuse_RRJ);
			gl_RRJ.uniform3fv(ks_Uniform_RRJ, materialSpecular_RRJ);
			gl_RRJ.uniform1f(shininess_Uniform_RRJ, materialShininess_RRJ);	

		}
		else
			gl_RRJ.uniform1i(LKeyPress_Uniform_RRJ, 0);


		gl_RRJ.bindTexture(gl_RRJ.TEXTURE_2D, textureMarble_RRJ);
		gl_RRJ.uniform1i(samplerUniform_RRJ, 0);

		gl_RRJ.bindVertexArray(vao_Cube_RRJ);

			gl_RRJ.drawArrays(gl_RRJ.TRIANGLE_FAN, 0, 4);
			gl_RRJ.drawArrays(gl_RRJ.TRIANGLE_FAN, 4, 4);
			gl_RRJ.drawArrays(gl_RRJ.TRIANGLE_FAN, 8, 4);
			gl_RRJ.drawArrays(gl_RRJ.TRIANGLE_FAN, 12, 4);
			gl_RRJ.drawArrays(gl_RRJ.TRIANGLE_FAN, 16, 4);
			gl_RRJ.drawArrays(gl_RRJ.TRIANGLE_FAN, 20, 4);

		gl_RRJ.bindVertexArray(null);

	


	gl_RRJ.useProgram(null);

	update();

	requestAnimationFrame_RRJ(draw, canvas_RRJ);
}

function update(){

	angle_Cube_RRJ = angle_Cube_RRJ - 0.7;

	if(angle_Cube_RRJ < 0.0)
		angle_Cube_RRJ = 360.0;
}








function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}