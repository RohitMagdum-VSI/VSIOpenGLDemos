var canvas = null;
var gl = null;
var bIsFullScreen = false;
var canvas_original_width = 0;
var canvas_original_height = 0;



const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3,
};


//For Starting Animation we need requestAnimationFrame()

var requestAnimationFrame = 
	window.requestAnimationFrame || 
	window.webkitRequestAnimationFrame ||
	window.mozRequestAnimationFrame || 
	window.oRequestAnimationFrame || 
	window.msRequestAnimationFrame || 
	null;

//For Stoping Animation we need cancelAnimationFrame()

var cancelAnimationFrame = 
	window.cancelAnimationFrame || 
	window.webkitCancelRequestAnimationFrame || window.webkitCancelAnimationFrame || 
	window.mozCancelRequestAnimationFrame || window.mozCancelAnimationFrame ||
	window.oCancelRequestAnimationFrame || window.oCancelAnimationFrame ||
	window.msCancelRequestAnimationFrame || window.msCancelAnimationFrame ||
	null;


//For Shader
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//For Uniform
var modelMatrix_Uniform;
var viewMatrix_Uniform;
var projectionMatrix_Uniform;


//For Projection Matrix;
var perspectiveProjectionMatrix;



//For Sphere
var vao_Sphere;
var vbo_Sphere_Position;
var vbo_Sphere_Normal;
var vbo_Sphere_Index;

const STACK = 30;
const SLICES = 30;

var sphere_Position;
var sphere_Normal;
var sphere_Index;
var angle_Sphere = 0.0;


//For Light Uniform
var la_Uniform;
var ld_Uniform;
var ls_Uniform;
var lightPosition_Uniform;

var ka_Uniform;
var kd_Uniform;
var ks_Uniform;
var shininess_Uniform;



//For Lights
var lightAmbient = [0.0, 0.0, 0.0];
var lightDiffuse =[1.0, 1.0, 1.0];
var lightSpecular = [1.0, 1.0, 1.0];
var lightPosition = [100.0, 100.0, 100.0, 1.0];
var bLights = false;


//For Material
var materialAmbient = [0.0, 0.0, 0.0];
var materialDiffuse = [1.0, 1.0, 1.0];
var materialSpecular = [1.0, 1.0, 1.0];
var materialShininess = 128.0;


function main(){

	canvas = document.getElementById("22-PerFragmentADSLight-RRJ");
	if(!canvas){
		console.log("Obtaining Canvas Failed!!\n");
		return;
	}
	else
		console.log("Canvas Obtained!!\n");

	canvas_original_width = canvas.width;
	canvas_original_height = canvas.height;

	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", resize, false);

	initialize();



	resize();
	draw();
}

function toggleFullScreen(){

	var fullscreen_element = 
		document.fullscreenElement ||
		document.webkitFullscreenElement ||
		document.mozFullScreenElement ||
		document.msFullscreenElement || 
		document.oFullscreenElement ||
		null;


	if(fullscreen_element == null){

		if(canvas.requestFullscreen)
			canvas.requestFullscreen();
		else if(canvas.webkitRequestFullscreen)
			canvas.webkitRequestFullscreen();
		else if(canvas.mozRequestFullScreen)
			canvas.mozRequestFullScreen();
		else if(canvas.msRequestFullscreen)
			canvas.msRequestFullscreen();
		
		bIsFullScreen = true;
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

		bIsFullScreen = false;
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
			if(bLights == false)
				bLights = true;
			else
				bLights = false;

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

	gl = canvas.getContext("webgl2");
	if(gl == null){
		console.log("Obtaining Context Failed!!\n");
		return;
	}
	else 
		console.log("Context Obtained!!\n");


	gl.viewportWidth = canvas.width;
	gl.viewportHeight = canvas.height;


	vertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
	var vertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec3 vNormal;" +

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
				
			

			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
			
		"}";

	gl.shaderSource(vertexShaderObject, vertexShaderSourceCode);
	gl.compileShader(vertexShaderObject);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var fragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +

		"in vec3 outLightDirection;" +
		"in vec3 outNormal;" +
		"in vec3 outViewer;" +


		"uniform vec3 u_la;" +
		"uniform vec3 u_ld;" +
		"uniform vec3 u_ls;" +

		
		"uniform vec3 u_ka;" +
		"uniform vec3 u_kd;" +
		"uniform vec3 u_ks;" +
		"uniform float u_shininess;" +

		"uniform int u_LKey;" +



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


			"FragColor = vec4(PhongLight, 1.0);" +
		"}";

	gl.shaderSource(fragmentShaderObject, fragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject = gl.createProgram();

	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");


	gl.linkProgram(shaderProgramObject);

	
	if(!gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	modelMatrix_Uniform = gl.getUniformLocation(shaderProgramObject, "u_model_matrix");
	viewMatrix_Uniform = gl.getUniformLocation(shaderProgramObject, "u_view_matrix");
	projectionMatrix_Uniform = gl.getUniformLocation(shaderProgramObject, "u_projection_matrix");

	la_Uniform = gl.getUniformLocation(shaderProgramObject, "u_la");
	ld_Uniform = gl.getUniformLocation(shaderProgramObject, "u_ld");
	ls_Uniform = gl.getUniformLocation(shaderProgramObject, "u_ls");
	lightPosition_Uniform = gl.getUniformLocation(shaderProgramObject, "u_light_position");
	

	ka_Uniform = gl.getUniformLocation(shaderProgramObject, "u_ka");
	kd_Uniform = gl.getUniformLocation(shaderProgramObject, "u_kd");
	ks_Uniform = gl.getUniformLocation(shaderProgramObject, "u_ks");
	shininess_Uniform = gl.getUniformLocation(shaderProgramObject, "u_shininess");

	LKeyPress_Uniform = gl.getUniformLocation(shaderProgramObject, "u_LKey");



	/********** Sphere Position and Normal **********/

	myMakeSphere(2.0, STACK, SLICES);

	vao_Sphere = gl.createVertexArray();
	gl.bindVertexArray(vao_Sphere);

		/********** Position **********/
		vbo_Sphere_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Sphere_Position);
		gl.bufferData(gl.ARRAY_BUFFER, sphere_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Normal **********/
		vbo_Sphere_Normal = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Sphere_Normal);
		gl.bufferData(gl.ARRAY_BUFFER, sphere_Normal, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Index **********/
		vbo_Sphere_Index = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphere_Index, gl.STATIC_DRAW);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	
	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.disable(gl.CULL_FACE);
	
	gl.clearDepth(1.0);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix = mat4.create();
}



function uninitialize(){


	if(vbo_Sphere_Index){
		gl.deleteBuffer(vbo_Sphere_Index);
		vbo_Sphere_Index = 0;
	}

	if(vbo_Sphere_Normal){
		gl.deleteBuffer(vbo_Sphere_Normal);
		vbo_Sphere_Normal = 0;
	}


	if(vbo_Sphere_Position){
		gl.deleteBuffer(vbo_Sphere_Position);
		vbo_Sphere_Position = 0;
	}

	if(vao_Sphere){
		gl.deleteVertexArray(vao_Sphere);
		vao_Sphere = 0;
	}

	if(shaderProgramObject){

		gl.useProgram(shaderProgramObject);

			if(fragmentShaderObject){
				gl.detachShader(shaderProgramObject, fragmentShaderObject);
				gl.deleteShader(fragmentShaderObject);
				fragmentShaderObject = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject){
				gl.detachShader(shaderProgramObject, vertexShaderObject);
				gl.deleteShader(vertexShaderObject);
				vertexShaderObject = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject);
		shaderProgramObject = 0;
	}
}

function resize(){

	if(bIsFullScreen == true){
		canvas.width = window.innerWidth;
		canvas.height = window.innerHeight;
	}
	else{
		canvas.width = canvas_original_width;
		canvas.height = canvas_original_height;
	}

	gl.viewport(0, 0, canvas.width, canvas.height);

	 mat4.identity(perspectiveProjectionMatrix);
	 mat4.perspective(perspectiveProjectionMatrix, 
	 				45.0,
	 				parseFloat(canvas.width) / parseFloat(canvas.height),
	 				0.1,
	 				100.0);
}



function draw(){

	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);



		
		/********** Sphere ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		mat4.rotateX(modelMatrix, modelMatrix, degToRad(90.0));
		//mat4.rotateY(modelMatrix, modelMatrix, degToRad(angle_Sphere));
		mat4.rotateZ(modelMatrix, modelMatrix, degToRad(angle_Sphere));
		
		gl.uniformMatrix4fv(modelMatrix_Uniform, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform, false, perspectiveProjectionMatrix)
		



		if(bLights == true){
			gl.uniform1i(LKeyPress_Uniform, 1);

			gl.uniform3fv(la_Uniform, lightAmbient);
			gl.uniform3fv(ld_Uniform, lightDiffuse);
			gl.uniform3fv(ls_Uniform, lightSpecular);
			gl.uniform4fv(lightPosition_Uniform, lightPosition);

			gl.uniform3fv(ka_Uniform, materialAmbient);
			gl.uniform3fv(kd_Uniform, materialDiffuse);
			gl.uniform3fv(ks_Uniform, materialSpecular);
			gl.uniform1f(shininess_Uniform, materialShininess);	

		}
		else
			gl.uniform1i(LKeyPress_Uniform, 0);


		gl.bindVertexArray(vao_Sphere);

			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index);
			gl.drawElements(gl.TRIANGLES, (STACK) * (SLICES) * 6, gl.UNSIGNED_SHORT, 0);

		gl.bindVertexArray(null);

	


	gl.useProgram(null);

	update();

	requestAnimationFrame(draw, canvas);
}

function update(){

	angle_Sphere = angle_Sphere + 0.3;

	if(angle_Sphere > 360.0)
		angle_Sphere = 0.0;
}







function myMakeSphere(fRadius, iStack, iSlices){


	sphere_Position = new Float32Array(iStack * iSlices * 3);
	sphere_Texcoord = new Float32Array(iStack * iSlices * 2);
	sphere_Normal = new Float32Array(iStack * iStack * 3);	
	sphere_Index = new Uint16Array((iStack) * (iSlices) * 6);

	var longitude;
	var latitude;
	var factorLat = (2.0 * Math.PI) / (iStack);
	var factorLon = Math.PI / (iSlices-1);

	for(var i = 0; i < iStack; i++){
		
		latitude = -Math.PI + i * factorLat;


		for(var j = 0; j < iSlices; j++){

			longitude = (Math.PI) - j * factorLon;

			//console.log(i + "/" + j + ": " + latitude + "/" + longitude);

			var x = fRadius * Math.sin(longitude) * Math.cos((latitude));
			var y = fRadius * Math.sin(longitude) * Math.sin((latitude));
			var z = fRadius * Math.cos((longitude));

			sphere_Position[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Position[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Position[(i * iSlices * 3)+ (j * 3) + 2] = z;

			//zconsole.log(i + "/" + j + "   " + x + "/" + y + "/" + z);

			
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Normal[(i * iSlices * 3)+ (j * 3) + 2] = z;

		}
	}


	var index = 0;
 	for(var i = 0; i < iStack ; i++){
 		for(var j = 0; j < iSlices ; j++){


 			if(i == (iStack - 1)){

 				var topLeft = (i * iSlices) + j;
	 			var bottomLeft = ((0) * iSlices) +(j);
	 			var topRight = topLeft + 1;
	 			var bottomRight = bottomLeft + 1;


	 			sphere_Index[index] = topLeft;
	 			sphere_Index[index + 1] = bottomLeft;
	 			sphere_Index[index + 2] = topRight;

	 			sphere_Index[index + 3] = topRight;
	 			sphere_Index[index + 4] = bottomLeft;
	 			sphere_Index[index + 5] = bottomRight;

 			}
 			else{

	 			var topLeft = (i * iSlices) + j;
	 			var bottomLeft = ((i + 1) * iSlices) +(j);
	 			var topRight = topLeft + 1;
	 			var bottomRight = bottomLeft + 1;


	 			sphere_Index[index] = topLeft;
	 			sphere_Index[index + 1] = bottomLeft;
	 			sphere_Index[index + 2] = topRight;

	 			sphere_Index[index + 3] = topRight;
	 			sphere_Index[index + 4] = bottomLeft;
	 			sphere_Index[index + 5] = bottomRight;
 			}

 			

 			index = index + 6;


 		}
 		

 	}
}


function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}