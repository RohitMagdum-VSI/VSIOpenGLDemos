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




//For Projection Matrix;
var perspectiveProjectionMatrix;
var modelMatrix;
var viewMatrix;

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


const PER_VERTEX = 1;
const PER_FRAGMENT = 2;
const X_ROT = 3;
const Y_ROT = 4;
const Z_ROT = 5;

var iWhichLight = PER_VERTEX;
var iWhichRotation = X_ROT;

//For Viewport Toggling
var iViewPortNo = 0;



//For Per Fragment Shader
var vertexShaderObject_PF;
var fragmentShaderObject_PF;
var shaderProgramObject_PF;


//For Uniform
var modelMatrix_Uniform_PF;
var viewMatrix_Uniform_PF;
var projectionMatrix_Uniform_PF;

//For Light Uniform
var la_Uniform_PF;
var ld_Uniform_PF;
var ls_Uniform_PF;
var lightPosition_Uniform_PF;

var ka_Uniform_PF;
var kd_Uniform_PF;
var ks_Uniform_PF;
var shininess_Uniform_PF;
var LKeyPress_Uniform_PF;



//For Per Vertex Shader
var vertexShaderObject_PV;
var fragmentShaderObject_PV;
var shaderProgramObject_PV;


//For Uniform
var modelMatrix_Uniform_PV;
var viewMatrix_Uniform_PV;
var projectionMatrix_Uniform_PV;

//For Light Uniform
var la_Uniform_PV;
var ld_Uniform_PV;
var ls_Uniform_PV;
var lightPosition_Uniform_PV;

var ka_Uniform_PV;
var kd_Uniform_PV;
var ks_Uniform_PV;
var shininess_Uniform_PV;
var LKeyPress_Uniform_PV;



//For Lights
var lightAmbient = [0.0, 0.0, 0.0];
var lightDiffuse =[1.0, 1.0, 1.0];
var lightSpecular = [1.0, 1.0, 1.0];
var lightPosition = [100.0, 100.0, 100.0, 1.0];
var bLights = false;



function main(){

	canvas = document.getElementById("26-24Spheres-RRJ");
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
			toggleFullScreen();
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
			iWhichLight = PER_FRAGMENT;
			break;

		
		//V
		case 86:
			iWhichLight = PER_VERTEX;
			break;

		//X
		case 88:
			iWhichRotation = X_ROT;
			break;

		//Y
		case 89:
			iWhichRotation = Y_ROT;
			break;

		//Z
		case 90:
			iWhichRotation = Z_ROT;
			break;


		//Q
		case 81:
			uninitialize();
			window.close();
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






	/******************** Per Vertex ********************/
	vertexShaderObject_PV = gl.createShader(gl.VERTEX_SHADER);
	var vertexShaderSourceCode_PV = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec3 vNormal;" +


		"uniform vec3 u_la;" +
		"uniform vec3 u_ld;" +
		"uniform vec3 u_ls;" +
		"uniform vec4 u_light_position;" +

		"uniform vec3 u_ka;" +
		"uniform vec3 u_kd;" +
		"uniform vec3 u_ks;" +
		"uniform float u_shininess;" +

		"uniform int u_LKey;" + 

		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_projection_matrix;" +

		"out vec3 outPhongLight;" +

		"void main() {" +

			
			"vec3 normalizeNormals;" + 
			"normalizeNormals = vNormal;" +
			"normalizeNormals = normalize(normalizeNormals);" +


			"if(u_LKey == 1) {" +

				"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +

				"vec3  source = normalize(vec3(u_light_position - eyeCoordinate));" +
				"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
				"vec3 normal = normalize(vec3(normalMatrix * normalizeNormals));" +
				"float S_Dot_N = max(dot(source, normal), 0.0);" +

				"vec3 viewer = normalize(vec3(-eyeCoordinate.xyz));" +
				"vec3 reflection = reflect(-source, normal);" +
				"float R_Dot_V = max(dot(reflection, viewer), 0.0);" + 

				"vec3 ambient = u_la * u_ka;" +
				"vec3 diffuse = u_ld * u_kd * S_Dot_N;" +
				"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess);" +
				"outPhongLight = ambient + diffuse + specular;" +

			"}"+ 
			"else{ " +
				"outPhongLight = vec3(1.0, 1.0, 1.0);" +
			"}" + 

			"gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" +
			
		"}";

	gl.shaderSource(vertexShaderObject_PV, vertexShaderSourceCode_PV);
	gl.compileShader(vertexShaderObject_PV);

	var shaderCompileStatus_PV = gl.getShaderParameter(vertexShaderObject_PV, gl.COMPILE_STATUS);

	if(shaderCompileStatus_PV == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_PV);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_PV = gl.createShader(gl.FRAGMENT_SHADER);
	var fragmentShaderSourceCode_PV = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec3 outPhongLight;" +
		"out vec4 FragColor;" +
		"void main(){" +
			"FragColor = vec4(outPhongLight, 1.0);" +
		"}";

	gl.shaderSource(fragmentShaderObject_PV, fragmentShaderSourceCode_PV);
	gl.compileShader(fragmentShaderObject_PV);

	shaderCompileStatus_PV = gl.getShaderParameter(fragmentShaderObject_PV, gl.COMPILE_STATUS);

	if(shaderCompileStatus_PV == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_PV);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_PV = gl.createProgram();

	gl.attachShader(shaderProgramObject_PV, vertexShaderObject_PV);
	gl.attachShader(shaderProgramObject_PV, fragmentShaderObject_PV);

	gl.bindAttribLocation(shaderProgramObject_PV, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_PV, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");


	gl.linkProgram(shaderProgramObject_PV);

	
	if(!gl.getProgramParameter(shaderProgramObject_PV, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject_PV);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	modelMatrix_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_model_matrix");
	viewMatrix_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_view_matrix");
	projectionMatrix_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_projection_matrix");

	la_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_la");
	ld_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ld");
	ls_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ls");
	lightPosition_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_light_position");
	

	ka_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ka");
	kd_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_kd");
	ks_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ks");
	shininess_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_shininess");

	LKeyPress_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_LKey");









	/******************** Per Fragment ********************/ 

	vertexShaderObject_PF = gl.createShader(gl.VERTEX_SHADER);
	var vertexShaderSourceCode_PF = 
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

	gl.shaderSource(vertexShaderObject_PF, vertexShaderSourceCode_PF);
	gl.compileShader(vertexShaderObject_PF);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_PF, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_PF);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_PF = gl.createShader(gl.FRAGMENT_SHADER);
	var fragmentShaderSourceCode_PF = 
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

	gl.shaderSource(fragmentShaderObject_PF, fragmentShaderSourceCode_PF);
	gl.compileShader(fragmentShaderObject_PF);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_PF, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_PF);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_PF = gl.createProgram();

	gl.attachShader(shaderProgramObject_PF, vertexShaderObject_PF);
	gl.attachShader(shaderProgramObject_PF, fragmentShaderObject_PF);

	gl.bindAttribLocation(shaderProgramObject_PF, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_PF, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");


	gl.linkProgram(shaderProgramObject_PF);

	
	if(!gl.getProgramParameter(shaderProgramObject_PF, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject_PF);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	modelMatrix_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_model_matrix");
	viewMatrix_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_view_matrix");
	projectionMatrix_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_projection_matrix");

	la_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_la");
	ld_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ld");
	ls_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ls");
	lightPosition_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_light_position");
	

	ka_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ka");
	kd_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_kd");
	ks_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ks");
	shininess_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_shininess");

	LKeyPress_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_LKey");



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

	gl.clearColor(0.250, 0.250, 0.250, 1.0);
	

	modelMatrix = mat4.create();
	viewMatrix = mat4.create();
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

	if(shaderProgramObject_PF){

		gl.useProgram(shaderProgramObject_PF);

			if(fragmentShaderObject_PF){
				gl.detachShader(shaderProgramObject_PF, fragmentShaderObject_PF);
				gl.deleteShader(fragmentShaderObject_PF);
				fragmentShaderObject_PF = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject_PF){
				gl.detachShader(shaderProgramObject_PF, vertexShaderObject_PF);
				gl.deleteShader(vertexShaderObject_PF);
				vertexShaderObject_PF = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_PF);
		shaderProgramObject_PF = 0;
	}



	if(shaderProgramObject_PV){

		gl.useProgram(shaderProgramObject_PV);

			if(fragmentShaderObject_PV){
				gl.detachShader(shaderProgramObject_PV, fragmentShaderObject_PV);
				gl.deleteShader(fragmentShaderObject_PV);
				fragmentShaderObject_PV = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject_PV){
				gl.detachShader(shaderProgramObject_PV, vertexShaderObject_PV);
				gl.deleteShader(vertexShaderObject_PV);
				vertexShaderObject_PV = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_PV);
		shaderProgramObject_PV = 0;
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

	var w = canvas.width;
	var h = canvas.height;


	if(iViewPortNo == 1)							/************ 1st SET ***********/
		gl.viewport( 0, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 2)
		gl.viewport( 0, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo == 3)
		gl.viewport( 0, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 4)
		gl.viewport( 0, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 5)
		gl.viewport( 0, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 6)
		gl.viewport( 0, 0, w / 6, h / 6);
	else if(iViewPortNo == 7)						/************ 2nd SET ***********/
		gl.viewport( 1 * w / 4, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 8)
		gl.viewport( 1 * w / 4, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo == 9)
		gl.viewport( 1 * w / 4, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 10)
		gl.viewport( 1 * w / 4, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 11)
		gl.viewport( 1 * w / 4, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 12)
		gl.viewport( 1 * w / 4, 0, w / 6, h / 6);
	else if(iViewPortNo == 13)						/************ 3rd SET ***********/
		gl.viewport( 2 * w / 4, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 14)						
		gl.viewport( 2 * w / 4, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo == 15)
		gl.viewport( 2 * w / 4, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 16)
		gl.viewport( 2 * w / 4, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 17)
		gl.viewport( 2 * w / 4, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 18)						
		gl.viewport( 2 * w / 4, 0, w / 6, h / 6);
	else if(iViewPortNo == 19)						/************ 4th SET ***********/
		gl.viewport( 3 * w / 4, 5 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 20)
		gl.viewport( 3 * w / 4, 4 * h / 6, w / 6,  h / 6);
	else if(iViewPortNo == 21)
		gl.viewport( 3 * w / 4, 3 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 22)
		gl.viewport( 3 * w / 4, 2 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 23)
		gl.viewport( 3 * w / 4, 1 * h / 6, w / 6, h / 6);
	else if(iViewPortNo == 24)
		gl.viewport( 3 * w / 4, 0, w / 6, h / 6);



	 mat4.identity(perspectiveProjectionMatrix);
	 mat4.perspective(perspectiveProjectionMatrix, 
	 				45.0,
	 				parseFloat(canvas.width) / parseFloat(canvas.height),
	 				0.1,
	 				100.0);
}



function draw(){

	

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);



	if(iWhichLight == PER_VERTEX){
				
		draw24SpherePerVertex();

	}
	else{

		draw24SpherePerFragment();
		
	}


	requestAnimationFrame(draw, canvas);
}

function update(){

	angle_Sphere = angle_Sphere + 0.08;

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



function draw24SpherePerVertex(){

			
	var materialAmbient = [0.0, 0.0, 0.0];
	var materialDiffuse = [0.0, 0.0, 0.0];
	var materialSpecular = [0.0, 0.0, 0.0];
	var materialShininess = 0.0;

	for(var i = 1 ; i <= 24; i++){


		if(i == 1){
			materialAmbient[0] = 0.0215;
			materialAmbient[1] = 0.1745;
			materialAmbient[2] = 0.215;
			

			materialDiffuse[0] = 0.07568;
			materialDiffuse[1] = 0.61424;
			materialDiffuse[2] = 0.07568;
			

			materialSpecular[0] = 0.633;
			materialSpecular[1] = 0.727811;
			materialSpecular[2] = 0.633;
			

			materialShininess = 0.6 * 128;

		}
		else if(i == 2){
			materialAmbient[0] = 0.135;
			materialAmbient[1] = 0.2225;
			materialAmbient[2] = 0.1575;
			

			materialDiffuse[0] = 0.54;
			materialDiffuse[1] = 0.89;
			materialDiffuse[2] = 0.63;
			

			materialSpecular[0] = 0.316228;
			materialSpecular[1] = 0.316228;
			materialSpecular[2] = 0.316228;
			

			materialShininess = 0.1 * 128;
		}
		else if(i == 3){
			materialAmbient[0] = 0.05375;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.06625;
			

			materialDiffuse[0] = 0.18275;
			materialDiffuse[1] = 0.17;
			materialDiffuse[2] = 0.22525;
			

			materialSpecular[0] = 0.332741;
			materialSpecular[1] = 0.328634;
			materialSpecular[2] = 0.346435;
			

			materialShininess = 0.3 * 128;
		}
		else if(i == 4){
			materialAmbient[0] = 0.25;
			materialAmbient[1] = 0.20725;
			materialAmbient[2] = 0.20725;
			

			materialDiffuse[0] = 1.0;
			materialDiffuse[1] = 0.829;
			materialDiffuse[2] = 0.829;
			

			materialSpecular[0] = 0.296648;
			materialSpecular[1] = 0.296648;
			materialSpecular[2] = 0.296648;
			

			materialShininess = 0.088 * 128;
		}
		else if(i == 5){
			materialAmbient[0] = 0.1745;
			materialAmbient[1] = 0.01175;
			materialAmbient[2] = 0.01175;
			

			materialDiffuse[0] = 0.61424;
			materialDiffuse[1] = 0.04136;
			materialDiffuse[2] = 0.04136;
			

			materialSpecular[0] = 0.727811;
			materialSpecular[1] = 0.626959;
			materialSpecular[2] = 0.626959;
			

			materialShininess = 0.6 * 128;
		}
		else if(i == 6){
			materialAmbient[0] = 0.1;
			materialAmbient[1] = 0.18725;
			materialAmbient[2] = 0.1745;
			

			materialDiffuse[0] = 0.396;
			materialDiffuse[1] = 0.74151;
			materialDiffuse[2] = 0.69102;
			

			materialSpecular[0] = 0.297254;
			materialSpecular[1] = 0.30829;
			materialSpecular[2] = 0.306678;
			

			materialShininess = 0.1 * 128;
		}
		else if(i == 7){
			materialAmbient[0] = 0.329412;
			materialAmbient[1] = 0.223529;
			materialAmbient[2] = 0.027451;
			

			materialDiffuse[0] = 0.780392;
			materialDiffuse[1] = 0.568627;
			materialDiffuse[2] = 0.113725;
			

			materialSpecular[0] = 0.992157;
			materialSpecular[1] = 0.941176;
			materialSpecular[2] = 0.807843;
			

			materialShininess = 0.21794872 * 128;
		}
		else if(i == 8){
			materialAmbient[0] = 0.2125;
			materialAmbient[1] = 0.1275;
			materialAmbient[2] = 0.054;
			

			materialDiffuse[0] = 0.714;
			materialDiffuse[1] = 0.4284;
			materialDiffuse[2] = 0.18144;
			

			materialSpecular[0] = 0.393548;
			materialSpecular[1] = 0.271906;
			materialSpecular[2] = 0.166721;
			

			materialShininess = 0.2 * 128;
		}
		else if(i == 9){
			materialAmbient[0] = 0.25;
			materialAmbient[1] = 0.25;
			materialAmbient[2] = 0.25;
			

			materialDiffuse[0] = 0.4;
			materialDiffuse[1] = 0.4;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.774597;
			materialSpecular[1] = 0.774597;
			materialSpecular[2] = 0.774597;
			

			materialShininess = 0.6 * 128;
		}
		else if(i == 10){
			materialAmbient[0] = 0.19125;
			materialAmbient[1] = 0.0735;
			materialAmbient[2] = 0.0225;
			

			materialDiffuse[0] = 0.7038;
			materialDiffuse[1] = 0.27048;
			materialDiffuse[2] = 0.0828;
			

			materialSpecular[0] = 0.256777;
			materialSpecular[1] = 0.137622;
			materialSpecular[2] = 0.086014;
			

			materialShininess = 0.1 * 128;
		}
		else if(i == 11){
			materialAmbient[0] = 0.24725;
			materialAmbient[1] = 0.1995;
			materialAmbient[2] = 0.0745;
			

			materialDiffuse[0] = 0.75164;
			materialDiffuse[1] = 0.60648;
			materialDiffuse[2] = 0.22648;
			

			materialSpecular[0] = 0.628281;
			materialSpecular[1] = 0.555802;
			materialSpecular[2] = 0.366065;
			

			materialShininess = 0.4 * 128;
		}
		else if(i == 12){
			materialAmbient[0] = 0.19225;
			materialAmbient[1] = 0.19225;
			materialAmbient[2] = 0.19225;
			

			materialDiffuse[0] = 0.50754;
			materialDiffuse[1] = 0.50754;
			materialDiffuse[2] = 0.50754;
			

			materialSpecular[0] = 0.508273;
			materialSpecular[1] = 0.508273;
			materialSpecular[2] = 0.508273;
			

			materialShininess = 0.4 * 128;
		}
		else if(i == 13){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.01;
			materialDiffuse[1] = 0.01;
			materialDiffuse[2] = 0.01;
			

			materialSpecular[0] = 0.5;
			materialSpecular[1] = 0.5;
			materialSpecular[2] = 0.5;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 14){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.1;
			materialAmbient[2] = 0.06;
			

			materialDiffuse[0] = 0.0;
			materialDiffuse[1] = 0.50980392;
			materialDiffuse[2] = 0.52980392;
			

			materialSpecular[0] = 0.50196078;
			materialSpecular[1] = 0.50196078;
			materialSpecular[2] = 0.50196078;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 15){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.1;
			materialDiffuse[1] = 0.35;
			materialDiffuse[2] = 0.1;
			

			materialSpecular[0] = 0.45;
			materialSpecular[1] = 0.55;
			materialSpecular[2] = 0.45;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 16){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.0;
			materialDiffuse[2] = 0.0;
			

			materialSpecular[0] = 0.7;
			materialSpecular[1] = 0.6;
			materialSpecular[2] = 0.6;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 17){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.55;
			materialDiffuse[1] = 0.55;
			materialDiffuse[2] = 0.55;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.70;
			materialSpecular[2] = 0.70;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 18){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.0;
			

			materialSpecular[0] = 0.60;
			materialSpecular[1] = 0.60;
			materialSpecular[2] = 0.50;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 19){
			materialAmbient[0] = 0.02;
			materialAmbient[1] = 0.02;
			materialAmbient[2] = 0.02;
			

			materialDiffuse[0] = 0.01;
			materialDiffuse[1] = 0.01;
			materialDiffuse[2] = 0.01;
			

			materialSpecular[0] = 0.4;
			materialSpecular[1] = 0.4;
			materialSpecular[2] = 0.4;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 20){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.05;
			

			materialDiffuse[0] = 0.4;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.5;
			

			materialSpecular[0] = 0.04;
			materialSpecular[1] = 0.7;
			materialSpecular[2] = 0.7;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 21){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.4;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.04;
			materialSpecular[1] = 0.7;
			materialSpecular[2] = 0.04;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 22){
			materialAmbient[0] = 0.05;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.4;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.04;
			materialSpecular[2] = 0.04;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 23){
			materialAmbient[0] = 0.05;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.05;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.5;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.70;
			materialSpecular[2] = 0.70;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 24){
			materialAmbient[0] = 0.05;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.70;
			materialSpecular[2] = 0.04;
			

			materialShininess = 0.078125 * 128;
		}


		iViewPortNo = i;
		resize();


		gl.useProgram(shaderProgramObject_PV);
		
		/********** Sphere ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		
		gl.uniformMatrix4fv(modelMatrix_Uniform_PV, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform_PV, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform_PV, false, perspectiveProjectionMatrix)
		


	
			
		if(bLights == true){
			//Per Vertex

			if(iWhichRotation == X_ROT)
				rotateX(angle_Sphere);
			else if(iWhichRotation == Y_ROT)
				rotateY(angle_Sphere);
			else if(iWhichRotation == Z_ROT)
				rotateZ(angle_Sphere);

			update();


			gl.uniform1i(LKeyPress_Uniform_PV, 1);

			gl.uniform3fv(la_Uniform_PV, lightAmbient);
			gl.uniform3fv(ld_Uniform_PV, lightDiffuse);
			gl.uniform3fv(ls_Uniform_PV, lightSpecular);
			gl.uniform4fv(lightPosition_Uniform_PV, lightPosition);

			gl.uniform3fv(ka_Uniform_PV, materialAmbient);
			gl.uniform3fv(kd_Uniform_PV, materialDiffuse);
			gl.uniform3fv(ks_Uniform_PV, materialSpecular);
			gl.uniform1f(shininess_Uniform_PV, materialShininess);	

		}
		else
			gl.uniform1i(LKeyPress_Uniform_PV, 0);
		
		

		gl.bindVertexArray(vao_Sphere);

			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index);
			gl.drawElements(gl.TRIANGLES, (STACK) * (SLICES) * 6, gl.UNSIGNED_SHORT, 0);

		gl.bindVertexArray(null);


		gl.useProgram(null);

	}
}




function draw24SpherePerFragment(){

			
	var materialAmbient = [0.0, 0.0, 0.0];
	var materialDiffuse = [0.0, 0.0, 0.0];
	var materialSpecular = [0.0, 0.0, 0.0];
	var materialShininess = 0.0;

	for(var i = 1 ; i <= 24; i++){


		if(i == 1){
			materialAmbient[0] = 0.0215;
			materialAmbient[1] = 0.1745;
			materialAmbient[2] = 0.215;
			

			materialDiffuse[0] = 0.07568;
			materialDiffuse[1] = 0.61424;
			materialDiffuse[2] = 0.07568;
			

			materialSpecular[0] = 0.633;
			materialSpecular[1] = 0.727811;
			materialSpecular[2] = 0.633;
			

			materialShininess = 0.6 * 128;

		}
		else if(i == 2){
			materialAmbient[0] = 0.135;
			materialAmbient[1] = 0.2225;
			materialAmbient[2] = 0.1575;
			

			materialDiffuse[0] = 0.54;
			materialDiffuse[1] = 0.89;
			materialDiffuse[2] = 0.63;
			

			materialSpecular[0] = 0.316228;
			materialSpecular[1] = 0.316228;
			materialSpecular[2] = 0.316228;
			

			materialShininess = 0.1 * 128;
		}
		else if(i == 3){
			materialAmbient[0] = 0.05375;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.06625;
			

			materialDiffuse[0] = 0.18275;
			materialDiffuse[1] = 0.17;
			materialDiffuse[2] = 0.22525;
			

			materialSpecular[0] = 0.332741;
			materialSpecular[1] = 0.328634;
			materialSpecular[2] = 0.346435;
			

			materialShininess = 0.3 * 128;
		}
		else if(i == 4){
			materialAmbient[0] = 0.25;
			materialAmbient[1] = 0.20725;
			materialAmbient[2] = 0.20725;
			

			materialDiffuse[0] = 1.0;
			materialDiffuse[1] = 0.829;
			materialDiffuse[2] = 0.829;
			

			materialSpecular[0] = 0.296648;
			materialSpecular[1] = 0.296648;
			materialSpecular[2] = 0.296648;
			

			materialShininess = 0.088 * 128;
		}
		else if(i == 5){
			materialAmbient[0] = 0.1745;
			materialAmbient[1] = 0.01175;
			materialAmbient[2] = 0.01175;
			

			materialDiffuse[0] = 0.61424;
			materialDiffuse[1] = 0.04136;
			materialDiffuse[2] = 0.04136;
			

			materialSpecular[0] = 0.727811;
			materialSpecular[1] = 0.626959;
			materialSpecular[2] = 0.626959;
			

			materialShininess = 0.6 * 128;
		}
		else if(i == 6){
			materialAmbient[0] = 0.1;
			materialAmbient[1] = 0.18725;
			materialAmbient[2] = 0.1745;
			

			materialDiffuse[0] = 0.396;
			materialDiffuse[1] = 0.74151;
			materialDiffuse[2] = 0.69102;
			

			materialSpecular[0] = 0.297254;
			materialSpecular[1] = 0.30829;
			materialSpecular[2] = 0.306678;
			

			materialShininess = 0.1 * 128;
		}
		else if(i == 7){
			materialAmbient[0] = 0.329412;
			materialAmbient[1] = 0.223529;
			materialAmbient[2] = 0.027451;
			

			materialDiffuse[0] = 0.780392;
			materialDiffuse[1] = 0.568627;
			materialDiffuse[2] = 0.113725;
			

			materialSpecular[0] = 0.992157;
			materialSpecular[1] = 0.941176;
			materialSpecular[2] = 0.807843;
			

			materialShininess = 0.21794872 * 128;
		}
		else if(i == 8){
			materialAmbient[0] = 0.2125;
			materialAmbient[1] = 0.1275;
			materialAmbient[2] = 0.054;
			

			materialDiffuse[0] = 0.714;
			materialDiffuse[1] = 0.4284;
			materialDiffuse[2] = 0.18144;
			

			materialSpecular[0] = 0.393548;
			materialSpecular[1] = 0.271906;
			materialSpecular[2] = 0.166721;
			

			materialShininess = 0.2 * 128;
		}
		else if(i == 9){
			materialAmbient[0] = 0.25;
			materialAmbient[1] = 0.25;
			materialAmbient[2] = 0.25;
			

			materialDiffuse[0] = 0.4;
			materialDiffuse[1] = 0.4;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.774597;
			materialSpecular[1] = 0.774597;
			materialSpecular[2] = 0.774597;
			

			materialShininess = 0.6 * 128;
		}
		else if(i == 10){
			materialAmbient[0] = 0.19125;
			materialAmbient[1] = 0.0735;
			materialAmbient[2] = 0.0225;
			

			materialDiffuse[0] = 0.7038;
			materialDiffuse[1] = 0.27048;
			materialDiffuse[2] = 0.0828;
			

			materialSpecular[0] = 0.256777;
			materialSpecular[1] = 0.137622;
			materialSpecular[2] = 0.086014;
			

			materialShininess = 0.1 * 128;
		}
		else if(i == 11){
			materialAmbient[0] = 0.24725;
			materialAmbient[1] = 0.1995;
			materialAmbient[2] = 0.0745;
			

			materialDiffuse[0] = 0.75164;
			materialDiffuse[1] = 0.60648;
			materialDiffuse[2] = 0.22648;
			

			materialSpecular[0] = 0.628281;
			materialSpecular[1] = 0.555802;
			materialSpecular[2] = 0.366065;
			

			materialShininess = 0.4 * 128;
		}
		else if(i == 12){
			materialAmbient[0] = 0.19225;
			materialAmbient[1] = 0.19225;
			materialAmbient[2] = 0.19225;
			

			materialDiffuse[0] = 0.50754;
			materialDiffuse[1] = 0.50754;
			materialDiffuse[2] = 0.50754;
			

			materialSpecular[0] = 0.508273;
			materialSpecular[1] = 0.508273;
			materialSpecular[2] = 0.508273;
			

			materialShininess = 0.4 * 128;
		}
		else if(i == 13){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.01;
			materialDiffuse[1] = 0.01;
			materialDiffuse[2] = 0.01;
			

			materialSpecular[0] = 0.5;
			materialSpecular[1] = 0.5;
			materialSpecular[2] = 0.5;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 14){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.1;
			materialAmbient[2] = 0.06;
			

			materialDiffuse[0] = 0.0;
			materialDiffuse[1] = 0.50980392;
			materialDiffuse[2] = 0.52980392;
			

			materialSpecular[0] = 0.50196078;
			materialSpecular[1] = 0.50196078;
			materialSpecular[2] = 0.50196078;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 15){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.1;
			materialDiffuse[1] = 0.35;
			materialDiffuse[2] = 0.1;
			

			materialSpecular[0] = 0.45;
			materialSpecular[1] = 0.55;
			materialSpecular[2] = 0.45;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 16){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.0;
			materialDiffuse[2] = 0.0;
			

			materialSpecular[0] = 0.7;
			materialSpecular[1] = 0.6;
			materialSpecular[2] = 0.6;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 17){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.55;
			materialDiffuse[1] = 0.55;
			materialDiffuse[2] = 0.55;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.70;
			materialSpecular[2] = 0.70;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 18){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.0;
			

			materialSpecular[0] = 0.60;
			materialSpecular[1] = 0.60;
			materialSpecular[2] = 0.50;
			

			materialShininess = 0.25 * 128;
		}
		else if(i == 19){
			materialAmbient[0] = 0.02;
			materialAmbient[1] = 0.02;
			materialAmbient[2] = 0.02;
			

			materialDiffuse[0] = 0.01;
			materialDiffuse[1] = 0.01;
			materialDiffuse[2] = 0.01;
			

			materialSpecular[0] = 0.4;
			materialSpecular[1] = 0.4;
			materialSpecular[2] = 0.4;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 20){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.05;
			

			materialDiffuse[0] = 0.4;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.5;
			

			materialSpecular[0] = 0.04;
			materialSpecular[1] = 0.7;
			materialSpecular[2] = 0.7;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 21){
			materialAmbient[0] = 0.0;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.4;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.04;
			materialSpecular[1] = 0.7;
			materialSpecular[2] = 0.04;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 22){
			materialAmbient[0] = 0.05;
			materialAmbient[1] = 0.0;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.4;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.04;
			materialSpecular[2] = 0.04;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 23){
			materialAmbient[0] = 0.05;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.05;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.5;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.70;
			materialSpecular[2] = 0.70;
			

			materialShininess = 0.078125 * 128;
		}
		else if(i == 24){
			materialAmbient[0] = 0.05;
			materialAmbient[1] = 0.05;
			materialAmbient[2] = 0.0;
			

			materialDiffuse[0] = 0.5;
			materialDiffuse[1] = 0.5;
			materialDiffuse[2] = 0.4;
			

			materialSpecular[0] = 0.70;
			materialSpecular[1] = 0.70;
			materialSpecular[2] = 0.04;
			

			materialShininess = 0.078125 * 128;
		}


		iViewPortNo = i;
		resize();


		gl.useProgram(shaderProgramObject_PF);
		
		/********** Sphere ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		
		gl.uniformMatrix4fv(modelMatrix_Uniform_PF, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform_PF, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform_PF, false, perspectiveProjectionMatrix)
		


	
			
		if(bLights == true){
			//Per Fragment
			

			if(iWhichRotation == X_ROT)
				rotateX(angle_Sphere);
			else if(iWhichRotation == Y_ROT)
				rotateY(angle_Sphere);
			else if(iWhichRotation == Z_ROT)
				rotateZ(angle_Sphere);

			update();

			gl.uniform1i(LKeyPress_Uniform_PF, 1);

			gl.uniform3fv(la_Uniform_PF, lightAmbient);
			gl.uniform3fv(ld_Uniform_PF, lightDiffuse);
			gl.uniform3fv(ls_Uniform_PF, lightSpecular);
			gl.uniform4fv(lightPosition_Uniform_PF, lightPosition);

			gl.uniform3fv(ka_Uniform_PF, materialAmbient);
			gl.uniform3fv(kd_Uniform_PF, materialDiffuse);
			gl.uniform3fv(ks_Uniform_PF, materialSpecular);
			gl.uniform1f(shininess_Uniform_PF, materialShininess);	

		}
		else
			gl.uniform1i(LKeyPress_Uniform_PF, 0);
		
		

		gl.bindVertexArray(vao_Sphere);

			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index);
			gl.drawElements(gl.TRIANGLES, (STACK) * (SLICES) * 6, gl.UNSIGNED_SHORT, 0);

		gl.bindVertexArray(null);


		gl.useProgram(null);

	}
}


function rotateX(angle){
	lightPosition[0] = 0.0;
	lightPosition[1] = 15.0 * Math.sin(degToRad(angle));
	lightPosition[2] = 15.0 * Math.cos(degToRad(angle));
}

function rotateY(angle){
	lightPosition[0] = 15.0 * Math.cos(degToRad(angle));
	lightPosition[1] = 0.0;
	lightPosition[2] = 15.0 * Math.sin(degToRad(angle));
}


function rotateZ(angle){
	lightPosition[0] = 15.0 * Math.cos(degToRad(angle));
	lightPosition[1] = 15.0 * Math.sin(degToRad(angle));
	lightPosition[2] = 0.0;
}

