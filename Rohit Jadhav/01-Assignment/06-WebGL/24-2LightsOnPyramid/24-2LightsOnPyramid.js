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


//For Pyramid
var vao_Pyramid;
var vbo_Pyramid_Position;
var vbo_Pyramid_Normal;
var angle_Pyramid = 0.0;



//For Light Toggling
const PER_VERTEX = 1;
const PER_FRAGMENT = 2;

var iWhichLight = PER_VERTEX;


//For Per Fragment Shader
var vertexShaderObject_PF;
var fragmentShaderObject_PF;
var shaderProgramObject_PF;


//For Uniform
var modelMatrix_Uniform_PF;
var viewMatrix_Uniform_PF;
var projectionMatrix_Uniform_PF;

//For Light Uniform
var red_la_Uniform_PF;
var red_ld_Uniform_PF;
var red_ls_Uniform_PF;
var red_lightPosition_Uniform_PF;

var blue_la_Uniform_PF;
var blue_ld_Uniform_PF;
var blue_ls_Uniform_PF;
var blue_lightPosition_Uniform_PF;

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
var red_la_Uniform_PV;
var red_ld_Uniform_PV;
var red_ls_Uniform_PV;
var red_lightPosition_Uniform_PV;

var blue_la_Uniform_PV;
var blue_ld_Uniform_PV;
var blue_ls_Uniform_PV;
var blue_lightPosition_Uniform_PV;

var ka_Uniform_PV;
var kd_Uniform_PV;
var ks_Uniform_PV;
var shininess_Uniform_PV;
var LKeyPress_Uniform_PV;



//For Lights
var bLights = false;
var red_lightAmbient = [0.0, 0.0, 0.0];
var red_lightDiffuse =[1.0, 0.0, 0.0];
var red_lightSpecular = [1.0, 0.0, 0.0];
var red_lightPosition = [-2.0, 0.0, 0.0, 1.0];

var blue_lightAmbient = [0.0, 0.0, 0.0];
var blue_lightDiffuse =[0.0, 0.0, 1.0];
var blue_lightSpecular = [0.0, 0.0, 1.0];
var blue_lightPosition = [2.0, 0.0, 0.0, 1.0];


//For Material
var materialAmbient = [0.0, 0.0, 0.0];
var materialDiffuse = [1.0, 1.0, 1.0];
var materialSpecular = [1.0, 1.0, 1.0];
var materialShininess = 128.0;


function main(){

	canvas = document.getElementById("24-2LightsOnPyramid-RRJ");
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


		"uniform vec3 u_la_red;" +
		"uniform vec3 u_ld_red;" +
		"uniform vec3 u_ls_red;" +
		"uniform vec4 u_light_position_red;" +

		"uniform vec3 u_la_blue;" +
		"uniform vec3 u_ld_blue;" +
		"uniform vec3 u_ls_blue;" +
		"uniform vec4 u_light_position_blue;" +


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


			"if(u_LKey == 1) {" +

				"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +

				"vec3  source_red = normalize(vec3(u_light_position_red - eyeCoordinate));" +

				"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
				"vec3 normal = normalize(vec3(normalMatrix * vNormal));" +
				
				"float SRed_Dot_N = max(dot(source_red, normal), 0.0);" +

				"vec3  source_blue = normalize(vec3(u_light_position_blue - eyeCoordinate));" +
				"float SBlue_Dot_N = max(dot(source_blue, normal), 0.0);" +


				"vec3 viewer = normalize(vec3(-eyeCoordinate.xyz));" +
				
				"vec3 reflection_red = reflect(-source_red, normal);" +
				"float RRed_Dot_V = max(dot(reflection_red, viewer), 0.0);" + 

				"vec3 reflection_blue = reflect(-source_blue, normal);" +
				"float RBlue_Dot_V = max(dot(reflection_blue, viewer), 0.0);" + 
	

				"vec3 ambient_red = u_la_red * u_ka;" +
				"vec3 diffuse_red = u_ld_red * u_kd * SRed_Dot_N;" +
				"vec3 specular_red = u_ls_red * u_ks * pow(RRed_Dot_V, u_shininess);" +
				"vec3 red = ambient_red + diffuse_red + specular_red;" +

				"vec3 ambient_blue = u_la_blue * u_ka;" +
				"vec3 diffuse_blue = u_ld_blue * u_kd * SBlue_Dot_N;" +
				"vec3 specular_blue = u_ls_blue * u_ks * pow(RBlue_Dot_V, u_shininess);" +
				"vec3 blue = ambient_blue + diffuse_blue + specular_blue;" +

				"outPhongLight = red + blue;" +

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

	red_la_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_la_red");
	red_ld_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ld_red");
	red_ls_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ls_red");
	red_lightPosition_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_light_position_red");

	blue_la_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_la_blue");
	blue_ld_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ld_blue");
	blue_ls_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_ls_blue");
	blue_lightPosition_Uniform_PV = gl.getUniformLocation(shaderProgramObject_PV, "u_light_position_blue");
	

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

		"uniform vec4 u_light_position_red;" +
		"uniform vec4 u_light_position_blue;" +

 
		"uniform mat4 u_model_matrix;" +
		"uniform mat4 u_view_matrix;" +
		"uniform mat4 u_projection_matrix;" +

		"out vec3 outViewer;" + 
		"out vec3 outLightDirection_red;" +
		"out vec3 outLightDirection_blue;" +
		"out vec3 outNormal;" + 		

		"void main() {" +


			"vec4 eyeCoordinate = u_view_matrix * u_model_matrix * vPosition;" +

			"outLightDirection_red = vec3(u_light_position_red - eyeCoordinate);" +
			"outLightDirection_blue = vec3(u_light_position_blue - eyeCoordinate);" +


			"mat3 normalMatrix = mat3(u_view_matrix * u_model_matrix);" +
			"outNormal = vec3(normalMatrix * vNormal);" +

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

		"in vec3 outLightDirection_red;" +
		"in vec3 outLightDirection_blue;" +
		"in vec3 outNormal;" +
		"in vec3 outViewer;" +


		
		"uniform vec3 u_la_red;" +
		"uniform vec3 u_ld_red;" +
		"uniform vec3 u_ls_red;" +
		"uniform vec4 u_light_position_red;" +

		"uniform vec3 u_la_blue;" +
		"uniform vec3 u_ld_blue;" +
		"uniform vec3 u_ls_blue;" +
		"uniform vec4 u_light_position_blue;" +

		
		"uniform vec3 u_ka;" +
		"uniform vec3 u_kd;" +
		"uniform vec3 u_ks;" +
		"uniform float u_shininess;" +

		"uniform int u_LKey;" +



		"out vec4 FragColor;" +
		"void main(){" +

			"vec3 PhongLight;" +

			"if(u_LKey == 1){" +

				"vec3 normalizeLightDirection_red = normalize(outLightDirection_red);" +
				"vec3 normalizeLightDirection_blue = normalize(outLightDirection_blue);" +


				"vec3 normalizeNormalVector = normalize(outNormal);" +

				"float SRed_Dot_N = max(dot(normalizeLightDirection_red, normalizeNormalVector), 0.0);" +
				"float SBlue_Dot_N = max(dot(normalizeLightDirection_blue, normalizeNormalVector), 0.0);" +


				"vec3 normalizeViewer = normalize(outViewer);" +
				"vec3 reflection_red = reflect(-normalizeLightDirection_red, normalizeNormalVector);" +
				"vec3 reflection_blue = reflect(-normalizeLightDirection_blue, normalizeNormalVector);" +


				"float RRed_Dot_V = max(dot(reflection_red, normalizeViewer), 0.0);" +
				"float RBlue_Dot_V = max(dot(reflection_blue, normalizeViewer), 0.0);" +

				
				"vec3 ambient_red = u_la_red * u_ka;" +
				"vec3 diffuse_red = u_ld_red * u_kd * SRed_Dot_N;" +
				"vec3 specular_red = u_ls_red * u_ks * pow(RRed_Dot_V, u_shininess);" +
				"vec3 red = ambient_red + diffuse_red + specular_red;" +

				"vec3 ambient_blue = u_la_blue * u_ka;" +
				"vec3 diffuse_blue = u_ld_blue * u_kd * SBlue_Dot_N;" +
				"vec3 specular_blue = u_ls_blue * u_ks * pow(RBlue_Dot_V, u_shininess);" +
				"vec3 blue = ambient_blue + diffuse_blue + specular_blue;" +

				"PhongLight = red + blue;" +

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

	red_la_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_la_red");
	red_ld_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ld_red");
	red_ls_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ls_red");
	red_lightPosition_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_light_position_red");

	blue_la_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_la_blue");
	blue_ld_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ld_blue");
	blue_ls_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ls_blue");
	blue_lightPosition_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_light_position_blue");
	

	ka_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ka");
	kd_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_kd");
	ks_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_ks");
	shininess_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_shininess");

	LKeyPress_Uniform_PF = gl.getUniformLocation(shaderProgramObject_PF, "u_LKey");



	/********** Sphere Position and Normal **********/

	var pyramid_Position = new Float32Array([
						//Face
						0.0, 1.0, 0.0,
						-1.0, -1.0, 1.0,
						1.0, -1.0, 1.0,
						//Right
						0.0, 1.0, 0.0,
						1.0, -1.0, 1.0,
						1.0, -1.0, -1.0,
						//Back
						0.0, 1.0, 0.0,
						1.0, -1.0, -1.0,
						-1.0, -1.0, -1.0,
						//Left
						0.0, 1.0, 0.0,
						-1.0, -1.0, -1.0,
						-1.0, -1.0, 1.0,
					]);


	var pyramid_Normal = new Float32Array([
					//Face
					0.0, 0.447214, 0.894427,
					0.0, 0.447214, 0.894427,
					0.0, 0.447214, 0.894427,


					//Right
					0.894427, 0.447214, 0.0,
					0.894427, 0.447214, 0.0,
					0.894427, 0.447214, 0.0,


					//Back
					0.0, 0.447214, -0.894427,
					0.0, 0.447214, -0.894427,
					0.0, 0.447214, -0.894427,

					//Left
					-0.894427, 0.447214, 0.0,
					-0.894427, 0.447214, 0.0,
					-0.894427, 0.447214, 0.0,
	
					]);


	/********** Pyramid ***********/
	vao_Pyramid = gl.createVertexArray();
	gl.bindVertexArray(vao_Pyramid);

		/********** Position **********/
		vbo_Pyramid_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Pyramid_Position);
		gl.bufferData(gl.ARRAY_BUFFER, pyramid_Position, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);



		/********** Normal **********/
		vbo_Pyramid_Normal = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Pyramid_Normal);
		gl.bufferData(gl.ARRAY_BUFFER, pyramid_Normal, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	
	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.disable(gl.CULL_FACE);
	
	gl.clearDepth(1.0);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix = mat4.create();
}



function uninitialize(){



	if(vbo_Pyramid_Normal){
		gl.deleteBuffer(vbo_Pyramid_Normal);
		vbo_Pyramid_Normal = 0;
	}

	if(vbo_Pyramid_Position){
		gl.deleteBuffer(vbo_Pyramid_Position);
		vbo_Pyramid_Position = 0;
	}

	if(vao_Pyramid){
		gl.deleteVertexArray(vao_Pyramid);
		vao_Pyramid = 0;
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




	if(iWhichLight == PER_VERTEX){


		gl.useProgram(shaderProgramObject_PV);
		
		/********** Pyramid ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		mat4.rotateY(modelMatrix, modelMatrix, degToRad(angle_Pyramid));
		
		gl.uniformMatrix4fv(modelMatrix_Uniform_PV, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform_PV, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform_PV, false, perspectiveProjectionMatrix)
		


	
			
		if(bLights == true){
			//Per Vertex
			gl.uniform1i(LKeyPress_Uniform_PV, 1);


			gl.uniform3fv(red_la_Uniform_PV, red_lightAmbient);
			gl.uniform3fv(red_ld_Uniform_PV, red_lightDiffuse);
			gl.uniform3fv(red_ls_Uniform_PV, red_lightSpecular);
			gl.uniform4fv(red_lightPosition_Uniform_PV, red_lightPosition);

			gl.uniform3fv(blue_la_Uniform_PV, blue_lightAmbient);
			gl.uniform3fv(blue_ld_Uniform_PV, blue_lightDiffuse);
			gl.uniform3fv(blue_ls_Uniform_PV, blue_lightSpecular);
			gl.uniform4fv(blue_lightPosition_Uniform_PV, blue_lightPosition);

			gl.uniform3fv(ka_Uniform_PV, materialAmbient);
			gl.uniform3fv(kd_Uniform_PV, materialDiffuse);
			gl.uniform3fv(ks_Uniform_PV, materialSpecular);
			gl.uniform1f(shininess_Uniform_PV, materialShininess);	

		}
		else
			gl.uniform1i(LKeyPress_Uniform_PV, 0);
		
		

		gl.bindVertexArray(vao_Pyramid);

			gl.drawArrays(gl.TRIANGLES, 0, 12);

		gl.bindVertexArray(null);


		gl.useProgram(null);

	}
	else if(iWhichLight == PER_FRAGMENT){

		gl.useProgram(shaderProgramObject_PF);
		
		/********** Pyramid ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		mat4.rotateY(modelMatrix, modelMatrix, degToRad(angle_Pyramid));
		
		gl.uniformMatrix4fv(modelMatrix_Uniform_PF, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform_PF, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform_PF, false, perspectiveProjectionMatrix)
		


	
			
		if(bLights == true){
			//Per Fragment
			gl.uniform1i(LKeyPress_Uniform_PF, 1);

			gl.uniform3fv(red_la_Uniform_PF, red_lightAmbient);
			gl.uniform3fv(red_ld_Uniform_PF, red_lightDiffuse);
			gl.uniform3fv(red_ls_Uniform_PF, red_lightSpecular);
			gl.uniform4fv(red_lightPosition_Uniform_PF, red_lightPosition);

			gl.uniform3fv(blue_la_Uniform_PF, blue_lightAmbient);
			gl.uniform3fv(blue_ld_Uniform_PF, blue_lightDiffuse);
			gl.uniform3fv(blue_ls_Uniform_PF, blue_lightSpecular);
			gl.uniform4fv(blue_lightPosition_Uniform_PF, blue_lightPosition);

			gl.uniform3fv(ka_Uniform_PF, materialAmbient);
			gl.uniform3fv(kd_Uniform_PF, materialDiffuse);
			gl.uniform3fv(ks_Uniform_PF, materialSpecular);
			gl.uniform1f(shininess_Uniform_PF, materialShininess);	

		}
		else
			gl.uniform1i(LKeyPress_Uniform_PF, 0);
		
		

		gl.bindVertexArray(vao_Pyramid);

			gl.drawArrays(gl.TRIANGLES, 0, 12);

		gl.bindVertexArray(null);


		gl.useProgram(null);

	}

	update();

	requestAnimationFrame(draw, canvas);
}

function update(){

	angle_Pyramid = angle_Pyramid + 0.3;

	if(angle_Pyramid > 360.0)
		angle_Pyramid = 0.0;
}




function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}