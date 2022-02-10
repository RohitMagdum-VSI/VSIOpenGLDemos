var canvas_RRJ = null;
var gl = null;
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
var vao_Sphere_RRJ;
var vbo_Sphere_Position_RRJ;
var vbo_Sphere_Normal_RRJ;
var vbo_Sphere_Index_RRJ;

const STACK = 30;
const SLICES = 30;

var sphere_Position_RRJ;
var sphere_Normal_RRJ;
var sphere_Index_RRJ;
var angle_Sphere_RRJ = 0.0;


//For Light Uniform
var la_Uniform_RRJ;
var ld_Uniform_RRJ;
var ls_Uniform_RRJ;
var lightPosition_Uniform_RRJ;

var ka_Uniform_RRJ;
var kd_Uniform_RRJ;
var ks_Uniform_RRJ;
var shininess_Uniform_RRJ;



//For Lights
var lightAmbient_RRJ = [0.0, 0.0, 0.0];
var lightDiffuse_RRJ =[1.0, 1.0, 1.0];
var lightSpecular_RRJ = [1.0, 1.0, 1.0];
var lightPosition_RRJ = [0.0, 0.0, 5.0, 1.0];
var bLights_RRJ = false;


//For Material
var materialAmbient_RRJ = [0.0, 0.0, 0.0];
var materialDiffuse_RRJ = [1.0, 1.0, 1.0];
var materialSpecular_RRJ = [1.0, 1.0, 1.0];
var materialShininess_RRJ = 128.0;



//For Spot Light Uniform
var spotLightDirectionUniform_RRJ;
var spotLightCutoffUniform_RRJ;
var spotLightExponentUniform_RRJ;
var constantAttenuationUniform_RRJ;
var linearAttenuationUniform_RRJ;
var quadraticAttenuationUniform_RRJ;

//For Spot Light Values
var PI = 3.1415926535;
var spotLightDirection_RRJ = [0.0, 0.0, -1.0, 1.0];
var spotLightCutoff_RRJ = Math.cos(5 * PI / 180.0);	//0.78539f
var spotLightExponent_RRJ = 20.0;
var constantAttenuation_RRJ = 1.0;
var linearAttenuation_RRJ = 0.09;
var quadraticAttenuation_RRJ = 0.032;


function main(){

	canvas_RRJ = document.getElementById("32-SpotLight-RRJ");
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

	var fullscreen_element = 
		document.fullscreenElement ||
		document.webkitFullscreenElement ||
		document.mozFullScreenElement ||
		document.msFullscreenElement || 
		document.oFullscreenElement ||
		null;


	if(fullscreen_element == null){

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

	gl = canvas_RRJ.getContext("webgl2");
	if(gl == null){
		console.log("Obtaining Context Failed!!\n");
		return;
	}
	else 
		console.log("Context Obtained!!\n");


	gl.viewportWidth = canvas_RRJ.width;
	gl.viewportHeight = canvas_RRJ.height;


	vertexShaderObject_RRJ = gl.createShader(gl.VERTEX_SHADER);
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

	gl.shaderSource(vertexShaderObject_RRJ, vertexShaderSourceCode);
	gl.compileShader(vertexShaderObject_RRJ);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject_RRJ, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject_RRJ);
		if(error.length > 0){
			alert("Vertex Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	fragmentShaderObject_RRJ = gl.createShader(gl.FRAGMENT_SHADER);
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


		"uniform vec4 u_spotLightDirection;" +
		"uniform float u_spotLightCutoff;" +
		"uniform float u_spotLightExponent;" +
		"uniform float u_constantAttenuation;" +
		"uniform float u_linearAttenuation;" +
		"uniform float u_quadraticAttenuation;" +

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

				"float d = length(normalizeLightDirection);" +
				"float attenuation = 1.0f / (u_quadraticAttenuation * d * d + u_linearAttenuation * d + u_constantAttenuation);" +

				"float spotDot = max(dot(-normalizeLightDirection, normalize(u_spotLightDirection.xyz)), 0.0);" +
				"float attenuationFactor;" +

				"if(spotDot > u_spotLightCutoff) {" +
					"attenuationFactor = pow(spotDot, u_spotLightExponent);" +
				"}" +
				"else {" +
					"attenuationFactor = 0.1f;" +
				"}" +

				"attenuation = attenuationFactor * attenuation;" +

				"vec3 ambient = u_la * u_ka * attenuation;" +
				"vec3 diffuse = u_ld * u_kd * S_Dot_N * attenuation;" +
				"vec3 specular = u_ls * u_ks * pow(R_Dot_V, u_shininess) * attenuation;" +
				"PhongLight = ambient + diffuse + specular;" +

			"}" +
			"else {" + 
				"PhongLight = vec3(1.0, 1.0, 1.0);" +
			"}" +


			"FragColor = vec4(PhongLight, 1.0);" +
		"}";

	gl.shaderSource(fragmentShaderObject_RRJ, fragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject_RRJ);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject_RRJ, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject_RRJ);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error: " + error);
			uninitialize();
			window.close();
		}
	}


	shaderProgramObject_RRJ = gl.createProgram();

	gl.attachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
	gl.attachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);

	gl.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_NORMAL, "vNormal");


	gl.linkProgram(shaderProgramObject_RRJ);

	
	if(!gl.getProgramParameter(shaderProgramObject_RRJ, gl.LINK_STATUS)){
		var error = gl.getProgramInfoLog(shaderProgramObject_RRJ);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	modelMatrix_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_model_matrix");
	viewMatrix_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_view_matrix");
	projectionMatrix_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_projection_matrix");

	la_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_la");
	ld_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_ld");
	ls_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_ls");
	lightPosition_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_light_position");
	

	ka_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_ka");
	kd_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_kd");
	ks_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_ks");
	shininess_Uniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_shininess");

	LKeyPress_Uniform = gl.getUniformLocation(shaderProgramObject_RRJ, "u_LKey");


	spotLightDirectionUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_spotLightDirection");
	spotLightCutoffUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_spotLightCutoff");
	spotLightExponentUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_spotLightExponent");
	constantAttenuationUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_constantAttenuation");
	linearAttenuationUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_linearAttenuation");
	quadraticAttenuationUniform_RRJ = gl.getUniformLocation(shaderProgramObject_RRJ, "u_quadraticAttenuation");


	/********** Sphere Position and Normal **********/

	myMakeSphere(2.0, STACK, SLICES);

	vao_Sphere_RRJ = gl.createVertexArray();
	gl.bindVertexArray(vao_Sphere_RRJ);

		/********** Position **********/
		vbo_Sphere_Position_RRJ = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
		gl.bufferData(gl.ARRAY_BUFFER, sphere_Position_RRJ, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Normal **********/
		vbo_Sphere_Normal_RRJ = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Sphere_Normal_RRJ);
		gl.bufferData(gl.ARRAY_BUFFER, sphere_Normal_RRJ, gl.STATIC_DRAW);
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_NORMAL, 
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_NORMAL);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);


		/********** Index **********/
		vbo_Sphere_Index_RRJ = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index_RRJ);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphere_Index_RRJ, gl.STATIC_DRAW);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

	gl.bindVertexArray(null);

	
	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.disable(gl.CULL_FACE);
	
	gl.clearDepth(1.0);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix_RRJ = mat4.create();
}



function uninitialize(){


	if(vbo_Sphere_Index_RRJ){
		gl.deleteBuffer(vbo_Sphere_Index_RRJ);
		vbo_Sphere_Index_RRJ = 0;
	}

	if(vbo_Sphere_Normal_RRJ){
		gl.deleteBuffer(vbo_Sphere_Normal_RRJ);
		vbo_Sphere_Normal_RRJ = 0;
	}


	if(vbo_Sphere_Position_RRJ){
		gl.deleteBuffer(vbo_Sphere_Position_RRJ);
		vbo_Sphere_Position_RRJ = 0;
	}

	if(vao_Sphere_RRJ){
		gl.deleteVertexArray(vao_Sphere_RRJ);
		vao_Sphere_RRJ = 0;
	}

	if(shaderProgramObject_RRJ){

		gl.useProgram(shaderProgramObject_RRJ);

			if(fragmentShaderObject_RRJ){
				gl.detachShader(shaderProgramObject_RRJ, fragmentShaderObject_RRJ);
				gl.deleteShader(fragmentShaderObject_RRJ);
				fragmentShaderObject_RRJ = 0;
				console.log("Fragment Shader Detach and Deleted!!\n");
			}

			if(vertexShaderObject_RRJ){
				gl.detachShader(shaderProgramObject_RRJ, vertexShaderObject_RRJ);
				gl.deleteShader(vertexShaderObject_RRJ);
				vertexShaderObject_RRJ = 0;
				console.log("Vertex Shader Detach and Deleted!!\n");
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject_RRJ);
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

	gl.viewport(0, 0, canvas_RRJ.width, canvas_RRJ.height);

	 mat4.identity(perspectiveProjectionMatrix_RRJ);
	 mat4.perspective(perspectiveProjectionMatrix_RRJ, 
	 				45.0,
	 				parseFloat(canvas_RRJ.width) / parseFloat(canvas_RRJ.height),
	 				0.1,
	 				100.0);
}



function draw(){

	var modelMatrix = mat4.create();
	var viewMatrix = mat4.create();

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject_RRJ);


		/********** Sphere ***********/
		mat4.identity(modelMatrix);
		mat4.identity(viewMatrix);
		mat4.translate(modelMatrix, modelMatrix, [0.0, 0.0, -5.0]);
		mat4.rotateX(modelMatrix, modelMatrix, degToRad(90.0));
		//mat4.rotateY(modelMatrix, modelMatrix, degToRad(angle_Sphere_RRJ));
		mat4.rotateZ(modelMatrix, modelMatrix, degToRad(angle_Sphere_RRJ));
		
		gl.uniformMatrix4fv(modelMatrix_Uniform_RRJ, false, modelMatrix);
		gl.uniformMatrix4fv(viewMatrix_Uniform_RRJ, false, viewMatrix);
		gl.uniformMatrix4fv(projectionMatrix_Uniform_RRJ, false, perspectiveProjectionMatrix_RRJ)
		



		if(bLights_RRJ == true){
			gl.uniform1i(LKeyPress_Uniform, 1);

			gl.uniform3fv(la_Uniform_RRJ, lightAmbient_RRJ);
			gl.uniform3fv(ld_Uniform_RRJ, lightDiffuse_RRJ);
			gl.uniform3fv(ls_Uniform_RRJ, lightSpecular_RRJ);
			gl.uniform4fv(lightPosition_Uniform_RRJ, lightPosition_RRJ);

			gl.uniform3fv(ka_Uniform_RRJ, materialAmbient_RRJ);
			gl.uniform3fv(kd_Uniform_RRJ, materialDiffuse_RRJ);
			gl.uniform3fv(ks_Uniform_RRJ, materialSpecular_RRJ);
			gl.uniform1f(shininess_Uniform_RRJ, materialShininess_RRJ);	


			gl.uniform4fv(spotLightDirectionUniform_RRJ, spotLightDirection_RRJ);
			gl.uniform1f(spotLightCutoffUniform_RRJ, spotLightCutoff_RRJ);
			gl.uniform1f(spotLightExponentUniform_RRJ, spotLightExponent_RRJ);

			gl.uniform1f(constantAttenuationUniform_RRJ, constantAttenuation_RRJ);
			gl.uniform1f(linearAttenuationUniform_RRJ, linearAttenuation_RRJ);
			gl.uniform1f(quadraticAttenuationUniform_RRJ, quadraticAttenuation_RRJ);	

		}
		else
			gl.uniform1i(LKeyPress_Uniform, 0);


		gl.bindVertexArray(vao_Sphere_RRJ);

			gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index_RRJ);
			gl.drawElements(gl.TRIANGLES, (STACK) * (SLICES) * 6, gl.UNSIGNED_SHORT, 0);

		gl.bindVertexArray(null);

	


	gl.useProgram(null);

	update();

	requestAnimationFrame(draw, canvas_RRJ);
}

function update(){

	angle_Sphere_RRJ = angle_Sphere_RRJ + 0.3;

	if(angle_Sphere_RRJ > 360.0)
		angle_Sphere_RRJ = 0.0;
}







function myMakeSphere(fRadius, iStack, iSlices){


	sphere_Position_RRJ = new Float32Array(iStack * iSlices * 3);
	sphere_Texcoord = new Float32Array(iStack * iSlices * 2);
	sphere_Normal_RRJ = new Float32Array(iStack * iStack * 3);	
	sphere_Index_RRJ = new Uint16Array((iStack) * (iSlices) * 6);

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

			sphere_Position_RRJ[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Position_RRJ[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Position_RRJ[(i * iSlices * 3)+ (j * 3) + 2] = z;

			//zconsole.log(i + "/" + j + "   " + x + "/" + y + "/" + z);

			
			sphere_Normal_RRJ[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Normal_RRJ[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Normal_RRJ[(i * iSlices * 3)+ (j * 3) + 2] = z;

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


	 			sphere_Index_RRJ[index] = topLeft;
	 			sphere_Index_RRJ[index + 1] = bottomLeft;
	 			sphere_Index_RRJ[index + 2] = topRight;

	 			sphere_Index_RRJ[index + 3] = topRight;
	 			sphere_Index_RRJ[index + 4] = bottomLeft;
	 			sphere_Index_RRJ[index + 5] = bottomRight;

 			}
 			else{

	 			var topLeft = (i * iSlices) + j;
	 			var bottomLeft = ((i + 1) * iSlices) +(j);
	 			var topRight = topLeft + 1;
	 			var bottomRight = bottomLeft + 1;


	 			sphere_Index_RRJ[index] = topLeft;
	 			sphere_Index_RRJ[index + 1] = bottomLeft;
	 			sphere_Index_RRJ[index + 2] = topRight;

	 			sphere_Index_RRJ[index + 3] = topRight;
	 			sphere_Index_RRJ[index + 4] = bottomLeft;
	 			sphere_Index_RRJ[index + 5] = bottomRight;
 			}

 			

 			index = index + 6;


 		}
 		

 	}
}


function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}