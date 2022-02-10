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
var mvpUniform_RRJ;

//For Projection Matrix;
var perspectiveProjectionMatrix_RRJ;



//For Sphere
var vao_Sphere_RRJ;
var vbo_Sphere_Position_RRJ;
var vbo_Sphere_Index_RRJ;

const STACK_RRJ = 30;
const SLICES_RRJ = 30;

var sphere_Position_RRJ;
var sphere_Texcoord_RRJ;
var sphere_Normal_RRJ;
var sphere_Index_RRJ;
var angle_Sphere_RRJ = 0.0;


//Stack
var my_ModelViewStack_RRJ = new Float32Array(32 * 4 * 4);
var iTop_RRJ = 0;


var shoulder_RRJ = 0;
var elbow_RRJ = 0;
const  AMC_CLK_RRJ = 1;
const  AMC_ANTICLK_RRJ = 2;
var  AMC_ROTATION_RRJ = AMC_CLK_RRJ;

function main(){

	canvas_RRJ = document.getElementById("29-RoboticArm-RRJ");
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



		//E
		case 69:
			if(AMC_ROTATION_RRJ == AMC_CLK_RRJ)
				elbow_RRJ = (elbow_RRJ	+ 3) %360;
			else
				elbow_RRJ = (elbow_RRJ - 3) % 360;
			break;


		//S
		case 83:
			if(AMC_ROTATION_RRJ == AMC_CLK_RRJ)
				shoulder_RRJ = (shoulder_RRJ + 3) % 360;
			else
				shoulder_RRJ = (shoulder_RRJ - 3) % 360;
			break;



		//R
		case 82:
			if(AMC_ROTATION_RRJ == AMC_CLK_RRJ)
				AMC_ROTATION_RRJ = AMC_ANTICLK_RRJ;
			else
				AMC_ROTATION_RRJ = AMC_CLK_RRJ;
			break;


		//F
		case 70:
			toggleFullScreen();
			break;

	}
	//console.log(event);

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
		"in vec4 vColor;" +
		"out vec4 outColor;" +
		"uniform mat4 u_mvp_matrix;" +
		"void main(void){" +
			"gl_Position = u_mvp_matrix * vPosition;" +
			"outColor = vColor;" +
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
		"in vec4 outColor;" +
		"out vec4 FragColor;" +
		"void main(){" +
			"FragColor = outColor;" +
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
	gl_RRJ.bindAttribLocation(shaderProgramObject_RRJ, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");


	gl_RRJ.linkProgram(shaderProgramObject_RRJ);

	
	if(!gl_RRJ.getProgramParameter(shaderProgramObject_RRJ, gl_RRJ.LINK_STATUS)){
		var error = gl_RRJ.getProgramInfoLog(shaderProgramObject_RRJ);
		if(error.length > 0){
			alert("Shader Program Linking Error: " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform_RRJ = gl_RRJ.getUniformLocation(shaderProgramObject_RRJ, "u_mvp_matrix");



	/********** Sphere Position and Normal **********/

	myMakeSphere(0.50, STACK_RRJ, SLICES_RRJ);

	vao_Sphere_RRJ = gl_RRJ.createVertexArray();
	gl_RRJ.bindVertexArray(vao_Sphere_RRJ);

		/********** Position **********/
		vbo_Sphere_Position_RRJ = gl_RRJ.createBuffer();
		gl_RRJ.bindBuffer(gl_RRJ.ARRAY_BUFFER, vbo_Sphere_Position_RRJ);
		gl_RRJ.bufferData(gl_RRJ.ARRAY_BUFFER, sphere_Position_RRJ, gl_RRJ.STATIC_DRAW);
		gl_RRJ.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl_RRJ.FLOAT,
							false,
							0, 0);
		gl_RRJ.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl_RRJ.bindBuffer(gl_RRJ.ARRAY_BUFFER, null);



		/********** Index **********/
		vbo_Sphere_Index_RRJ = gl_RRJ.createBuffer();
		gl_RRJ.bindBuffer(gl_RRJ.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index_RRJ);
		gl_RRJ.bufferData(gl_RRJ.ELEMENT_ARRAY_BUFFER, sphere_Index_RRJ, gl_RRJ.STATIC_DRAW);
		gl_RRJ.bindBuffer(gl_RRJ.ELEMENT_ARRAY_BUFFER, null);

	gl_RRJ.bindVertexArray(null);

	
	gl_RRJ.enable(gl_RRJ.DEPTH_TEST);
	gl_RRJ.depthFunc(gl_RRJ.LEQUAL);

	gl_RRJ.disable(gl_RRJ.CULL_FACE);
	
	gl_RRJ.clearDepth(1.0);

	gl_RRJ.clearColor(0.0, 0.0, 0.0, 1.0);
	

	perspectiveProjectionMatrix_RRJ = mat4.create();
}



function uninitialize(){


	if(vbo_Sphere_Index_RRJ){
		gl_RRJ.deleteBuffer(vbo_Sphere_Index_RRJ);
		vbo_Sphere_Index_RRJ = 0;
	}


	if(vbo_Sphere_Position_RRJ){
		gl_RRJ.deleteBuffer(vbo_Sphere_Position_RRJ);
		vbo_Sphere_Position_RRJ = 0;
	}

	if(vao_Sphere_RRJ){
		gl_RRJ.deleteVertexArray(vao_Sphere_RRJ);
		vao_Sphere_RRJ = 0;
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

	var modelViewMatrix_RRJ = mat4.create();
	var modelViewProjectionMatrix_RRJ = mat4.create();


	gl_RRJ.clear(gl_RRJ.COLOR_BUFFER_BIT | gl_RRJ.DEPTH_BUFFER_BIT);


	gl_RRJ.useProgram(shaderProgramObject_RRJ);



		
		/********** Sphere ***********/
		mat4.identity(modelViewMatrix_RRJ);
		mat4.identity(modelViewProjectionMatrix_RRJ);

		mat4.translate(modelViewMatrix_RRJ, modelViewMatrix_RRJ, [0.0, 0.0, -12.0]);

		


		mat4.rotateZ(modelViewMatrix_RRJ, modelViewMatrix_RRJ, degToRad(shoulder_RRJ));
		mat4.translate(modelViewMatrix_RRJ, modelViewMatrix_RRJ, [1.0, 0.0, 0.0]);
		my_glPushMatrix(modelViewMatrix_RRJ);


		mat4.scale(modelViewMatrix_RRJ, modelViewMatrix_RRJ, [2.0, 0.50, 1.0]);

		mat4.multiply(modelViewProjectionMatrix_RRJ, perspectiveProjectionMatrix_RRJ, modelViewMatrix_RRJ);

		
		
		gl_RRJ.uniformMatrix4fv(mvpUniform_RRJ, false, modelViewProjectionMatrix_RRJ);

		gl_RRJ.bindVertexArray(vao_Sphere_RRJ);

			//gl_RRJ.drawArrays(gl_RRJ.POINTS, 0, 30*30);
			gl_RRJ.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 0.50, 0.350, 0.05);
			gl_RRJ.bindBuffer(gl_RRJ.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index_RRJ);
			gl_RRJ.drawElements(gl_RRJ.TRIANGLES, (STACK_RRJ) * (SLICES_RRJ) * 6, gl_RRJ.UNSIGNED_SHORT, 0);

		gl_RRJ.bindVertexArray(null);


		//Elbow
		mat4.identity(modelViewMatrix_RRJ);
		modelViewMatrix_RRJ = my_glPopMatrix();

		mat4.translate(modelViewMatrix_RRJ, modelViewMatrix_RRJ, [1.0, 0.0, 0.0]);
		mat4.rotateZ(modelViewMatrix_RRJ, modelViewMatrix_RRJ, degToRad(elbow_RRJ));
		mat4.translate(modelViewMatrix_RRJ, modelViewMatrix_RRJ, [1.0, 0.0, 0.0]);


		mat4.scale(modelViewMatrix_RRJ, modelViewMatrix_RRJ, [2.0, 0.50, 1.0]);

		mat4.multiply(modelViewProjectionMatrix_RRJ, perspectiveProjectionMatrix_RRJ, modelViewMatrix_RRJ);

		
		
		gl_RRJ.uniformMatrix4fv(mvpUniform_RRJ, false, modelViewProjectionMatrix_RRJ);

		gl_RRJ.bindVertexArray(vao_Sphere_RRJ);

			//gl_RRJ.drawArrays(gl_RRJ.POINTS, 0, 30*30);
			gl_RRJ.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 0.50, 0.350, 0.05);
			gl_RRJ.bindBuffer(gl_RRJ.ELEMENT_ARRAY_BUFFER, vbo_Sphere_Index_RRJ);
			gl_RRJ.drawElements(gl_RRJ.TRIANGLES, (STACK_RRJ) * (SLICES_RRJ) * 6, gl_RRJ.UNSIGNED_SHORT, 0);

		gl_RRJ.bindVertexArray(null);



	gl_RRJ.useProgram(null);


	requestAnimationFrame_RRJ(draw, canvas_RRJ);
}




function myMakeSphere(fRadius, iStack, iSlices){


	sphere_Position_RRJ = new Float32Array(iStack * iSlices * 3);
	sphere_Texcoord_RRJ = new Float32Array(iStack * iSlices * 2);
	sphere_Normal_RRJ = new Float32Array(iStack * iStack * 3);	
	sphere_Index_RRJ = new Uint16Array((iStack) * (iSlices) * 6);

	var longitude_RRJ;
	var latitude_RRJ;
	var factorLat_RRJ = (2.0 * Math.PI) / (iStack);
	var factorLon_RRJ = Math.PI / (iSlices-1);

	for(var i = 0; i < iStack; i++){
		
		latitude_RRJ = -Math.PI + i * factorLat_RRJ;


		for(var j = 0; j < iSlices; j++){

			longitude_RRJ = (Math.PI) - j * factorLon_RRJ;

			//console.log(i + "/" + j + ": " + latitude_RRJ + "/" + longitude_RRJ);

			var x = fRadius * Math.sin(longitude_RRJ) * Math.cos((latitude_RRJ));
			var y = fRadius * Math.sin(longitude_RRJ) * Math.sin((latitude_RRJ));
			var z = fRadius * Math.cos((longitude_RRJ));

			sphere_Position_RRJ[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Position_RRJ[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Position_RRJ[(i * iSlices * 3)+ (j * 3) + 2] = z;

			//zconsole.log(i + "/" + j + "   " + x + "/" + y + "/" + z);

			
			sphere_Normal_RRJ[(i * iSlices * 3)+ (j * 3) + 0] = x;
			sphere_Normal_RRJ[(i * iSlices * 3)+ (j * 3) + 1] = y;
			sphere_Normal_RRJ[(i * iSlices * 3)+ (j * 3) + 2] = z;

		}
	}


	var index_RRJ = 0;
 	for(var i = 0; i < iStack ; i++){
 		for(var j = 0; j < iSlices ; j++){


 			if(i == (iStack - 1)){

 				var topLeft_RRJ = (i * iSlices) + j;
	 			var bottomLeft_RRJ = ((0) * iSlices) +(j);
	 			var topRight_RRJ = topLeft_RRJ + 1;
	 			var bottomRight_RRJ = bottomLeft_RRJ + 1;


	 			sphere_Index_RRJ[index_RRJ] = topLeft_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 1] = bottomLeft_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 2] = topRight_RRJ;

	 			sphere_Index_RRJ[index_RRJ + 3] = topRight_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 4] = bottomLeft_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 5] = bottomRight_RRJ;

 			}
 			else{

	 			var topLeft_RRJ = (i * iSlices) + j;
	 			var bottomLeft_RRJ = ((i + 1) * iSlices) +(j);
	 			var topRight_RRJ = topLeft_RRJ + 1;
	 			var bottomRight_RRJ = bottomLeft_RRJ + 1;


	 			sphere_Index_RRJ[index_RRJ] = topLeft_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 1] = bottomLeft_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 2] = topRight_RRJ;

	 			sphere_Index_RRJ[index_RRJ + 3] = topRight_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 4] = bottomLeft_RRJ;
	 			sphere_Index_RRJ[index_RRJ + 5] = bottomRight_RRJ;
 			}

 			

 			index_RRJ = index_RRJ + 6;


 		}
 		

 	}
}


function degToRad(angle){
	return(angle * (Math.PI / 180.0));
}



function my_glPushMatrix(matrix){

	for(var  i = 0; i < 16; i++){
		my_ModelViewStack_RRJ[(iTop_RRJ * 16) + i] = matrix[i];
	}
	iTop_RRJ = iTop_RRJ + 1;

}



function my_glPopMatrix(){

	var temp_RRJ = mat4.create();
	mat4.identity(temp_RRJ);


	iTop_RRJ = iTop_RRJ - 1;
	if(iTop_RRJ < 0)
		iTop_RRJ = 0;

	for(var  i = 0; i < 16; i++){
		temp_RRJ[i] = my_ModelViewStack_RRJ[(iTop_RRJ * 16) + i];
		my_ModelViewStack_RRJ[(iTop_RRJ * 16) + i] = 0;
	}

	

	return(temp_RRJ);
}