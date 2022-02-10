var canvas = null;
var gl = null;
var bIsFullScreen = false;
var canvas_original_width = 0;
var canvas_original_height = 0;

const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3
};


//For Shader
var vertexShaderObject;
var fragmentShaderObject;
var shaderProgramObject;

//For Uniform
var mvpUniform;

//For matrix
var perspectiveProjectionMatrix;


//For Triangle
var vao_Triangle;
var vbo_Triangle_Position;
var vbo_Triangle_Color;

//For Wand
var vao_Wand;
var vbo_Wand_Color;
var vbo_Wand_Position;


//For InCircle
var vao_Circle;
var vbo_Circle_Position;
var vbo_Circle_Color;

var incircle_Position = new Float32Array(3000 * 3);

var circle_Color = new Float32Array(3000 * 3);
var incircle_Center = new Float32Array(3);
var incircle_Radius;


//For Animation
var Tri_X = 0.001;
var Tri_Y = 0.001;
var Cir_X = 0.001;
var Cir_Y = 0.001;
var Wand_Y = 0.001;
var angle = 0.0;



//For Starting Animation we need requestAnimationFrame()

var requestAnimationFrame = 
	window.requestAnimationFrame || window.webkitRequestAnimationFrame ||
	window.mozRequestAnimationFrame || window.oRequestAnimationFrame || 
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

function main(){

	canvas = document.getElementById("12-DeathlyHallows-RRJ");
	if(!canvas)
		console.log("Obtaining Canvas Failed!!\n");
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
		document.wegkitFullscreenElement || 
		document.mozFullScreenElement ||
		document.msFullscreenElement ||
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
		else if(document.mozCancelFullScreen)
			document.mozCancelFullScreen();
		else if(document.webkitExitFullscreen)
			document.webkitExitFullscreen();
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

	var szVertexShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"in vec4 vPosition;" +
		"in vec4 vColor;" + 
		"out vec4 outColor;" +
 		"uniform mat4 u_mvp_matrix;" +
		"void main(){" +
			"gl_Position = u_mvp_matrix * vPosition;" +
			"outColor = vColor;" + 
		"}";

	gl.shaderSource(vertexShaderObject, szVertexShaderSourceCode);
	gl.compileShader(vertexShaderObject);

	var shaderCompileStatus = gl.getShaderParameter(vertexShaderObject, gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(vertexShaderObject);
		if(error.length > 0){
			alert("VertexShader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}


	fragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
	var szFragmentShaderSourceCode = 
		"#version 300 es" +
		"\n" +
		"precision highp float;" +
		"in vec4 outColor;" +
		"out vec4 FragColor;" +
		"void main() {" +
			"FragColor = outColor;" +
		"}";

	gl.shaderSource(fragmentShaderObject, szFragmentShaderSourceCode);
	gl.compileShader(fragmentShaderObject);

	shaderCompileStatus = gl.getShaderParameter(fragmentShaderObject,  gl.COMPILE_STATUS);

	if(shaderCompileStatus == false){
		var error = gl.getShaderInfoLog(fragmentShaderObject);
		if(error.length > 0){
			alert("Fragment Shader Compilation Error : " + error);
			uninitialize();
			window.close();

		}
	}



	shaderProgramObject = gl.createProgram();

	gl.attachShader(shaderProgramObject, vertexShaderObject);
	gl.attachShader(shaderProgramObject, fragmentShaderObject);

	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
	gl.bindAttribLocation(shaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");

	gl.linkProgram(shaderProgramObject);

	var programLinkStatus = gl.getProgramParameter(shaderProgramObject, gl.LINK_STATUS);

	if(programLinkStatus == false){
		var error = gl.getProgramInfoLog(shaderProgramObject);
		if(error.length > 0){
			alert("Program Linking Error : " + error);
			uninitialize();
			window.close();
		}
	}


	mvpUniform = gl.getUniformLocation(shaderProgramObject, "u_mvp_matrix");



	/********** Position and Color **********/
	
	var triangle_Position = new Float32Array([
						0.0, 0.70, 0.0,
						-0.70, -0.70, 0.0,
						0.70, -0.70, 0.0,
						]);
	
	var triangle_Color = new Float32Array([
						1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						1.0, 1.0, 1.0,
						]);
		

	var X = (triangle_Position[6] + triangle_Position[3]) / 2.0;

	var wand_Position = new Float32Array([
						triangle_Position[0], triangle_Position[1], triangle_Position[2],
						X, triangle_Position[7], 0.0
						]);
	

	var wand_Color = new Float32Array([
						1.0, 1.0, 1.0,
						1.0, 1.0, 1.0
					]);
		

	/********** To Calculate InCircle Radius and Center **********/
	calculation(triangle_Position);
	
	fillCircle_Position(incircle_Position, 1);



	
	/********** Triangle **********/
	vao_Triangle = gl.createVertexArray();
	gl.bindVertexArray(vao_Triangle);

		/********** Position **********/
		vbo_Triangle_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Triangle_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
				triangle_Position,
				gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Color **********/
		vbo_Triangle_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Triangle_Color);
		gl.bufferData(gl.ARRAY_BUFFER, 
				triangle_Color,
				gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	/********** Wand **********/
	vao_Wand = gl.createVertexArray();
	gl.bindVertexArray(vao_Wand);

		/********** Position **********/
		vbo_Wand_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Wand_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
					wand_Position,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Color **********/
		vbo_Wand_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Wand_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
					wand_Color,
					gl.STATIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

	gl.bindVertexArray(null);


	/********** Circle **********/
	vao_Circle = gl.createVertexArray();
	gl.bindVertexArray(vao_Circle);

		/********** Position **********/
		vbo_Circle_Position = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Circle_Position);
		gl.bufferData(gl.ARRAY_BUFFER,
				incircle_Position,
				gl.STATIC_DRAW);
		
		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION,
							3,
							gl.FLOAT,
							false,
							0, 0);

		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);

		/********** Color **********/
		/*vbo_Circle_Color = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vbo_Circle_Color);
		gl.bufferData(gl.ARRAY_BUFFER,
				circle_Color,
				gl.DYNAMIC_DRAW);

		gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_COLOR,
							3,
							gl.FLOAT,
							false,
							0, 0);
		gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_COLOR);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);*/
		gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 1.0, 1.0, 1.0);

	gl.bindVertexArray(null);



	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);

	gl.clearColor(0.0, 0.0, 0.0, 1.0);

	perspectiveProjectionMatrix = mat4.create();
}


function uninitialize(){

	console.log("in uninitialize() !!\n");

	if(vbo_Circle_Color){
		gl.deleteBuffer(vbo_Circle_Color);
		vbo_Circle_Color = 0;
	}

	if(vbo_Circle_Position){
		gl.deleteBuffer(vbo_Circle_Position);
		vbo_Circle_Position = 0;
	}

	if(vao_Circle){
		gl.deleteVertexArray(vao_Circle);
		vao_Circle = 0;
	}

	if(vbo_Wand_Color){
		gl.deleteBuffer(vbo_Wand_Color);
		vbo_Wand_Color = 0;
	}

	if(vbo_Wand_Position){
		gl.deleteBuffer(vbo_Wand_Position);
		vbo_Wand_Position = 0;
	}

	if(vao_Wand){
		gl.deleteVertexArray(vao_Wand);
		vao_Wand = 0;
	}

	

	if(vbo_Triangle_Color){
		gl.deleteBuffer(vbo_Triangle_Color);
		vbo_Triangle_Color = 0;
	}

	if(vbo_Triangle_Position){
		gl.deleteBuffer(vbo_Triangle_Position);
		vbo_Triangle_Position = 0;
	}

	if(vao_Triangle){
		gl.deleteVertexArray(vao_Triangle);
		vao_Triangle = 0;
	}


	if(shaderProgramObject){

		gl.useProgram(shaderProgramObject);

			if(fragmentShaderObject){
				gl.detachShader(shaderProgramObject, fragmentShaderObject);
				gl.deleteShader(fragmentShaderObject);
				fragmentShaderObject = null;
			}

			if(vertexShaderObject){
				gl.detachShader(shaderProgramObject, vertexShaderObject);
				gl.deleteShader(vertexShaderObject);
				vertexShaderObject = null;
			}

		gl.useProgram(null);
		gl.deleteProgram(shaderProgramObject);
		shaderProgramObject = null;
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

	mat4.perspective(perspectiveProjectionMatrix, 
					45.0,
					parseFloat(canvas.width) / parseFloat(canvas.height),
					0.1,
					100.0);
}

function draw(){

	var modelViewMatrix = mat4.create();
	var modelViewProjectionMatrix = mat4.create();
	

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	gl.useProgram(shaderProgramObject);

		
		/********** Triangle **********/
		mat4.translate(modelViewMatrix, modelViewMatrix, [5.0 - Tri_X, -2.50 + Tri_Y, -6.0])

		if (Tri_X < 5.0 && Cir_X < 5.0)
				mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angle));

		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.lineWidth(1.5);

		gl.bindVertexArray(vao_Triangle);

		gl.drawArrays(gl.LINE_LOOP, 0, 3);

		gl.bindVertexArray(null);




		/********** Circle **********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);

		mat4.translate(modelViewMatrix, modelViewMatrix, [-5.0 + Cir_X, -2.50 + Cir_Y, -6.0])
		mat4.rotateY(modelViewMatrix, modelViewMatrix, degToRad(angle));
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);
		
		gl.bindVertexArray(vao_Circle);

		gl.drawArrays(gl.LINE_LOOP, 0, 3000);

		gl.bindVertexArray(null);


		/********** Wand **********/
		mat4.identity(modelViewMatrix);
		mat4.identity(modelViewProjectionMatrix);

		mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 2.50 - Wand_Y, -6.0]);
		mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);
		
		gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

		gl.bindVertexArray(vao_Wand);
		gl.drawArrays(gl.LINES, 0, 2);
		gl.bindVertexArray(null);



	gl.useProgram(null);


	Tri_X = Tri_X + 0.008;
	Tri_Y = Tri_Y + 0.004;

	if (Tri_X > 5.0 && Tri_Y > 2.50) {
		Tri_X = 5.0;
		Tri_Y = 2.5;
	}

	Cir_X = Cir_X + 0.008;
	Cir_Y = Cir_Y + 0.004;

	if (Cir_X > 5.0 && Cir_Y > 2.5) {
		Cir_X = 5.0;
		Cir_Y = 2.5;
	}

	Wand_Y = Wand_Y + 0.004;
	if (Wand_Y > 2.5)
		Wand_Y = 2.5;

	angle = angle + 2.0;


	requestAnimationFrame(draw, canvas);
}




function calculation(arr){
	var a, b, c;
	var s;

	//Distance Formula
	a = Math.sqrt(Math.pow((arr[6] - arr[3]), 2) + Math.pow((arr[7] - arr[4]), 2));
	b = Math.sqrt(Math.pow((arr[6] - arr[0]), 2) + Math.pow((arr[7] - arr[1]), 2));
	c = Math.sqrt(Math.pow((arr[3] - arr[0]), 2) + Math.pow((arr[4] - arr[1]), 2));

	s = (a + b + c) / 2;

	incircle_Radius = (Math.sqrt(s * (s - a) * (s - b) * (s - c)) / s);

	incircle_Center[0] = (a * arr[0] + b * arr[3] + c * arr[6]) / (a+ b+ c);
	incircle_Center[1] = (a * arr[1] + b * arr[4] + c * arr[7]) / (a+ b+ c);
	incircle_Center[2] = 0.0;


	console.log("Incircle_Radius: %f\n", incircle_Radius);
	console.log("InCenter x: %f      y: %f      z: %f     \n", incircle_Center[0], incircle_Center[1], incircle_Center[2]);

}


function fillCircle_Position(arr, iFlag){
	

	var index;

	if(iFlag == 1){
		//InCircle
		for(var i = 0; i < 9000; i = i + 3){
			var x = (2.0 * Math.PI * i / 3000);
			arr[i] = (incircle_Radius * Math.cos(x)) + incircle_Center[0];
			arr[i + 1] = (incircle_Radius * Math.sin(x)) + incircle_Center[1];
			arr[i + 2] = 0.0;

			index = i;
		}

		console.log(index);
	}
	
}

function degToRad(angle){
	return(angle * Math.PI / 180.0);
}


