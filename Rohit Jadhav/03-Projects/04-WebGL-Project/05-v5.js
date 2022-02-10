var bIsFullScreen = false;
var canvas_original_width = 0;
var canvas_original_height = 0;

const WebGLMacros = {
	AMC_ATTRIBUTE_POSITION:0,
	AMC_ATTRIBUTE_COLOR:1,
	AMC_ATTRIBUTE_NORMAL:2,
	AMC_ATTRIBUTE_TEXCOORD0:3,
	AMC_ATTRIBUTE_TEXCOORD1:4,
	AMC_ATTRIBUTE_MODELMATRIX:5,
};


const SceneConst = {
	//Choice
	NORMAL:0,
	REFLECT_TEX: 1,
	REFRACT_TEX: 2,

	//Mode
	MEMORY: 3,
	REALITY:4,
	SHADOW_MAP: 5,
};

const Lights = {

	DEPTH_MAP: 0,
	SHADOW: 1,
	SINGLE_LIGHT: 2,
	FOG: 3,
	POINT_LIGHT:4,
};






//For Audio
var audio_Memory;
var audio_Reality;

//For Test
var testMouse = 0;


var camPos = new Float32Array([

		435.4332275390625, 96.04071044921875, 738.7861938476562,
		73.6368179321289, 61.90877914428711, 326.4933166503906,
		 -33.198665618896484, 97.04361724853516, 28.207141876220703,
		554.1475830078125, 454.499267578125, -633.2557983398438,
		-453.4507141113281, 114.38528442382812, 187.73741149902344,
		270.9049072265625, 75.3759994506836, 87.1541976928711,


		464.2924499511719, 79.02506256103516, -80.09159851074219,
		 626.295166015625, 177.23167419433594, -602.6117553710938,

		 //8
		 621.868896484375, 62.430076599121094, -759.9530639648438,
		 264.72210693359375, 544.7759094238281, 452.39666748046875,
		  0.868896484375, 40.430076599121094, -0.9530639648438, 
		   8.281262397766113, 272.2020568847656, 74.19743347167969,




	]);


var camFront = new Float32Array([
		 0.1736481785774231, 0, -0.9848077297210693,
	]);

var camAngle = new Float32Array([
		0, -80,
		7.25, -165.75,
		-1.5, -215.75,
		-82, -88.75,
		-9, -91.25,
		-11.25, -93.5,

		50.5, -117.5,
		-7.25, -218.25,

		//8
		53.5, -230.5,
		-79.0, -47.75,
		0.5, -90.5,
		-71.75, -88,


	])


//For External
var canvas = null;
var gl = null;
var canvas_Width;
var canvas_Height;
var texture_Scene_Memory;
var texture_Scene_Reality;
var global_viewMatrix;
var pointLight_Position;

//For matrix
var gPerspectiveProjectionMatrix;



//For Camera
var gCameraPosition;
var gCameraLookingAt;
var gCameraFront;
var gCameraDirection;
var gCameraUp;

var gCameraSpeed = 2.0;
var gYawAngle = -90.0;
var gPitchAngle = 0.0;
var gLastX = 1336.0 / 2.0;
var gLastY = 768.0 / 2.0;
var gSensitivity = 0.25;





//For Water Reflection and Refraction
var gFBO_Reflection;
var gFBO_Reflection_Tex;
var gFBO_Reflection_DepthRenderbuffer;

var gFBO_Refraction;
var gFBO_Refraction_Tex;
var gFBO_Refraction_Depth_Tex;

const gFBO_Water_Tex_Width = 1366.0;
const gFBO_Water_Tex_Height = 768.0;

var gChoice_Reflection_Refraction_Uniform;
var gWaterLevelUniform;


//For Model
var gShaderProgramObject_Model_AAW;
var gModel_Tree_Memory = null;
var gModel_Tree_Reality = null;
var gModel_Lamp = null;

//For Instancing
var sceneModelMatrix;
const ASM_INSTANCED_COUNT = 25;
var gMemoryModelPos = new Float32Array(ASM_INSTANCED_COUNT * 3);
var gRealityModelPos = new Float32Array(ASM_INSTANCED_COUNT * 3);





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

async function main(){

	canvas = document.getElementById("05-v5");
	if(!canvas)
		console.log("Obtaining Canvas Failed!!\n");
	else
		console.log("Canvas Obtained!!\n");


	canvas_original_width = canvas.width;
	canvas_original_height = canvas.height;

	window.addEventListener("keydown", keyDown, false);
	window.addEventListener("click", mouseDown, false);
	window.addEventListener("resize", resize, false);

	if(testMouse == 1)
		window.addEventListener("mousemove", mouseMove, false);


	gCameraPosition = vec3.create();
	gCameraLookingAt = vec3.create();
	gCameraFront = vec3.create();
	gCameraUp = vec3.create();
	gCameraDirection = vec3.create();

	vec3.set(gCameraPosition, 0.0, gWater_Level + 3.0, 0.0);
	vec3.set(gCameraLookingAt, 0.0, 0.0, 0.0); 
	vec3.set(gCameraFront, 0.0, 0.0, -1.0); 
	vec3.set(gCameraUp, 0.0, 1.0, 0.0); 
	vec3.set(gCameraDirection, 0.0, 0.0, 0.0);


	//Audio
	audio_Memory = document.createElement("audio");
	audio_Memory.src = "two.mp3";
	audio_Memory.play();

	audio_Reality = document.createElement("audio");
	audio_Reality.src = "Reality.wav";


	initialize();

	//new 
    	var ASM_modelMatrixValues = new Float32Array(16 * ASM_INSTANCED_COUNT);
	var ASM_translateMatrix = mat4.create();
	var ASM_scaleMatrix = mat4.create();
	var ASM_modelViewMatrix = mat4.create();


	//Memory
	for (var i = 0; i < ASM_INSTANCED_COUNT - 10; i++)
	{

		var x, y, z;
		x = gMemoryModelPos[i * 3 + 0];
		y = gMemoryModelPos[i * 3 + 1];
		z = gMemoryModelPos[i * 3 + 2];

		mat4.translate(ASM_translateMatrix, ASM_translateMatrix, [x, y, z]);
		mat4.scale(ASM_scaleMatrix, ASM_scaleMatrix, [8.2, 5.2, 8.2]);
		mat4.multiply(ASM_modelViewMatrix, ASM_translateMatrix, ASM_scaleMatrix);
		for (var j = 0; j < 16; j++)
		{
			ASM_modelMatrixValues[(i * 16) + j] = ASM_modelViewMatrix[j];	

		}		
		mat4.identity(ASM_translateMatrix);
		mat4.identity(ASM_scaleMatrix);
		mat4.identity(ASM_modelViewMatrix);

	}


	gModel_Tree_Memory = new Model();
	sceneModelMatrix = mat4.create();
	const response = await fetch('04-Model/tree2.obj');
	const objSource = await response.text();
	gModel_Tree_Memory.parseOBJ(objSource, true, ASM_modelMatrixValues, ASM_INSTANCED_COUNT - 10);


	//Reality
	mat4.identity(ASM_translateMatrix);
	mat4.identity(ASM_scaleMatrix);
	mat4.identity(ASM_modelViewMatrix);

	for (var i = 0; i < ASM_INSTANCED_COUNT - 5; i++)
	{

		var x, y, z;
		x = gRealityModelPos[i * 3 + 0];
		y = gRealityModelPos[i * 3 + 1];
		z = gRealityModelPos[i * 3 + 2];

		mat4.translate(ASM_translateMatrix, ASM_translateMatrix, [x, y, z]);
		mat4.scale(ASM_scaleMatrix, ASM_scaleMatrix, [20.2, 10.2, 20.2]);
		mat4.multiply(ASM_modelViewMatrix, ASM_translateMatrix, ASM_scaleMatrix);
		for (var j = 0; j < 16; j++)
		{
			ASM_modelMatrixValues[(i * 16) + j] = ASM_modelViewMatrix[j];	

		}		
		mat4.identity(ASM_translateMatrix);
		mat4.identity(ASM_scaleMatrix);
		mat4.identity(ASM_modelViewMatrix);

	}


	gModel_Tree_Reality = new Model();
	const response_reality = await fetch('04-Model/tree2.obj');
	const objSource_reality = await response_reality.text();
	gModel_Tree_Reality.parseOBJ(objSource_reality, true, ASM_modelMatrixValues, ASM_INSTANCED_COUNT - 4);



	// Lamp
	pointLight_Position = [

			148.5803985595703, 72.41515350341797, 534.0863037109375, 1,
			-550.1490478515625, 130.08404541015625, -325.2705993652344, 1, 
			36.141178131103516, 103.46526336669922, -421.6470642089844, 1,
			309.2078552246094, 83.46615600585938, -405.5843200683594, 1,
			-614.4000244140625, 99.6836929321289, -212.8313751220703, 1,
			252.9882354736328, 61.251041412353516, -188.7372589111328, 1,
			-365.4274597167969, 81.49134826660156, 261.0196228027344, 1,
			485.8980407714844, 107.91449737548828, -453.7725524902344, 1,
		  	-534.0863037109375, 113.0845718383789, 429.6784362792969, 1,
			293.1451110839844, 97.69038391113281, 534.0863037109375, 1,

		];


	mat4.identity(ASM_translateMatrix);
	mat4.identity(ASM_scaleMatrix);
	mat4.identity(ASM_modelViewMatrix);

	for (var i = 0; i < TOTAL_POINT_LIGHT; i++)
	{

		var x, y, z;
		x = pointLight_Position[i * 4 + 0];
		y = pointLight_Position[i * 4 + 1] - 22.0;
		z = pointLight_Position[i * 4 + 2];

		mat4.translate(ASM_translateMatrix, ASM_translateMatrix, [x, y, z]);
		mat4.scale(ASM_scaleMatrix, ASM_scaleMatrix, [3.2, 3.2, 3.2]);
		mat4.multiply(ASM_modelViewMatrix, ASM_translateMatrix, ASM_scaleMatrix);
		for (var j = 0; j < 16; j++)
		{
			ASM_modelMatrixValues[(i * 16) + j] = ASM_modelViewMatrix[j];	

		}		
		mat4.identity(ASM_translateMatrix);
		mat4.identity(ASM_scaleMatrix);
		mat4.identity(ASM_modelViewMatrix);

	}


	gModel_Lamp = new Model();
	const response_lamp = await fetch('04-Model/lamp.obj');
	const objSource_lamp = await response_lamp.text();
	gModel_Lamp.parseOBJ(objSource_lamp, true, ASM_modelMatrixValues, TOTAL_POINT_LIGHT);



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

		//Esc
		case 27:
			uninitialize();
			window.close();
			break;


		// case 49:
		// 	SceneNumber = Scene.DAY;	
		// 	break;	

		// case 50:
		// 	SceneNumber = Scene.NIGHT;
		// 	break;

		// case 51:
		// 	SceneNumber = Scene.END;
		// 	break;

		//W
		case 87:
		case 119:

			moveForward();
			// var temp = vec3.create();

			// vec3.multiply(temp, gCameraFront, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);
			// vec3.add(gCameraPosition, gCameraPosition, temp);

			
			//console.log("W Press", gCameraPosition);
			break;

		//S	
		case 83:
		case 115:
			moveBackward();
			// var temp = vec3.create();

			// vec3.multiply(temp, gCameraFront, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);
			// vec3.subtract(gCameraPosition, gCameraPosition,  temp);
			// //console.log("S Press", gCameraPosition);
			break;

		//A	
		case 65:
		case 97:
			moveLeft();
			//console.log("A Press")
			// var temp = vec3.create();;
			// var crossProduct = vec3.create();
			// vec3.cross(crossProduct, gCameraFront, gCameraUp);
			// vec3.normalize(crossProduct, crossProduct);
			// vec3.multiply(temp, crossProduct, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);

			// vec3.subtract(gCameraPosition, gCameraPosition, temp);
			break;


		//D	
		case 68:
		case 100:
			moveRight();
			//console.log("D Press");
			// var crossProduct = vec3.create();
			// var temp = vec3.create();

			// vec3.cross(crossProduct, gCameraFront, gCameraUp);
			// vec3.normalize(crossProduct, crossProduct);
			// vec3.multiply(temp, crossProduct, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);

			// vec3.add(gCameraPosition, gCameraPosition, temp);
			break;


		case 88:
		case 120:
			console.log(gCameraPosition);
			console.log(gCameraFront);
			console.log(gPitchAngle, gYawAngle);
			// console.log(gYawAngle);
			// console.log(gPitchAngle);
			break;

		//F
		case 70:
			toggleFullScreen();
			break;
	}
}

function mouseDown(){

}


function moveForward(){

	var temp = vec3.create();

	vec3.multiply(temp, gCameraFront, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);
	vec3.add(gCameraPosition, gCameraPosition, temp);
}

function moveBackward(){

	var temp = vec3.create();

	vec3.multiply(temp, gCameraFront, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);
	vec3.subtract(gCameraPosition, gCameraPosition,  temp);
	//console.log("S Press", gCameraPosition);

}

function moveLeft(){
	var temp = vec3.create();;
	var crossProduct = vec3.create();
	vec3.cross(crossProduct, gCameraFront, gCameraUp);
	vec3.normalize(crossProduct, crossProduct);
	vec3.multiply(temp, crossProduct, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);

	vec3.subtract(gCameraPosition, gCameraPosition, temp);
}

function moveRight(){
	var crossProduct = vec3.create();
	var temp = vec3.create();

	vec3.cross(crossProduct, gCameraFront, gCameraUp);
	vec3.normalize(crossProduct, crossProduct);
	vec3.multiply(temp, crossProduct, [gCameraSpeed, gCameraSpeed, gCameraSpeed]);

	vec3.add(gCameraPosition, gCameraPosition, temp);
}
	

function mouseMove(event){


	var xoffset = event.offsetX - gLastX;
	var yoffset = gLastY - event.offsetY;

	gLastX = event.offsetX;
	gLastY = event.offsetY;

	xoffset *= gSensitivity;
	yoffset *= gSensitivity;

	gYawAngle += xoffset;
	gPitchAngle += yoffset;

	if(gPitchAngle > 89.0)
		gPitchAngle = 89.0;

	if(gPitchAngle < -89.0)
		gPitchAngle = -89.0;


	var x = Math.cos(degToRad(gYawAngle)) * Math.cos(degToRad(gPitchAngle));
	var y = Math.sin(degToRad(gPitchAngle));
	var z = Math.sin(degToRad(gYawAngle)) * Math.cos(degToRad(gPitchAngle));

	vec3.set(gCameraDirection, x, y, z);

	vec3.normalize(gCameraDirection, gCameraDirection);

	vec3.copy(gCameraFront, gCameraDirection);

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


	// var a = gl.getSupportedExtensions();
	// //var a = gl.getExtension('WRBGL_depth_texture');
	// console.log(a);



	initialize_Terrain();


	initialize_HeightMap();

	initialize_water();

	initialize_CubeMap();

	initialize_testRect();

	initializeShader_Model_AAW();

	initialize_Fade();

	initialize_Cloud();

	initialize_PointSprite();


	LoadTexture();


	initialize_CreditRoll();


	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LEQUAL);
	gl.clearDepth(1.0);

	gl.clearColor(0.0, 0.0, 0.50, 1.0);

	gPerspectiveProjectionMatrix = mat4.create();
	global_viewMatrix = mat4.create();

	//For Memory Terrain
	draw_HeightMap(1);

	//For Reality Terrain
	draw_HeightMap(2);


	//For Transform Feedback
	draw_terrain_tf(SceneConst.MEMORY, SceneConst.NORMAL, texture_Scene_Memory);

	draw_terrain_tf(SceneConst.REALITY, SceneConst.NORMAL, texture_Scene_Reality);

}



function LoadTexture(){



	gTreeTexture_AAW = gl.createTexture();
	gTreeTexture_AAW.image = new Image();
	gTreeTexture_AAW.image.src = "04-Model/tree.png";
	gTreeTexture_AAW.image.onload = function(){

		gl.bindTexture(gl.TEXTURE_2D, gTreeTexture_AAW);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
		//gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, gTreeTexture_AAW.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}


	gLampTexture_AAW = gl.createTexture();
	gLampTexture_AAW.image = new Image();
	gLampTexture_AAW.image.src = "04-Model/lamp.png";
	gLampTexture_AAW.image.onload = function(){
		gl.bindTexture(gl.TEXTURE_2D, gLampTexture_AAW);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		//gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, gLampTexture_AAW.image);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindTexture(gl.TEXTURE_2D, null);
	}


}


function uninitialize(){


	if(gModel_Tree_Memory){
		gModel_Tree_Memory.deallocate();
		gModel_Tree_Memory = null;
	}


	if(gModel_Tree_Reality){
		gModel_Tree_Reality.deallocate();
		gModel_Tree_Reality = null;
	}



	uninitialize_Terrain();

	uninitialize_HeightMap();

	uninitialize_water();

	uninitialize_testRect();

	uninitialize_Model();

	uninitialize_CubeMap();

	uninitialize_Fade();

	uninitialize_Cloud();

	uninitialize_PointSprite();

	uninitialize_CreditRoll();

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

	canvas_Width = canvas.width;
	canvas_Height = canvas.height;

	mat4.perspective(gPerspectiveProjectionMatrix, 
					45.0,
					parseFloat(canvas.width) / parseFloat(canvas.height),
					0.1,
					4000.0);
}





const Scene = {

	NOT_START:-1,

	BLACKOUT:0,

	FADE_IN:1,

	DAY1: 2,
	DAY2: 3,
	DAY3: 4,
	DAY4: 5,
	DAY5: 6,
	DAY6: 7,
	DAY7: 8,

	DELAY: 9,

	NIGHT1: 10,
	NIGHT2: 11,

	INTRO:12,

	REALITY: 13,
	REALITY1: 14,
	REALITY2: 15,
	REALITY3: 16,

	DELAY2: 17,

	END_CRED: 18,
	ALL_END: 19,

};

SceneNumber = Scene.BLACKOUT;



var timer = 0.0;
var y_angle = 0.25;
var x_angle = 0.15;
var fade = 1.0;
var cloudFade = 0.0;


function draw(playTime){

	var miliSec = 1000.0;
	var tempCamPos = vec3.create();
	var tempCamFr = vec3.create();

	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


	if(testMouse == 1){
		// vec3.set(gCameraPosition,  camPos[0 * 3 + 0], camPos[0 * 3 + 1], camPos[0 * 3 + 2]);
		// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);

		// draw_Memory(Lights.POINT_LIGHT);

		// display_CreditRoll(10, 1.0);
		
		// draw_Reality(0);

	}
	else{
		switch(SceneNumber){

			case Scene.BLACKOUT:

				if(playTime > (13.0 * miliSec)){
					SceneNumber = Scene.FADE_IN;
					fade = 1.0;
					gCameraPosition[1] += 27.0;
					gPitchAngle += 25;
					invertPitchAngle(gPitchAngle);
				}

				break;

			case Scene.FADE_IN:


				draw_Memory(Lights.SINGLE_LIGHT, 0);
				draw_Fade(fade);

				fade -= 0.005;

				if(playTime > (25.0 * miliSec)){
					SceneNumber = Scene.DAY1;
					fade = 0.0;
				}
				break;

			case Scene.DAY1:


				gYawAngle += y_angle;
				setYawn(gYawAngle);
				draw_Memory(Lights.SINGLE_LIGHT, 0);


				if(playTime > (49.0 * miliSec)){

					SceneNumber = Scene.DAY2;	
				
					// console.log("On3 Old: ", gCameraFront);
					// console.log("On3 Old: ", gCameraPosition);
	

					vec3.set(gCameraPosition,  camPos[0 * 3 + 0], camPos[0 * 3 + 1], camPos[0 * 3 + 2]);	
					
					gPitchAngle = camAngle[0 * 2 + 0];
					gYawAngle = camAngle[0 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					

					// console.log("On3: ", gCameraFront);
					// console.log(camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					
					// console.log("On3: ", gCameraPosition);
					fade = 1.0;

				}
				if(playTime > (44.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}

				break;


			case Scene.DAY2:


				moveForward();
				draw_Memory(Lights.SINGLE_LIGHT, 0);
				

				if(playTime > (61.0 * miliSec)){
					SceneNumber = Scene.DAY3;

					vec3.set(gCameraPosition,  camPos[1 * 3 + 0], camPos[1 * 3 + 1], camPos[1 * 3 + 2]);	
					
					gPitchAngle = camAngle[1 * 2 + 0];
					gYawAngle = camAngle[1 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 1.0;

				}	
				
				if(playTime > (55.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}

				if(playTime < (53.50 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}

				break;


			case Scene.DAY3:

				draw_Memory(Lights.SINGLE_LIGHT, 0);


				if(playTime > (78.0 * miliSec)){
					SceneNumber = Scene.DAY4;

					vec3.set(gCameraPosition,  camPos[2 * 3 + 0], camPos[2 * 3 + 1], camPos[2 * 3 + 2]);	
					
					gPitchAngle = camAngle[2 * 2 + 0];
					gYawAngle = camAngle[2 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 1.0;

				}


				if(playTime > (70.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (65.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}

				break;


			case Scene.DAY4:


				gYawAngle -= y_angle;
				setYawn(gYawAngle);

				draw_Memory(Lights.SINGLE_LIGHT, 0);


				if(playTime > (95.0 * miliSec)){
					SceneNumber = Scene.DAY5;

					vec3.set(gCameraPosition,  camPos[3 * 3 + 0], camPos[3 * 3 + 1], camPos[3 * 3 + 2]);	
					
					gPitchAngle = camAngle[3 * 2 + 0];
					gYawAngle = camAngle[3 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 1.0;

				}


				if(playTime > (89.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (83.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}
				break;


			case Scene.DAY5:


				moveLeft();
				draw_Memory(Lights.SINGLE_LIGHT, 1);


				if(playTime > (113.0 * miliSec)){
					SceneNumber = Scene.DAY6;

					vec3.set(gCameraPosition,  camPos[4 * 3 + 0], camPos[4 * 3 + 1], camPos[4 * 3 + 2]);	
					
					gPitchAngle = camAngle[4 * 2 + 0];
					gYawAngle = camAngle[4 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 1.0;

				}


				if(playTime > (107.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (100.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}
				break;


			case Scene.DAY6:


				moveRight();
				draw_Memory(Lights.SINGLE_LIGHT, 0);


				if(playTime > (130.0 * miliSec)){
					SceneNumber = Scene.DAY7;

					vec3.set(gCameraPosition,  camPos[5 * 3 + 0], camPos[5 * 3 + 1], camPos[5 * 3 + 2]);	
					
					gPitchAngle = camAngle[5 * 2 + 0];
					gYawAngle = camAngle[5 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 1.0;

				}


				if(playTime > (123.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (118.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}
				break;	



			case Scene.DAY7:


				gPitchAngle += x_angle;
				invertPitchAngle(gPitchAngle);
				draw_Memory(Lights.SINGLE_LIGHT, 0);


				if(playTime > (146.0 * miliSec)){
					SceneNumber = Scene.DELAY;
					// console.log("End Day7");

					// vec3.set(gCameraPosition,  camPos[6 * 3 + 0], camPos[6 * 3 + 1], camPos[6 * 3 + 2]);	
					
					// gPitchAngle = camAngle[6 * 2 + 0];
					// gYawAngle = camAngle[6 * 2 + 1];

					// // vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					// SetCamFront(gPitchAngle, gYawAngle);

					// fade = 0.0;
				}


				if(playTime > (141 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (135.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}
				break;


			case Scene.DELAY: 

				//draw_PointSprite();

				if(playTime > (147.0 * miliSec)){
					SceneNumber = Scene.NIGHT1;
					console.log("End DELAY");

					// vec3.set(gCameraPosition,  camPos[6 * 3 + 0], camPos[6 * 3 + 1], camPos[6 * 3 + 2]);	
					
					gPitchAngle = 50.0;
					// gYawAngle = camAngle[6 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 0.0;

				}
				break;		

			case Scene.NIGHT1:


				gPitchAngle -= x_angle;
				invertPitchAngle(gPitchAngle);
				draw_Memory(Lights.POINT_LIGHT, 0);
				draw_PointSprite();		


				if(playTime > (157.0 * miliSec)){
					SceneNumber = Scene.NIGHT2;

					// vec3.set(gCameraPosition,  camPos[7 * 3 + 0], camPos[7 * 3 + 1], camPos[7 * 3 + 2]);	
					
					// gPitchAngle = camAngle[7 * 2 + 0];
					// gYawAngle = camAngle[7 * 2 + 1];

					// // vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					// SetCamFront(gPitchAngle, gYawAngle);

					fade = 0.0;
					y_angle = 0.3;

				}

				break;	



			case Scene.NIGHT2:


				if(playTime > (159.0 * miliSec)){
					gYawAngle -= y_angle;
					setYawn(gYawAngle);
				}


				draw_Memory(Lights.POINT_LIGHT, 0);
				draw_PointSprite();


				if(playTime > (175 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime > (180.0 * miliSec)){
					SceneNumber = Scene.INTRO;

					// vec3.set(gCameraPosition,  camPos[8 * 3 + 0], camPos[8 * 3 + 1], camPos[8 * 3 + 2]);	
					
					// gPitchAngle = camAngle[8 * 2 + 0];
					// gYawAngle = camAngle[8 * 2 + 1];

					// // vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					// SetCamFront(gPitchAngle, gYawAngle);

					// fade = 1.0;
					
					
					

				}


				break;	





			case Scene.INTRO:

				if(playTime < (188.0 * miliSec))
					display_CreditRoll(1, 1.0);
				else if(playTime < (192.0 * miliSec))
					display_CreditRoll(2, 1.0);
				else if(playTime < (196.0 * miliSec))
					display_CreditRoll(3, 1.0);
				else if(playTime < (200.0 * miliSec))
					display_CreditRoll(4, 1.0);
				else if(playTime < (204.0 * miliSec))
					display_CreditRoll(5, 1.0);
				else if(playTime < (208.0 * miliSec))
					display_CreditRoll(10, 1.0);
				else if(playTime < (212.0 * miliSec))
					display_CreditRoll(6, 1.0);
				else if(playTime < (214.0 * miliSec))
					display_CreditRoll(7, 1.0);
				else if(playTime < (218.0 * miliSec))
					display_CreditRoll(8, 1.0);
				else if(playTime < (220.0 * miliSec))
					display_CreditRoll(9, 1.0);
				else{
					SceneNumber = ALL_END;
				}
	

				// if(playTime > (196 * miliSec)){
				// 	SceneNumber = Scene.DELAY2;

				// 	// audio_Reality.play();

				// 	fade = 0.0;
				// }

				break;

			case Scene.DELAY2:


				if(playTime > (198.0 * miliSec)){
					SceneNumber = Scene.END_CRED;

					vec3.set(gCameraPosition,  camPos[8 * 3 + 0], camPos[8 * 3 + 1], camPos[8 * 3 + 2]);	
					
					gPitchAngle = camAngle[8 * 2 + 0];
					gYawAngle = camAngle[8 * 2 + 1];

					x_angle = 0.15;

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

				}


				break;


			case Scene.REALITY:

				gPitchAngle -= x_angle;
				invertPitchAngle(gPitchAngle);
				draw_Reality(0);

				if(playTime > (210.0 * miliSec)){

					SceneNumber = Scene.REALITY1;
					y_angle = 0.18;
				}
				break;

			case Scene.REALITY1:

				if(playTime < (215.0 * miliSec)){

					gYawAngle -= y_angle;
					setYawn(gYawAngle);
					draw_Reality(0);
					
				}

				else if(playTime < (225.0 * miliSec)){
					gYawAngle += y_angle;
					setYawn(gYawAngle);
					draw_Reality(0);
				}

				else if(playTime < (230.0 * miliSec)){
					moveForward();
					draw_Reality(0);
				}

				else if(playTime < (238.0 * miliSec)){
					moveForward();
					draw_Reality(0);
					draw_Fade(fade);
					fade += 0.01;
				}
				else{
					SceneNumber = Scene.REALITY2;

					vec3.set(gCameraPosition,  camPos[9 * 3 + 0], camPos[9 * 3 + 1], camPos[9 * 3 + 2]);	
					
					gPitchAngle = camAngle[9 * 2 + 0];
					gYawAngle = camAngle[9 * 2 + 1];

					x_angle = 0.15;

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					fade = 1.0;
					cloudFade = 0.0;


				}
				break;

			case Scene.REALITY2:


				moveLeft();
				draw_Reality(0);	
				

				if(playTime > (255.0 * miliSec)){
					SceneNumber = Scene.REALITY3;

					vec3.set(gCameraPosition,  camPos[10 * 3 + 0], camPos[10 * 3 + 1], camPos[10 * 3 + 2]);	
					
					gPitchAngle = camAngle[10 * 2 + 0];
					gYawAngle = camAngle[10 * 2 + 1];

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

					//window.addEventListener("mousemove", mouseMove, false);

					console.log("R2");
					fade = 1.0;

				}


				if(playTime > (250.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (243.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}
				

				break;


			case Scene.REALITY3:
			
				gYawAngle += y_angle;
				setYawn(gYawAngle);
				draw_Reality(1);


				if(playTime > (285.0 * miliSec)){
					SceneNumber = Scene.END;
					console.log("ED");

					vec3.set(gCameraPosition,  camPos[11 * 3 + 0], camPos[11 * 3 + 1], camPos[11 * 3 + 2]);	
					
					gPitchAngle = camAngle[11 * 2 + 0];
					gYawAngle = camAngle[11 * 2 + 1];
					
					// window.addEventListener("mousemove", mouseMove, false);

					// vec3.set(gCameraFront, camFront[0 * 3 + 0], camFront[0 * 3 + 1], camFront[0 * 3 + 2]);
					SetCamFront(gPitchAngle, gYawAngle);

				}


				if(playTime > (280.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (260.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}

				break;




			case Scene.END:

				moveBackward();
				draw_Reality(1);

				if(playTime > (310.0 * miliSec)){
					SceneNumber = Scene.END_CRED;
					console.log("ED");
				}


				if(playTime > (305.0 * miliSec)){
					draw_Fade(fade);
					fade += 0.01;
				}


				if(playTime < (290.0 * miliSec)){
					draw_Fade(fade);
					fade -= 0.01;
				}


				break;


			case Scene.END_CRED:


				if(playTime < (202.0 * miliSec))
					display_CreditRoll(4, 1.0);
				else if(playTime < (206.0 * miliSec))
					display_CreditRoll(5, 1.0);
				else if(playTime < (210.0 * miliSec))
					display_CreditRoll(10, 1.0);
				else if(playTime < (214.0 * miliSec))
					display_CreditRoll(6, 1.0);
				else if(playTime < (218.0 * miliSec))
					display_CreditRoll(7, 1.0);
				else if(playTime < (220.0 * miliSec))
					display_CreditRoll(8, 1.0);
				else if(playTime < (224.0 * miliSec))
					display_CreditRoll(9, 1.0);
				else{
					SceneNumber = ALL_END;
				}
			
				break;		





		}

	}


	requestAnimationFrame(draw, canvas);	
}


function draw_Memory(LightMode, shadow){


	// //For Reflection Texture
	gl.bindFramebuffer(gl.FRAMEBUFFER, gFBO_Reflection);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		draw_terrain(SceneConst.MEMORY, SceneConst.REFLECT_TEX, texture_Scene_Memory, LightMode, [0.0, 1.0, 0.0, -gWater_Level + 1.0]);
		
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);


	//For Refraction Texture
	gl.bindFramebuffer(gl.FRAMEBUFFER, gFBO_Refraction);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		draw_terrain(SceneConst.MEMORY, SceneConst.REFRACT_TEX, texture_Scene_Memory, LightMode, [0.0, -1.0, 0.0, gWater_Level]);
		
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);


	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);	
	
	if(shadow == 0)
		draw_terrain(SceneConst.MEMORY, SceneConst.NORMAL, texture_Scene_Memory, LightMode, [0.0, 0.0, 0.0, 0.0]);
	else
		draw_terrain_with_shadow(SceneConst.MEMORY, SceneConst.NORMAL, texture_Scene_Memory, LightMode, 0);

	if(LightMode == Lights.POINT_LIGHT)
		display_water(1);
	else
		display_water(0);

	// display_testRect(gFBO_Reflection_Tex);
	// display_testRect(gFBO_Refraction_Tex);


}



function draw_Reality(choice){
	
	// draw_terrain(SceneConst.REALITY, SceneConst.NORMAL, texture_Scene_Reality);
	draw_terrain_with_shadow(SceneConst.REALITY, SceneConst.NORMAL, texture_Scene_Reality, Lights.POINT_LIGHT, choice);

}




function invertPitchAngle(pitchAngle){

	var x = Math.cos(degToRad(gYawAngle)) * Math.cos(degToRad(pitchAngle));
	var y = Math.sin(degToRad(pitchAngle));
	var z = Math.sin(degToRad(gYawAngle)) * Math.cos(degToRad(pitchAngle));

	vec3.set(gCameraDirection, x, y, z);

	vec3.normalize(gCameraDirection, gCameraDirection);

	vec3.copy(gCameraFront, gCameraDirection);
}


function setYawn(yawAngle){

	var x = Math.cos(degToRad(yawAngle)) * Math.cos(degToRad(gPitchAngle));
	var y = Math.sin(degToRad(gPitchAngle));
	var z = Math.sin(degToRad(yawAngle)) * Math.cos(degToRad(gPitchAngle));

	vec3.set(gCameraDirection, x, y, z);

	vec3.normalize(gCameraDirection, gCameraDirection);

	vec3.copy(gCameraFront, gCameraDirection);
}

function SetCamFront(pitchAngle, yawAngle){

	var x = Math.cos(degToRad(yawAngle)) * Math.cos(degToRad(pitchAngle));
	var y = Math.sin(degToRad(pitchAngle));
	var z = Math.sin(degToRad(yawAngle)) * Math.cos(degToRad(pitchAngle));

	vec3.set(gCameraDirection, x, y, z);

	vec3.normalize(gCameraDirection, gCameraDirection);

	vec3.copy(gCameraFront, gCameraDirection);
}

function degToRad(angle){
	return(angle * (RRJ_PI / 180.0));
}



