//global variables
var canvas=null;
var gl=null;//WebGL context
var bFullscreen=false;
var canvas_original_width;
var canvas_original_height;

//When whole WebGLMacros is const, means all inside it are automatically const
//This concept is called Named Value Pairing / Key Value Coding or Object Initializer
const WebGLMacros=
{
    HAD_ATTRIBUTE_POSITION:0,
    HAD_ATTRIBUTE_COLOR:1,
    HAD_ATTRIBUTE_NORMAL:2,
    HAD_ATTRIBUTE_TEXTURE0:3,
};

//Shader Object
var vertexShaderObject;
var fragmentShaderObject;

//Program Object
var shaderProgramObject;

var checkImageHeight=64;
var checkImageWidth=64;
var checkImage1d = new Uint8Array(64*64*4);
//var checkImage1d = [];
var SquareVertices = new Float32Array(12);
var iterator=0;

var vao_Square;
var vbo_Pos,vbo_Texture;
var mvpUniform;

var perspectiveProjectionMatrix;

var uniform_texture0_sampler;
var Square_texture=0;

//To start Animation : to have requestAnimationFrame to be called "cross-browser" compatible
var requestAnimationFrame=window.requestAnimationFrame||
window.webkitRequestAnimationFrame||
window.mozRequestAnimationFrame||
window.oRequestAnimationFrame||
window.msRequestAnimationFrame;

var cancelAnimationFrame=window.cancelAnimationFrame||
window.webkitCancelAnimationFrame||
window.webkitCancelRequestAnimationFrame||
window.mozCancelRequestAnimationFrame||
window.mozCancelAnimmationFrame||
window.oCancelRequestAnimationFrame||
window.oCancelAnimationFrame||
window.msCancelRequestAnimationFrame||
window.msCancelAnimationFrame;

//onload function
function main()
{
    //get <canvas> element
    canvas = document.getElementById("HAD");
    if(!canvas)
        console.log("Obtaining Canvas Failed\n");
    else
        console.log("Obtaining Canvas Succeeded\n");

    canvas_original_width=canvas.width;
    canvas_original_height=canvas.height;
   
    //register keyboard's keydown event handler
    window.addEventListener("keydown",keyDown,false);
    window.addEventListener("click",mouseDown,false);
    window.addEventListener("resize",resize,false);

    //initialize WebGL
    init();

    //start drawing here as warming - up
    resize();
    draw();
}

function toggleFullScreen()
{
    var fullscreen_element=document.fullscreenElement||document.webkitFullscreenElement||document.mozFullScreenElement||document.msFullscreenElement||null;

    if(fullscreen_element==null)
    {
        if(canvas.requestFullscreen)
            canvas.requestFullscreen();
        else if(canvas.mozRequestFullScreen)
            canvas.mozRequestFullScreen();
        else if(canvas.webkitRequestFullscreen)
            canvas.webkitRequestFullscreen();
        else if(canvas.msRequestFullscreen)
            canvas.msRequestFullscreen();
        bFullscreen=true;
    }

    else
    {
        if(document.exitFullscreen)
            document.exitFullscreen();
        else if(document.mozCancelFullScreen)
            document.mozCancelFullScreen();
        else if(document.webkitExitFullscreen)
            document.webkitExitFullscreen();
        else if(document.msExitFullscreen)
            document.msExitFullscreen();
        bFullscreen=false;
    }
}

function init()
{
    //Get WebGL 2.0 Context
    gl=canvas.getContext("webgl2");
    if(gl==null)
    {
        console.log("Failed to get the rendering context for WebGL");
        return;
    }
    console.log("Rendering context for WebGL Obtained");

    gl.viewportWidth=canvas.width;
    gl.viewportHeight=canvas.height;

    //Vertex Shader
    var vertexShaderSourceCode=
    "#version 300 es                            "+/*version is 300 because WebGL 2.0 is OpenGL ES 3.0 complaint no 3.2 complaint*/
    "\n                                         "+
    "in vec4 vPosition;                         "+
    "in vec2 vTexture0_Coord;                   "+
    "out vec2 out_texture0_coord;               "+
    "uniform mat4 u_mvp_matrix;                 "+
    "out vec4 out_color;                        "+
    "void main(void)                            "+
    "{                                          "+
    "   out_texture0_coord = vTexture0_Coord;   "+
    "   gl_Position = u_mvp_matrix * vPosition; "+
    "}";

    vertexShaderObject=gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShaderObject,vertexShaderSourceCode);
    gl.compileShader(vertexShaderObject);
    //Instead of glGetShaderiv() we need to use gl.getShaderParamter in WebGL
    if(gl.getShaderParameter(vertexShaderObject,gl.COMPILE_STATUS)==false)
    {
        var error=gl.getShaderInfoLog(vertexShaderObject);
        if(error.length>0)
        {
            alert(error);
            uninitialize();
        }
    }

    //Fragment Shader
    var fragmentShaderSourceCode=
    "#version 300 es                                            "+
    "\n                                                         "+
    "precision highp float;                                     "+
    "in vec2 out_texture0_coord;                                "+
    "uniform highp sampler2D u_texture0_sampler;                "+
    "out vec4 FragColor;                                        "+
    "void main(void)                                            "+
    "{                                                          "+
    "FragColor = texture(u_texture0_sampler,out_texture0_coord);"+                
    "}";

    fragmentShaderObject=gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShaderObject,fragmentShaderSourceCode);
    gl.compileShader(fragmentShaderObject);
    if(gl.getShaderParameter(fragmentShaderObject,gl.COMPILE_STATUS)==false)
    {
        var error = gl.getShaderInfoLog(fragmentShaderObject);
        if(error.length>0)
        {
            alert(error);
            uninitialize();
        }
    }

    //Shader Program
    shaderProgramObject=gl.createProgram();
    gl.attachShader(shaderProgramObject,vertexShaderObject);
    gl.attachShader(shaderProgramObject,fragmentShaderObject);

    //Pre-link binding of shader program object with shader's attributes 
    gl.bindAttribLocation(shaderProgramObject,WebGLMacros.HAD_ATTRIBUTE_POSITION,"vPosition");

    gl.bindAttribLocation(shaderProgramObject,WebGLMacros.HAD_ATTRIBUTE_TEXTURE0,"vTexture0_Coord");

    //Linking
    gl.linkProgram(shaderProgramObject);
    if(!gl.getProgramParameter(shaderProgramObject,gl.LINK_STATUS))
    {
        var error = gl.getProgramInfoLog(shaderProgramObject);
        if(error.length>0)
        {
            alert(error);
            uninitialize();
        }
    }

    makeCheckImage();

    //Load Texture
    Square_texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,Square_texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT,true);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,checkImageWidth,checkImageHeight,0,gl.RGBA,gl.UNSIGNED_BYTE,checkImage1d);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D,null);
    

    //Get Uniform Locations
    mvpUniform = gl.getUniformLocation(shaderProgramObject,"u_mvp_matrix");

    uniform_texture0_sampler=gl.getUniformLocation(shaderProgramObject,"u_texture0_sampler");

    /*Texcoord Array*/
    var SquareTexcoords = new Float32Array([
                                    0.0,0.0,
                                    1.0,0.0,
                                    1.0,1.0,
                                    0.0,1.0,]);
 
    //In WebGL we use createVertexArray() instead of glGenVertexArrays(), and also we can create on 1 vao at a time
    //Vertex Array Object Square
    vao_Square=gl.createVertexArray();
    gl.bindVertexArray(vao_Square);

    //Vertex Buffer Object
    vbo_Pos=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Pos);
    gl.bufferData(gl.ARRAY_BUFFER,SquareVertices,gl.DYNAMIC_DRAW);
    
    gl.vertexAttribPointer(WebGLMacros.HAD_ATTRIBUTE_POSITION,3,gl.FLOAT,false,0,0);

    gl.enableVertexAttribArray(WebGLMacros.HAD_ATTRIBUTE_POSITION);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    //Texture Buffer Object
    vbo_Texture=gl.createBuffer();

    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Texture);
    gl.bufferData(gl.ARRAY_BUFFER,SquareTexcoords,gl.STATIC_DRAW);
    
    gl.vertexAttribPointer(WebGLMacros.HAD_ATTRIBUTE_TEXTURE0,2,gl.FLOAT,false,0,0);

    gl.enableVertexAttribArray(WebGLMacros.HAD_ATTRIBUTE_TEXTURE0);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    gl.bindVertexArray(null);

    gl.clearColor(0.0,0.0,0.0,1.0);

    gl.enable(gl.DEPTH_TEST);
    gl.clearDepth(1.0);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);

    perspectiveProjectionMatrix=mat4.create();
}

function resize()
{
    if(bFullscreen==true)
    {
        canvas.width=window.innerWidth;
        canvas.height=window.innerHeight;
    }
    else
    {
        canvas.width=canvas_original_width;
        canvas.height=canvas_original_height;
    }

    gl.viewport(0,0,canvas.width,canvas.height);

    //Orthographic Projection
    /*if(canvas.width <= canvas.height)
        mat4.ortho(orthographicProjectionMatrix,-100.0,100.0,(-100.0 * (canvas.height/canvas.width)),(100.0 * (canvas.height/canvas.width)),-100.0,100.0);
    else
        mat4.ortho(orthographicProjectionMatrix,(-100.0 * (canvas.width/canvas.height)),(100.0 * (canvas.width/canvas.height)),-100.0,100.0,-100.0,100.0);*/

    mat4.perspective(perspectiveProjectionMatrix,degToRad(60),canvas.width/canvas.height,1.0,30.0);
}

function draw()
{
    // code
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(shaderProgramObject);

    var modelViewMatrix=mat4.create();
    var modelViewProjectionMatrix=mat4.create();

    mat4.translate(modelViewMatrix,modelViewMatrix,[0.0,0.0,-3.0]);

    mat4.multiply(modelViewProjectionMatrix,perspectiveProjectionMatrix,modelViewMatrix);

    gl.uniformMatrix4fv(mvpUniform,false,modelViewProjectionMatrix);

    gl.bindTexture(gl.TEXTURE_2D,Square_texture);
    gl.uniform1i(uniform_texture0_sampler, 0);

    gl.bindVertexArray(vao_Square);

    SquareVertices[0] = 0.0;
	SquareVertices[1] = 1.0;
	SquareVertices[2] = 0.0;
	SquareVertices[3] = -2.0;
	SquareVertices[4] = 1.0;
	SquareVertices[5] = 0.0;
	SquareVertices[6] = -2.0;
	SquareVertices[7] = -1.0;
	SquareVertices[8] = 0.0;
	SquareVertices[9] = 0.0;
	SquareVertices[10] = -1.0;
    SquareVertices[11] = 0.0;
    
    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Pos);
    gl.bufferData(gl.ARRAY_BUFFER,SquareVertices,gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    gl.drawArrays(gl.TRIANGLE_FAN,0,4);

    SquareVertices[0] = 2.41421;
	SquareVertices[1] = 1.0;
	SquareVertices[2] = -1.41421;
	SquareVertices[3] = 1.0;
	SquareVertices[4] = 1.0;
	SquareVertices[5] = 0.0;
	SquareVertices[6] = 1.0;
	SquareVertices[7] = -1.0;
	SquareVertices[8] = 0.0;
	SquareVertices[9] = 2.41421;
	SquareVertices[10] = -1.0;
    SquareVertices[11] = -1.41421;
    
    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Pos);
    gl.bufferData(gl.ARRAY_BUFFER,SquareVertices,gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    gl.drawArrays(gl.TRIANGLE_FAN,0,4);

    gl.bindVertexArray(null);

    gl.useProgram(null);
    
    // animation loop
    requestAnimationFrame(draw, canvas);
}

function makeCheckImage()
{
    var i,j,k,c;

    for(i=0;i<checkImageHeight;i++)
    {
        for(j=0;j<checkImageWidth;j++)
        {
            c=((i&0x8)^(j&0x8))*0xff;
            checkImage1d[iterator]=c;
            checkImage1d[iterator+1]=c;
            checkImage1d[iterator+2]=c;
            checkImage1d[iterator+3]=0xff;
            iterator+=4;
        }
    }
}

function degToRad(degrees)
{
    return(degrees * Math.PI/180.0);
}

function keyDown(event)
{
    switch(event.keyCode)
    {
        case 70: //F or f
            toggleFullScreen();
            break;

        case 27: //ESC
            uninitialize();
            window.close();
            break;
    }
}

function mouseDown()
{
    
}

function uninitialize()
{
    if(vao_Square)
    {
        gl.deleteVertexArray(vao_Square);
        vao_Square=null;
    }

    if(vbo_Pos)
    {
        gl.deleteBuffer(vbo_Pos);
        vbo_Pos=null;
    }

    if(vbo_Texture)
    {
        gl.deleteBuffer(vbo_Texture);
        vbo_Texture=null;
    }

    if(shaderProgramObject)
    {
        if(fragmentShaderObject)
        {
            gl.detachShader(shaderProgramObject,fragmentShaderObject);
            gl.deleteShader(fragmentShaderObject);
            fragmentShaderObject=null;
        }

        if(vertexShaderObject)
        {
            gl.detachShader(shaderProgramObject,vertexShaderObject);
            gl.deleteShader(vertexShaderObject);
            vertexShaderObject=null;
        }

        gl.deleteProgram(shaderProgramObject);
        shaderProgramObject=null;
    }
}