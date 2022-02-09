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

var vao_Triangle,vao_Square;
var vbo_Pos,vbo_Color;
var mvpUniform;

var perspectiveProjectionMatrix;

//To start Animation : to have requestAnimationFrame to be called "cross-browser" compatible
var requestAnimationFrame=window.requestAnimationFrame||
window.webkitRequestAnimationFrame||
window.mozRequestAnimationFrame||
window.oRequestAnimationFrame||
window.msRequestAnimationFrame;

var cancelAnimationFrame=window.cancelAnimationFrame||window.webkitCancelAnimationFrame||window.webkitCancelRequestAnimationFrame
                        ||window.mozCancelRequestAnimationFrame||window.mozCancelAnimmationFrame||window.oCancelRequestAnimationFrame||window.oCancelAnimationFrame  
                        ||window.msCancelRequestAnimationFrame||window.msCancelAnimationFrame;

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
    "in vec4 vColor;                            "+
    "uniform mat4 u_mvp_matrix;                 "+
    "out vec4 out_color;                        "+
    "void main(void)                            "+
    "{                                          "+
    "   out_color=vColor;                       "+
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
    "#version 300 es                    "+
    "\n                                 "+
    "precision highp float;             "+
    "in vec4 out_color;                 "+
    "out vec4 FragColor;                "+
    "void main(void)                    "+
    "{                                  "+
    "FragColor = out_color;             "+
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

    gl.bindAttribLocation(shaderProgramObject,WebGLMacros.HAD_ATTRIBUTE_COLOR,"vColor");

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

    //Get Uniform Locations
    mvpUniform = gl.getUniformLocation(shaderProgramObject,"u_mvp_matrix");

    /*Vertices Array*/
    var triangleVertices = new Float32Array([0.0,1.0,0.0,
                                            -1.0,-1.0,0.0,
                                            1.0,-1.0,0.0]);

    var triangleColors = new Float32Array([1.0,0.0,0.0,
                                           0.0,1.0,0.0,
                                           0.0,0.0,1.0]);

    var squareVertices = new Float32Array([1.0,1.0,0.0,
                                          -1.0,1.0,0.0,
                                          -1.0,-1.0,0.0,
                                           1.0,-1.0,0.0]);

    //Vertex Array Object Triangle
    //In WebGL we use createVertexArray() instead of glGenVertexArrays(), and also we can create on 1 vao at a time
    vao_Triangle=gl.createVertexArray();
    gl.bindVertexArray(vao_Triangle);

    //Vertex Buffer Object
    vbo_Pos=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Pos);
    gl.bufferData(gl.ARRAY_BUFFER,triangleVertices,gl.STATIC_DRAW);
    
    gl.vertexAttribPointer(WebGLMacros.HAD_ATTRIBUTE_POSITION,3,gl.FLOAT,false,0,0);

    gl.enableVertexAttribArray(WebGLMacros.HAD_ATTRIBUTE_POSITION);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    //Color Buffer Object
    vbo_Color=gl.createBuffer();

    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Color);
    gl.bufferData(gl.ARRAY_BUFFER,triangleColors,gl.STATIC_DRAW);
    
    gl.vertexAttribPointer(WebGLMacros.HAD_ATTRIBUTE_COLOR,3,gl.FLOAT,false,0,0);

    gl.enableVertexAttribArray(WebGLMacros.HAD_ATTRIBUTE_COLOR);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    gl.bindVertexArray(null);

    //Vertex Array Object Square
    vao_Square=gl.createVertexArray();
    gl.bindVertexArray(vao_Square);

    //Vertex Buffer Object
    vbo_Pos=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,vbo_Pos);
    gl.bufferData(gl.ARRAY_BUFFER,squareVertices,gl.STATIC_DRAW);
    
    gl.vertexAttribPointer(WebGLMacros.HAD_ATTRIBUTE_POSITION,3,gl.FLOAT,false,0,0);

    gl.enableVertexAttribArray(WebGLMacros.HAD_ATTRIBUTE_POSITION);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);

    gl.vertexAttrib3f(WebGLMacros.HAD_ATTRIBUTE_COLOR,0.392, 0.584, 0.929);

    gl.bindVertexArray(null);

    gl.clearColor(0.0,0.0,0.0,1.0);

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

    mat4.perspective(perspectiveProjectionMatrix,45.0,canvas.width/canvas.height,0.1,100.0);
}

function draw()
{
    // code
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(shaderProgramObject);

    var modelViewMatrix=mat4.create();
    var modelViewProjectionMatrix=mat4.create();
    var translationMatrix=mat4.create();

    //For Triangle
    mat4.translate(modelViewMatrix,modelViewMatrix,[-1.5,0.0,-6.0]);

    //mat4.multiply(modelViewMatrix,translationMatrix,modelViewMatrix);

    mat4.multiply(modelViewProjectionMatrix,perspectiveProjectionMatrix,modelViewMatrix);

    gl.uniformMatrix4fv(mvpUniform,false,modelViewProjectionMatrix);

    gl.bindVertexArray(vao_Triangle);

    gl.drawArrays(gl.TRIANGLES,0,3);

    gl.bindVertexArray(null);

    mat4.identity(modelViewMatrix);
    mat4.identity(modelViewProjectionMatrix);
    mat4.identity(translationMatrix);

    mat4.translate(modelViewMatrix,modelViewMatrix,[1.5,0.0,-6.0]);

    //mat4.multiply(modelViewMatrix,translationMatrix,modelViewMatrix);

    mat4.multiply(modelViewProjectionMatrix,perspectiveProjectionMatrix,modelViewMatrix);

    gl.uniformMatrix4fv(mvpUniform,false,modelViewProjectionMatrix);

    gl.bindVertexArray(vao_Square);

    gl.drawArrays(gl.TRIANGLE_FAN,0,4);
    
    gl.bindVertexArray(null);

    gl.useProgram(null);
    
    // animation loop
    requestAnimationFrame(draw, canvas);
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
    if(vao_Triangle)
    {
        gl.deleteVertexArray(vao_Triangle);
        vao_Triangle=null;
    }

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

    if(vbo_Color)
    {
        gl.deleteBuffer(vbo_Color);
        vbo_Color=null;
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