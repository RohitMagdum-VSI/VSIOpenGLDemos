// onload function
var canvas = null;
var gl = null;
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

const WebGLMacros = {
    AMC_ATTRIBUTE_POSITION: 0,
    AMC_ATTRIBUTE_COLOR: 1,
    AMC_ATTRIBUTE_NORMAL: 2,
    AMC_ATTRIBUTE_TEXTURE0: 3
};

var gShaderProgramObject;
var gVertexShaderObject;
var gFragmentShaderObject;

var vao;
var vbo_position;

var mvpUniform;

var perspectiveProjectionMatrix;

var requestAnimationFrame =
    window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    window.mozRequestAnimationFrame ||
    window.oRequestAnimationFrame ||
    window.msRequestAnimationFrame;

var cancelAnimationFrame =
    window.cancelAnimationFrame ||
    window.webkitCancelRequestAnimationFrame ||
    window.webkitCancelAnimationFrame ||
    window.mozCancelRequestAnimationFrame ||
    window.mozCancelAnimationFrame ||
    window.oCancelRequestAnimationFrame ||
    window.oCancelAnimationFrame ||
    window.msCancelRequestAnimationFrame ||
    window.msCancelAnimationFrame;

// prettier-ignore
var data_ecg = new Float32Array([-0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353, -0.087912, -0.10884, -0.10047, 0.071167, 0.0833, -0.07954, -0.63632, -0.28885, -0.041863, 0.0083726, 0.083726, 0.10047, 0.16745, 0.23443, 0.3056, 0.35165, 0.427, 0.42282, 0.32234, 0.17582, 0.012559, -0.050235, -0.066981, -0.092098, -0.092098, -0.092098, -0.087912, -0.07954, -0.066981, -0.075353, -0.083726, -0.096285, -0.087912, -0.087912, -0.092098, -0.10466, -0.087912, -0.087912, -0.092098, -0.087912, -0.087912, -0.07954, -0.066981, -0.075353]);

var points = null;
var counter = 0;

function main() {
    //S1: Get canvas
    canvas = document.getElementById("AMC");
    if (!canvas) console.log("Obtaining Canvas failed!\n");
    else console.log("Obtaining Canvas successful!\n");

    canvas_original_width = canvas.width;
    canvas_original_height = canvas.height;

    console.log("Canvas width : " + canvas.width + "Canvas height :" + canvas.height);

    //register keybord and mouse event handler
    window.addEventListener("keydown", keyDown, false);
    window.addEventListener("click", mouseDown, false);
    window.addEventListener("resize", resize, false);

    //initialize WebGL
    init();

    //start drawing and warming up
    resize();
    draw();
}

function toggleFullScreen() {
    var fullscreen_element =
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement ||
        null;

    if (fullscreen_element == null) {
        if (canvas.requestFullscreen) canvas.requestFullscreen();
        else if (canvas.mozRequestFullScreen) canvas.mozRequestFullScreen();
        else if (canvas.webkitFullscreen) canvas.webkitFullscreen();
        else if (canvas.msRequestFullscreen) canvas.msRequestFullscreen();
        bFullscreen = true;
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.mozCancelFullScreen) document.mozCancelFullScreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
        else if (document.msExitFullscreen) document.msExitFullscreen();
        bFullscreen = false;
    }
}

function init() {
    //S2: Get webgl 2.0 context
    gl = canvas.getContext("webgl2");
    if (!gl) {
        console.log("Failed to get rendering context for WebGL\n");
        return;
    } else console.log("Got rendering context for WebGL!\n");

    //set viewport width and height
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;

    //vertex shader
    var vertexShaderSourceCode =
        "#version 300 es" +
        "\n" +
        "in vec4 vPosition;" +
        "in vec4 vColor;" +
        "uniform mat4 u_mvp_matrix;" +
        "out vec4 out_color;" +
        "void main(void)" +
        "{" +
        "   gl_Position = u_mvp_matrix * vPosition;" +
        "   out_color = vColor;" +
        "}";

    gVertexShaderObject = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(gVertexShaderObject, vertexShaderSourceCode);
    gl.compileShader(gVertexShaderObject);
    if (gl.getShaderParameter(gVertexShaderObject, gl.COMPILE_STATUS) == false) {
        var error = gl.getShaderInfoLog(gVertexShaderObject);
        if (error.length > 0) {
            alert("\nVertex Shader compilation log : " + error);
            uninitialize();
        }
    }

    //fragment shader
    var fragmentShaderSourceCode =
        "#version 300 es" +
        "\n" +
        "precision highp float;" +
        "in vec4 out_color;" +
        "out vec4 FragColor;" +
        "void main(void)" +
        "{" +
        "   FragColor = out_color;" +
        "}";

    gFragmentShaderObject = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(gFragmentShaderObject, fragmentShaderSourceCode);
    gl.compileShader(gFragmentShaderObject);
    if (gl.getShaderParameter(gFragmentShaderObject, gl.COMPILE_STATUS) == false) {
        var error = gl.getShaderInfoLog(gFragmentShaderObject);
        if (error.length > 0) {
            alert("\nFragment Shader compilation log : " + error);
            uninitialize();
        }
    }

    //shader program
    gShaderProgramObject = gl.createProgram();
    gl.attachShader(gShaderProgramObject, gVertexShaderObject);
    gl.attachShader(gShaderProgramObject, gFragmentShaderObject);

    //pre-linking
    gl.bindAttribLocation(gShaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_POSITION, "vPosition");

    gl.bindAttribLocation(gShaderProgramObject, WebGLMacros.AMC_ATTRIBUTE_COLOR, "vColor");

    //linking
    gl.linkProgram(gShaderProgramObject);
    if (gl.getProgramParameter(gShaderProgramObject, gl.LINK_STATUS) == false) {
        var error = gl.getProgramInfoLog(gShaderProgramObject);
        if (error.length > 0) {
            alert("\nProgram linking log : " + error);
            uninitialize();
        }
    }

    //get uniform location
    mvpUniform = gl.getUniformLocation(gShaderProgramObject, "u_mvp_matrix");

    //vertices, color, texture, vao, vbo , shader attribs
    ecgData();
    //create vao
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    //create vbo
    vbo_position = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo_position);
    gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
    gl.vertexAttribPointer(WebGLMacros.AMC_ATTRIBUTE_POSITION, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(WebGLMacros.AMC_ATTRIBUTE_POSITION);

    gl.vertexAttrib3f(WebGLMacros.AMC_ATTRIBUTE_COLOR, 0.0, 1.0, 0.0);

    //unbind vbo and vao
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);

    //set clear color
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    gl.enable(gl.DEPTH_TEST);

    perspectiveProjectionMatrix = mat4.create();
}

function resize() {
    if (bFullscreen == true) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    } else {
        canvas.width = canvas_original_width;
        canvas.height = canvas_original_height;
    }

    gl.viewport(0, 0, canvas.width, canvas.height);

    //perspective
    mat4.perspective(
        perspectiveProjectionMatrix,
        45.0,
        parseFloat(canvas.width) / parseFloat(canvas.height),
        0.1,
        100.0
    );
}

function draw() {
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(gShaderProgramObject);

    var modelViewMatrix = mat4.create();
    var modelViewProjectionMatrix = mat4.create();

    mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);
    mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

    gl.uniformMatrix4fv(mvpUniform, false, modelViewProjectionMatrix);

    gl.bindVertexArray(vao);
    gl.lineWidth(2.0);

    gl.drawArrays(gl.LINE_STRIP, 0, counter);

    gl.bindVertexArray(null);

    gl.useProgram(null);

    counter = counter + 1;
    if (counter > 500) {
        counter = 0;
    }

    //animation loop
    requestAnimationFrame(draw, canvas);
}

function uninitialize() {
    if (vao) {
        gl.deleteVertexArray(vao);
        vao = null;
    }

    if (vbo_position) {
        gl.deleteBuffer(vbo_position);
        vbo_position = null;
    }

    if (gShaderProgramObject) {
        if (gFragmentShaderObject) {
            gl.detachShader(gShaderProgramObject, gFragmentShaderObject);
            gl.deleteShader(gFragmentShaderObject);
            gFragmentShaderObject = null;
        }
        if (gVertexShaderObject) {
            gl.detachShader(gShaderProgramObject, gVertexShaderObject);
            gl.deleteShader(gVertexShaderObject);
            gVertexShaderObject = null;
        }

        gl.deleteProgram(gShaderProgramObject);
        gShaderProgramObject = null;
    }
}

function keyDown(event) {
    switch (event.keyCode) {
        case 27:
            uninitialize();
            window.close();
            break;

        case 70:
            toggleFullScreen();
            break;
    }
}

function mouseDown() {}

function ecgData() {
    let data_size = 500;
    let size = data_size;
    //let offset = data_size + data_size;
    let scale = 1.0;
    let offset_y = 0.0;
    let space = (4.4 / size) * (parseFloat(canvas.width) / parseFloat(canvas.height));
    let pos = (-size * space) / 2.0;
    let dataPoints = [];

    for (let i = 0; i < 500; i++) {
        let data = scale * data_ecg[i] + offset_y;
        dataPoints.push(pos, data);
        pos = pos + space;
    }

    points = new Float32Array(dataPoints);
}
