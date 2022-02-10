package com.rtr.mandelbrotcube;

//by user
import android.content.Context;
import android.view.Gravity;
import android.graphics.Color;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

//for opengl
import android.opengl.GLSurfaceView;
import android.opengl.GLES32;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

//opengl buffers
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//matrix math
import android.opengl.Matrix;


public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer,OnGestureListener,OnDoubleTapListener{

    private final Context context;
    private GestureDetector gestureDetector;

    private int shaderProgramObject;
    private int vertexShaderObject;
    private int fragmentShaderObject;

    private int[] vao_cube = new int[1];

    private int[] vbo_position_cube = new int[1];
    private int[] vbo_normal_cube = new int[1];

    private int mvpUniform;
    private int mvUniform;
    private int pUniform;
    private int lightPositionUniform;
    private int lKeyPressedUniform;
    private int maxIterationUniform;
    private int zoomUniform;
    private int xCenterUniform;
    private int yCenterUniform;
    private int innerColorUniform;
    private int outerColor1Uniform;
    private int  outerColor2Uniform;

    private float[] perspectiveProjectionMatrix = new float[16];

    private float angleCube = 0.0f;

    private boolean gbAnimation = false;
    private boolean gbLight = false;

    public GLESView(Context drawingContext){
        super(drawingContext);
        context = drawingContext;

        setEGLContextClientVersion(3);
        setRenderer(this);
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

        gestureDetector = new GestureDetector(drawingContext,this,null,false);
        gestureDetector.setOnDoubleTapListener(this);
    }

    //Renderer's method
    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config)
    {
        String version = gl.glGetString(GL10.GL_VERSION);
        String glslVersion = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
        //String vendor = gl.glGetString(GLES32.vendor);
        //String renderer = gl.glGetString(GLES32.renderer);

        System.out.println("RTR: " + version);
        System.out.println("RTR: " + glslVersion);
        //System.out.println("RTR: " + vendor);
        //System.out.println("RTR: " + renderer);

        initialize();
    }

    @Override
    public void onSurfaceChanged(GL10 unused, int width, int height)
    {
        resize(width, height);
    }

    @Override
    public void onDrawFrame(GL10 unused)
    {
        if(gbAnimation == true)
        {
            update();
        }

        display();
    }

    //our callbacks/ custom methods

    private void initialize()
    {

        //vertex shader
        vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

        final String vertexShaderSourceCode = String.format(
            "#version 320 es" +
            "\n" +
            "in vec4 vPosition;" +
            "in vec3 vNormal;" +
            "uniform mat4 u_mv_matrix;" +
            "uniform mat4 u_p_matrix;" +
            "uniform vec4 u_light_position;" +
            "uniform int u_lKeyPressed;" +
            "const float specularContribution = 0.3;" +
            "const float diffuseContribution = 1.0 - specularContribution;" +
            "float diffuse;" +
            "float spec;" +
            "out float lightIntensity;" +
            "out vec3 position;" +
            "void main(void)" +
            "{" +
            "	if(u_lKeyPressed == 1)" +
            "	{" +
            "		vec4 eyeCoords = u_mv_matrix * vPosition;" +
            "		mat3 normal_matrix = mat3(transpose(inverse(u_mv_matrix)));" +
            "		vec3 tNormal = normalize(normal_matrix * vNormal);" +
            "		vec3 s = normalize(vec3(u_light_position - eyeCoords));" +
            "		vec3 reflection_vector = reflect(-s, tNormal);" +
            "		vec3 viewer_vector = normalize(vec3(-eyeCoords.xyz));" +
            "		diffuse = max(dot(s, tNormal), 0.0);" +
            "		spec= max(dot(reflection_vector, viewer_vector), 0.0);" +
            "		spec = pow(spec, 128.0);" +
            "	}" +
            "	lightIntensity = diffuseContribution * diffuse + specularContribution * spec;" +
            "	position = vPosition.xyz;" +
            "	gl_Position = u_p_matrix * u_mv_matrix * vPosition;" +
            "}");

        GLES32.glShaderSource(vertexShaderObject, vertexShaderSourceCode);

        GLES32.glCompileShader(vertexShaderObject);

        //compilation error checking

        int[] iShaderCompileStatus = new int[1];
        int[] iInfoLogLength = new int[1];
        String szInfoLog = null;

        GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS,iShaderCompileStatus,0);
        if(iShaderCompileStatus[0] == GLES32.GL_FALSE)
        {
            GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
            if(iInfoLogLength[0] > 0)
            {
                szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
                System.out.println("RTR: Vertex Shader Compilation log: " + szInfoLog);

                uninitialize();
                System.exit(0);
            }
        }

        //fragment shader
        fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

        final String fragmentShaderSourceCode = String.format(
            "#version 320 es" +
            "\n" +
            "precision highp float;" +
            "precision highp int;" +
            "uniform int u_lKeyPressed;" +
            "uniform float u_max_iteration;" +
            "uniform float u_zoom;" +
            "uniform float u_x_center;" +
            "uniform float u_y_center;" +
            "uniform vec3 u_inner_color;" +
            "uniform vec3 u_outer_color_1;" +
            "uniform vec3 u_outer_color_2;" +
            "in vec3 position;" +
            "in float lightIntensity;" +
            "out vec4 fragColor;" +
            "void main(void)" +
            "{" +
            "	if(u_lKeyPressed == 1)" +
            "	{" +
            "		float real = position.x * u_zoom + u_x_center;" +
            "		float imag = position.y * u_zoom + u_y_center;" +
            "		float cReal = real;" +
            "		float cImag = imag;" +
            "		float r2 = 0.0;" +
            "		float iter;" +
            "		for(iter = 0.0; iter < u_max_iteration && r2 < 4.0; ++iter)" +
            "		{" +
            "			float tempReal = real;" +
            "			real = (tempReal * tempReal) - (imag * imag) + cReal;" +
            "			imag = 2.0 * tempReal * imag + cImag;" +
            "			r2 = (real * real) + (imag * imag);" +
            "		}" +
            "		vec3 clr;" +
            "		if(r2 < 4.0)" +
            "			clr = u_inner_color;" +
            "		else" +
            "			clr = mix(u_outer_color_1, u_outer_color_2, fract(iter * 0.05));" +
            "		clr *= lightIntensity;" +
            "		fragColor = vec4(clr,1.0);" +
            "	}" +
            "	else" +
            "	{" +
            "		fragColor = vec4(1.0, 1.0, 1.0, 1.0);" +
            "	}" +
            "}");

        GLES32.glShaderSource(fragmentShaderObject, fragmentShaderSourceCode);

        GLES32.glCompileShader(fragmentShaderObject);

        //compilation error checking

        iShaderCompileStatus[0] = 0;
        iInfoLogLength[0] = 0;
        szInfoLog = null;

        GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS,iShaderCompileStatus,0);
        if(iShaderCompileStatus[0] == GLES32.GL_FALSE)
        {
            GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
            if(iInfoLogLength[0] > 0)
            {
                szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
                System.out.println("RTR: Fragment Shader Compilation log: " + szInfoLog);

                uninitialize();
                System.exit(0);
            }
        }

        //Shader program
        shaderProgramObject = GLES32.glCreateProgram();

        GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
        GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

        //prelinking binding to attributes
        GLES32.glBindAttribLocation(shaderProgramObject,
                        GLESMacros.AMC_ATTRIBUTE_POSITION,
                        "vPosition");

        GLES32.glBindAttribLocation(shaderProgramObject,
                        GLESMacros.AMC_ATTRIBUTE_NORMAL,
                        "vNormal");

        GLES32.glLinkProgram(shaderProgramObject);
        //compilation error checking

        int[] iProgramLinkStatus = new int[1];
        iInfoLogLength[0] = 0;
        szInfoLog = null;

        GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS,iProgramLinkStatus,0);
        if(iProgramLinkStatus[0] == GLES32.GL_FALSE)
        {
            GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
            if(iInfoLogLength[0] > 0)
            {
                szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
                System.out.println("RTR: Shader Program linking log: " + szInfoLog);

                uninitialize();
                System.exit(0);
            }
        }

        //get uniform location
        mvUniform = GLES32.glGetUniformLocation(shaderProgramObject,
		"u_mv_matrix");

        pUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_p_matrix");

        lKeyPressedUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_lKeyPressed");

        maxIterationUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_max_iteration");

        zoomUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_zoom");

        xCenterUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_x_center");

        yCenterUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_y_center");

        innerColorUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_inner_color");

        outerColor1Uniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_outer_color_1");

        outerColor2Uniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_outer_color_2");

        lightPositionUniform = GLES32.glGetUniformLocation(shaderProgramObject,
            "u_light_position");

        final float cubeVertices[] = new float[]{
            1.0f, 1.0f, -1.0f,
            -1.0f, 1.0f, -1.0f,
            -1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f, 1.0f,
            1.0f, -1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            -1.0f, 1.0f, 1.0f,
            -1.0f, -1.0f, 1.0f,
            1.0f, -1.0f, 1.0f,
            1.0f, 1.0f, -1.0f,
            -1.0f, 1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
            1.0f, -1.0f, -1.0f,
            1.0f, 1.0f, -1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, -1.0f, 1.0f,
            1.0f, -1.0f, -1.0f,
            -1.0f, 1.0f, -1.0f,
            -1.0f, 1.0f, 1.0f,
            -1.0f, -1.0f, 1.0f,
            -1.0f, -1.0f, -1.0f };

        final float cubeNormals[] = new float[]{
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,

            0.0f, -1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,
            0.0f, -1.0f, 0.0f,

            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,

            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,

            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,

            -1.0f, 0.0f, 0.0f,
            -1.0f, 0.0f, 0.0f,
            -1.0f, 0.0f, 0.0f,
            -1.0f, 0.0f, 0.0f };

        //create vao and bind vao
        //cube
        GLES32.glGenVertexArrays(1, vao_cube, 0);

        GLES32.glBindVertexArray(vao_cube[0]);

        //position
        GLES32.glGenBuffers(1, vbo_position_cube, 0);

        GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_position_cube[0]);

        ByteBuffer byteBufferPositionCube = ByteBuffer.allocateDirect(cubeVertices.length * 4);
        byteBufferPositionCube.order(ByteOrder.nativeOrder());

        FloatBuffer positionBufferCube = byteBufferPositionCube.asFloatBuffer();
        positionBufferCube.put(cubeVertices);
        positionBufferCube.position(0);

        GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
                        cubeVertices.length * 4,
                        positionBufferCube,
                        GLES32.GL_STATIC_DRAW);

        GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
                        3,
                        GLES32.GL_FLOAT,
                        false,
                        0,
                        0);

        GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);

        GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


        //color
        GLES32.glGenBuffers(1, vbo_normal_cube, 0);

        GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_normal_cube[0]);

        ByteBuffer byteBufferNormalCube = ByteBuffer.allocateDirect(cubeNormals.length * 4);
        byteBufferNormalCube.order(ByteOrder.nativeOrder());

        FloatBuffer normalBufferCube = byteBufferNormalCube.asFloatBuffer();
        normalBufferCube.put(cubeNormals);
        normalBufferCube.position(0);

        GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
                        cubeNormals.length * 4,
                        normalBufferCube,
                        GLES32.GL_STATIC_DRAW);

        GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_NORMAL,
                        3,
                        GLES32.GL_FLOAT,
                        false,
                        0,
                        0);

        GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_NORMAL);

        GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


        GLES32.glBindVertexArray(0);

        GLES32.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        GLES32.glDisable(GLES32.GL_CULL_FACE);

	    GLES32.glEnable(GLES32.GL_DEPTH_TEST);
	    GLES32.glDepthFunc(GLES32.GL_LEQUAL);

        Matrix.setIdentityM(perspectiveProjectionMatrix,0);

    }


    private void resize(int width, int height)
    {
        if(height < 0)
        {
            height = 1;
        }

        GLES32.glViewport(0,0,width,height);
        Matrix.perspectiveM(perspectiveProjectionMatrix,
                0,
                45.0f,
                width / height,
                0.1f,
                100.0f);
    }

    private void display()
    {
        GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

        GLES32.glUseProgram(shaderProgramObject);

        float[] modelViewMatrix = new float[16];
        float[] projectionMatrix = new float[16];
        float[] rotationMatrix = new float[16];

        //cube
        //identity
        Matrix.setIdentityM(modelViewMatrix,0);
        Matrix.setIdentityM(projectionMatrix,0);
        Matrix.setIdentityM(rotationMatrix, 0);

        //tranformation
        Matrix.translateM(modelViewMatrix, 0, 0.0f, 0.0f, -3.0f);
        Matrix.scaleM(modelViewMatrix, 0, 0.75f, 0.75f, 0.75f);

        Matrix.setRotateM(rotationMatrix, 0, angleCube, 0.0f, 1.0f, 0.0f);
        Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotationMatrix, 0);

        Matrix.multiplyMM(projectionMatrix, 0,
                            perspectiveProjectionMatrix, 0,
                            projectionMatrix, 0);

        GLES32.glUniformMatrix4fv(mvUniform, 1, false, modelViewMatrix, 0);
        GLES32.glUniformMatrix4fv(pUniform, 1, false, projectionMatrix, 0);

        if(gbLight == true)
        {
            GLES32.glUniform1i(lKeyPressedUniform, 1);
            GLES32.glUniform1f(maxIterationUniform, 100.0f);
            GLES32.glUniform1f(zoomUniform, 1.5f);
            GLES32.glUniform1f(xCenterUniform, -0.5f);
            GLES32.glUniform1f(yCenterUniform, 0.0f);
            GLES32.glUniform3f(innerColorUniform, 0.0f, 0.0f, 0.0f);
            GLES32.glUniform3f(outerColor1Uniform, 0.0f, 1.0f, 0.0f);
            GLES32.glUniform3f(outerColor2Uniform, 0.0f, 0.0f, 0.0f);
            GLES32.glUniform4f(lightPositionUniform, 0.0f, 0.0f, 2.0f, 1.0f);
        }
        else
        {
            GLES32.glUniform1i(lKeyPressedUniform, 0);
        }

        GLES32.glBindVertexArray(vao_cube[0]);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 0, 4);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 4, 4);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 8, 4);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 12, 4);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 16, 4);
        GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN, 20, 4);
        GLES32.glBindVertexArray(0);


        GLES32.glUseProgram(0);

        requestRender();
    }

    private void update()
    {
        angleCube = angleCube - 1.0f;
        if (angleCube <= -360.0f)
        {
            angleCube = 0.0f;
        }
    }

    private void uninitialize()
    {
        if (vbo_normal_cube[0] != 0)
        {
            GLES32.glDeleteBuffers(1, vbo_normal_cube, 0);
            vbo_normal_cube[0] = 0;
        }

        if (vbo_position_cube[0] != 0)
        {
            GLES32.glDeleteBuffers(1, vbo_position_cube, 0);
            vbo_position_cube[0] = 0;
        }

        if (vao_cube[0] != 0)
        {
            GLES32.glDeleteVertexArrays(1, vao_cube, 0);
            vao_cube[0] = 0;
        }

        if (shaderProgramObject != 0)
        {
            int[] shaderCount = new int[1];
            int shaderNumber;

            GLES32.glUseProgram(shaderProgramObject);

            //ask the program how many shaders are attached to you?
            GLES32.glGetProgramiv(shaderProgramObject,
                GLES32.GL_ATTACHED_SHADERS,
                shaderCount,
                0);

            int[] shaders = new int[shaderCount[0]];

            if (shaders[0] != 0)
            {
                //get shaders
                GLES32.glGetAttachedShaders(shaderProgramObject,
                    shaderCount[0],
                    shaderCount,
                    0,
                    shaders,
                    0);

                for (shaderNumber = 0; shaderNumber < shaderCount[0]; shaderNumber++)
                {
                    //detach
                    GLES32.glDetachShader(shaderProgramObject,
                        shaders[shaderNumber]);

                    //delete
                    GLES32.glDeleteShader(shaders[shaderNumber]);

                    //explicit 0
                    shaders[shaderNumber] = 0;
                }
            }

            GLES32.glDeleteProgram(shaderProgramObject);
            shaderProgramObject = 0;

            GLES32.glUseProgram(0);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event){

        int eventaction = event.getAction();
        if(!gestureDetector.onTouchEvent(event)){
            super.onTouchEvent(event);
        }

        return(true);
    }

    @Override
    public boolean onDoubleTap(MotionEvent e){
        if (gbAnimation == false)
        {
            gbAnimation = true;
        }
        else
        {
            gbAnimation = false;
        }
        return(true);
    }

    @Override
    public boolean onDoubleTapEvent(MotionEvent e){
        return(true);
    }

    @Override
    public boolean onSingleTapConfirmed(MotionEvent e){
        if (gbLight == false)
        {
            gbLight = true;
        }
        else
        {
            gbLight = false;
        }

        return(true);
    }

    @Override
    public boolean onDown(MotionEvent e){
        return(true);
    }

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY){
        return(true);
    }

    @Override
    public void onLongPress(MotionEvent e){
    }

    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2,float distanceX, float distanceY){
        uninitialize();
        System.exit(0);
        return(true);
    }

    @Override
    public void onShowPress(MotionEvent e){

    }

    @Override
    public boolean onSingleTapUp(MotionEvent e){
        return(true);
    }
}
