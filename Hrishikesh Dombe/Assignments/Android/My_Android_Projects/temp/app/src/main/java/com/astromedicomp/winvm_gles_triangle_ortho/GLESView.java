package com.astromedicomp.winvm_gles_triangle_ortho;

import android.content.Context; // for drawing context related
import android.opengl.GLSurfaceView; // for OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; // for OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; // for EGLConfig needed as param type EGLConfig
import android.opengl.GLES31; // for OpenGLES 3.2
import android.view.MotionEvent; // for "MotionEvent"
import android.view.GestureDetector; // for GestureDetector
import android.view.GestureDetector.OnGestureListener; // OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; // for OnDoubleTapListener

// for vbo
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.opengl.Matrix; // for Matrix math

// A view for OpenGLES3 graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
    private final Context context;
    
    private GestureDetector gestureDetector;
    
    private int vertexShaderObject;
    private int fragmentShaderObject;
    private int shaderProgramObject;
    
    private int[] vao = new int[1];
    private int[] vbo = new int[1];
    private int mvpUniform;

    private float orthographicProjectionMatrix[]=new float[16]; // 4x4 matrix
    
    public GLESView(Context drawingContext)
    {
        super(drawingContext);
        
        context=drawingContext;

        // accordingly set EGLContext to current supported version of OpenGL-ES
        setEGLContextClientVersion(3);

        // set Renderer for drawing on the GLSurfaceView
        setRenderer(this);

        // Render the view only when there is a change in the drawing data
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        
        gestureDetector = new GestureDetector(context, this, null, false);
        gestureDetector.setOnDoubleTapListener(this);
    }
    
    // overriden method of GLSurfaceView.Renderer ( Init Code )
    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config)
    {
        // get OpenGL-ES version
        String glesVersion = gl.glGetString(GL10.GL_VERSION);
        System.out.println("VDG: OpenGL-ES Version = "+glesVersion);
        // get GLSL version
        String glslVersion=gl.glGetString(GLES31.GL_SHADING_LANGUAGE_VERSION);
        System.out.println("VDG: GLSL Version = "+glslVersion);

        initialize(gl);
    }
 
    // overriden method of GLSurfaceView.Renderer ( Chnge Size Code )
    @Override
    public void onSurfaceChanged(GL10 unused, int width, int height)
    {
        resize(width, height);
    }

    // overriden method of GLSurfaceView.Renderer ( Rendering Code )
    @Override
    public void onDrawFrame(GL10 unused)
    {
        display();
    }
    
    // Handling 'onTouchEvent' Is The Most IMPORTANT,
    // Because It Triggers All Gesture And Tap Events
    @Override
    public boolean onTouchEvent(MotionEvent e)
    {
        // code
        int eventaction = e.getAction();
        if(!gestureDetector.onTouchEvent(e))
            super.onTouchEvent(e);
        return(true);
    }
    
    // abstract method from OnDoubleTapListener so must be implemented
    @Override
    public boolean onDoubleTap(MotionEvent e)
    {
        return(true);
    }
    
    // abstract method from OnDoubleTapListener so must be implemented
    @Override
    public boolean onDoubleTapEvent(MotionEvent e)
    {
        // Do Not Write Any code Here Because Already Written 'onDoubleTap'
        return(true);
    }
    
    // abstract method from OnDoubleTapListener so must be implemented
    @Override
    public boolean onSingleTapConfirmed(MotionEvent e)
    {
        return(true);
    }
    
    // abstract method from OnGestureListener so must be implemented
    @Override
    public boolean onDown(MotionEvent e)
    {
        // Do Not Write Any code Here Because Already Written 'onSingleTapConfirmed'
        return(true);
    }
    
    // abstract method from OnGestureListener so must be implemented
    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
    {
        return(true);
    }
    
    // abstract method from OnGestureListener so must be implemented
    @Override
    public void onLongPress(MotionEvent e)
    {
    }
    
    // abstract method from OnGestureListener so must be implemented
    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY)
    {
        uninitialize();
        System.exit(0);
        return(true);
    }
    
    // abstract method from OnGestureListener so must be implemented
    @Override
    public void onShowPress(MotionEvent e)
    {
    }
    
    // abstract method from OnGestureListener so must be implemented
    @Override
    public boolean onSingleTapUp(MotionEvent e)
    {
        return(true);
    }

    private void initialize(GL10 gl)
    {
        // ***********************************************
        // Vertex Shader
        // ***********************************************
        // create shader
        vertexShaderObject=GLES31.glCreateShader(GLES31.GL_VERTEX_SHADER);
        
        // vertex shader source code
        final String vertexShaderSourceCode =String.format
        (
         "#version 310 es"+
         "\n"+
         "in vec4 vPosition;"+
         "uniform mat4 u_mvp_matrix;"+
         "void main(void)"+
         "{"+
         "gl_Position = u_mvp_matrix * vPosition;"+
         "}"
        );
        
        // provide source code to shader
        GLES31.glShaderSource(vertexShaderObject,vertexShaderSourceCode);
        
        // compile shader & check for errors
        GLES31.glCompileShader(vertexShaderObject);
        int[] iShaderCompiledStatus = new int[1];
        int[] iInfoLogLength = new int[1];
        String szInfoLog=null;
        GLES31.glGetShaderiv(vertexShaderObject, GLES31.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
        if (iShaderCompiledStatus[0] == GLES31.GL_FALSE)
        {
            GLES31.glGetShaderiv(vertexShaderObject, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
            if (iInfoLogLength[0] > 0)
            {
                szInfoLog = GLES31.glGetShaderInfoLog(vertexShaderObject);
                System.out.println("VDG: Vertex Shader Compilation Log = "+szInfoLog);
                uninitialize();
                System.exit(0);
           }
        }

        // ***********************************************
        // Fragment Shader
        // ***********************************************
        // create shader
        fragmentShaderObject=GLES31.glCreateShader(GLES31.GL_FRAGMENT_SHADER);
        
        // fragment shader source code
        final String fragmentShaderSourceCode =String.format
        (
         "#version 310 es"+
         "\n"+
         "precision highp float;"+
         "out vec4 FragColor;"+
         "void main(void)"+
         "{"+
         "FragColor = vec4(1.0, 1.0, 1.0, 1.0);"+
         "}"
        );
        
        // provide source code to shader
        GLES31.glShaderSource(fragmentShaderObject,fragmentShaderSourceCode);
        
        // compile shader and check for errors
        GLES31.glCompileShader(fragmentShaderObject);
        iShaderCompiledStatus[0] = 0; // re-initialize
        iInfoLogLength[0] = 0; // re-initialize
        szInfoLog=null; // re-initialize
        GLES31.glGetShaderiv(fragmentShaderObject, GLES31.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
        if (iShaderCompiledStatus[0] == GLES31.GL_FALSE)
        {
            GLES31.glGetShaderiv(fragmentShaderObject, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
            if (iInfoLogLength[0] > 0)
            {
                szInfoLog = GLES31.glGetShaderInfoLog(fragmentShaderObject);
                System.out.println("VDG: Fragment Shader Compilation Log = "+szInfoLog);
                uninitialize();
                System.exit(0);
            }
        }
        
        // create shader program
        shaderProgramObject=GLES31.glCreateProgram();
        
        // attach vertex shader to shader program
        GLES31.glAttachShader(shaderProgramObject,vertexShaderObject);
        
        // attach fragment shader to shader program
        GLES31.glAttachShader(shaderProgramObject,fragmentShaderObject);
        
        // pre-link binding of shader program object with vertex shader attributes
        GLES31.glBindAttribLocation(shaderProgramObject,GLESMacros.VDG_ATTRIBUTE_VERTEX,"vPosition");
        
        // link the two shaders together to shader program object
        GLES31.glLinkProgram(shaderProgramObject);
        int[] iShaderProgramLinkStatus = new int[1];
        iInfoLogLength[0] = 0; // re-initialize
        szInfoLog=null; // re-initialize
        GLES31.glGetProgramiv(shaderProgramObject, GLES31.GL_LINK_STATUS, iShaderProgramLinkStatus, 0);
        if (iShaderProgramLinkStatus[0] == GLES31.GL_FALSE)
        {
            GLES31.glGetProgramiv(shaderProgramObject, GLES31.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
            if (iInfoLogLength[0] > 0)
            {
                szInfoLog = GLES31.glGetProgramInfoLog(shaderProgramObject);
                System.out.println("VDG: Shader Program Link Log = "+szInfoLog);
                uninitialize();
                System.exit(0);
            }
        }

        // get MVP uniform location
        mvpUniform=GLES31.glGetUniformLocation(shaderProgramObject,"u_mvp_matrix");
        
        // *** vertices, colors, shader attribs, vbo, vao initializations ***
        final float triangleVertices[]= new float[]
        {
            0.0f, 50.0f,0.0f, // appex
            -50.0f,-50.0f,0.0f, // left-bottom
            50.0f,-50.0f,0.0f // right-bottom
        };

        GLES31.glGenVertexArrays(1,vao,0);
        GLES31.glBindVertexArray(vao[0]);

        GLES31.glGenBuffers(1,vbo,0);
        GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER,vbo[0]);
        
        ByteBuffer byteBuffer=ByteBuffer.allocateDirect(triangleVertices.length * 4);
        byteBuffer.order(ByteOrder.nativeOrder());
        FloatBuffer verticesBuffer=byteBuffer.asFloatBuffer();
        verticesBuffer.put(triangleVertices);
        verticesBuffer.position(0);
        
        GLES31.glBufferData(GLES31.GL_ARRAY_BUFFER,
                            triangleVertices.length * 4,
                            verticesBuffer,
                            GLES31.GL_STATIC_DRAW);
        
        GLES31.glVertexAttribPointer(GLESMacros.VDG_ATTRIBUTE_VERTEX,
                                     3,
                                     GLES31.GL_FLOAT,
                                     false,0,0);
        
        GLES31.glEnableVertexAttribArray(GLESMacros.VDG_ATTRIBUTE_VERTEX);
        
        GLES31.glBindBuffer(GLES31.GL_ARRAY_BUFFER,0);
        GLES31.glBindVertexArray(0);

        // enable depth testing
        GLES31.glEnable(GLES31.GL_DEPTH_TEST);
        // depth test to do
        GLES31.glDepthFunc(GLES31.GL_LEQUAL);
        // We will always cull back faces for better performance
        GLES31.glEnable(GLES31.GL_CULL_FACE);
        
        // Set the background color
        GLES31.glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        
        // set projectionMatrix to identitu matrix
        Matrix.setIdentityM(orthographicProjectionMatrix,0);
    }
    
    private void resize(int width, int height)
    {
        // code
        GLES31.glViewport(0, 0, width, height);
        
        // Orthographic Projection => left,right,bottom,top,near,far
        if (width <= height)
            Matrix.orthoM(orthographicProjectionMatrix,0,-100.0f, 100.0f, (-100.0f * (height / width)), (100.0f * (height / width)), -100.0f, 100.0f);
        else
            Matrix.orthoM(orthographicProjectionMatrix,0,(-100.0f * (width / height)), (100.0f * (width / height)), -100.0f, 100.0f, -100.0f, 100.0f);
    }
    
    public void display()
    {
        // code
        GLES31.glClear(GLES31.GL_COLOR_BUFFER_BIT | GLES31.GL_DEPTH_BUFFER_BIT);
        
        // use shader program
        GLES31.glUseProgram(shaderProgramObject);

        // OpenGL-ES drawing
        float modelViewMatrix[]=new float[16];
        float modelViewProjectionMatrix[]=new float[16];
        
        // set modelview & modelviewprojection matrices to identity
        Matrix.setIdentityM(modelViewMatrix,0);
        Matrix.setIdentityM(modelViewProjectionMatrix,0);

        // multiply the modelview and projection matrix to get modelviewprojection matrix
        Matrix.multiplyMM(modelViewProjectionMatrix,0,orthographicProjectionMatrix,0,modelViewMatrix,0);
        
        // pass above modelviewprojection matrix to the vertex shader in 'u_mvp_matrix' shader variable
        // whose position value we already calculated in initWithFrame() by using glGetUniformLocation()
        GLES31.glUniformMatrix4fv(mvpUniform,1,false,modelViewProjectionMatrix,0);
        
        // bind vao
        GLES31.glBindVertexArray(vao[0]);
        
        // draw, either by glDrawTriangles() or glDrawArrays() or glDrawElements()
        GLES31.glDrawArrays(GLES31.GL_TRIANGLES,0,3); // 3 (each with its x,y,z ) vertices in triangleVertices array
        
        // unbind vao
        GLES31.glBindVertexArray(0);
        
        // un-use shader program
        GLES31.glUseProgram(0);

        // render/flush
        requestRender();
    }
    
    void uninitialize()
    {
        // code
        // destroy vao
        if(vao[0] != 0)
        {
            GLES31.glDeleteVertexArrays(1, vao, 0);
            vao[0]=0;
        }
        
        // destroy vao
        if(vbo[0] != 0)
        {
            GLES31.glDeleteBuffers(1, vbo, 0);
            vbo[0]=0;
        }

        if(shaderProgramObject != 0)
        {
            if(vertexShaderObject != 0)
            {
                // detach vertex shader from shader program object
                GLES31.glDetachShader(shaderProgramObject, vertexShaderObject);
                // delete vertex shader object
                GLES31.glDeleteShader(vertexShaderObject);
                vertexShaderObject = 0;
            }
            
            if(fragmentShaderObject != 0)
            {
                // detach fragment  shader from shader program object
                GLES31.glDetachShader(shaderProgramObject, fragmentShaderObject);
                // delete fragment shader object
                GLES31.glDeleteShader(fragmentShaderObject);
                fragmentShaderObject = 0;
            }
        }

        // delete shader program object
        if(shaderProgramObject != 0)
        {
            GLES31.glDeleteProgram(shaderProgramObject);
            shaderProgramObject = 0;
        }
    }
}
