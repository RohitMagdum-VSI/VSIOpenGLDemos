package com.astromedicomp.opengl_2d_checkerboard;

import android.content.Context; //For drawing context related
import android.opengl.GLSurfaceView; //For OpenGL Surface View and all related
import javax.microedition.khronos.opengles.GL10; //For OpenGLES 1.0 needed as param type GL10
import javax.microedition.khronos.egl.EGLConfig; //For EGLConfig needed as param type EGLConfig
import android.opengl.GLES32; // For OpenGLES 3.2
//import android.view.Gravity;
import android.view.MotionEvent; // For "MotionEvent"
import android.view.GestureDetector; // For GestureDetector
import android.view.GestureDetector.OnGestureListener; // OnGestureListener
import android.view.GestureDetector.OnDoubleTapListener; // OnDoubleTapListener

//For vbo
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

//For Matrix math
import android.opengl.Matrix;

import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.opengl.GLUtils;

//A View for OpenGLES3 graphics which also receives touch events
public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnGestureListener, OnDoubleTapListener
{
	//Class variables or class fields
	private final Context context;

	private GestureDetector gestureDetector;

	//Shader Object
	private int vertexShaderObject;
	private int fragmentShaderObject;

	//Program Object
	private int shaderProgramObject;

	//As there are no pointers in java, when we want to use any variable as out parameter
	//we use array with only one member.
	//Vertex Array Object
	private int[] vao_square = new int[1];
	private int[] vbo_position = new int[1];
	private int[] vbo_texture = new int[1];

	private int mvpUniform;
	private int texture0_sampler_uniform;

	private int checkImageHeight = 64;
	private int checkImageWidth = 64;

	private int[][][] checkImage = new int[checkImageHeight][checkImageWidth][4];

	private int[] texture_checkerboard = new int[1];

	private byte[] checkImage1d = new byte[16384];

	private int iterator=0;

	//4 x 4 matrix
	private float perspectiveProjectionMatrix[] = new float[16];

	public GLESView(Context drawingContext)
	{
		super(drawingContext);

		context = drawingContext;

		//Accordingly set EGLContext to current supported version of OpenGL-ES
		setEGLContextClientVersion(3);

		//Set Renderer for drawing on GLSurfaceView
		setRenderer(this);

		//Render the view only when there is a change in drawing data
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);

		//this means 'handle' i.e who is going to handle 
		gestureDetector = new GestureDetector(context,this,null,false);
	
		//this means 'handle' i.e who is going to handle
		gestureDetector.setOnDoubleTapListener(this);
	}

	//overriden method of GLSurfaceView.Renderer(Init Code)
	@Override
	public void onSurfaceCreated(GL10 gl,EGLConfig config)
	{
		//OpenGL-ES version check
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("HAD: OpenGL-ES Version"+version);//"+" for concatination

		String glslVersion=gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("HAD: GLSL Version"+glslVersion);

		initialize(gl);
	}

	//overriden method of GLSurfaceView.Renderer(Change Size Code)
	@Override
	public void onSurfaceChanged(GL10 unused,int width,int height)
	{
		resize(width,height);
	}

	//overriden method of GLSurfaceView.Renderer(Rendering Code)
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}

	//Handling 'onTouchEvent' Is The Most IMPORTANT, Because It Triggers All Gesture And Events
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		//code
		int eventaction=e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//Do not write any code here because already written 'onDOubleTap'
		return(true);
	}

	//abstract method from OnDoubleTapListener so must be implemented
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onDown(MotionEvent e)
	{
		//Do not write any code here because already written 'onSingleTapConfirmed'
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onFling(MotionEvent e1,MotionEvent e2,float velocityX,float velocityY)
	{
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onLongPress(MotionEvent e)
	{
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2,float distanceX,float distanceY)
	{
		uninitialize();
		System.exit(0);
		return(true);
	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public void onShowPress(MotionEvent e)
	{

	}

	//abstract method from OnGestureListener so must be implemented
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return(true);
	}

	private void initialize(GL10 gl)
	{
		//Vertex Shader
		//Create Vertex Shader

		vertexShaderObject=GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		//vertex shader source code
		final String vertexShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"in vec4 vPosition;"+
			"in vec2 vTexture0_Coord;"+
			"uniform mat4 u_mvp_matrix;"+
			"out vec2 out_texture0_coord;"+
			"void main(void)"+
			"{"+
			"gl_Position = u_mvp_matrix * vPosition;"+
			"out_texture0_coord=vTexture0_Coord;"+
			"}"
		);

		//Provide Source Code to Shader
		GLES32.glShaderSource(vertexShaderObject,vertexShaderSourceCode);

		//Compile Shader & Check errors
		GLES32.glCompileShader(vertexShaderObject);
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null; 
		GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(vertexShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("HAD: Vertex Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Fragment Shader
		//Create Shader
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		//Fragment Shader Source Code
		final String fragmentShaderSourceCode = String.format
		(
			"#version 320 es"+
			"\n"+
			"precision highp float;"+
			"in vec2 out_texture0_coord;"+
			"out vec4 FragColor;"+
			"uniform highp sampler2D u_texture0_sampler;"+
			"void main(void)"+
			"{"+
			"FragColor = texture(u_texture0_sampler , out_texture0_coord);"+
			"}"
		);

		GLES32.glShaderSource(fragmentShaderObject,fragmentShaderSourceCode);

		//Compile Shader and check for errors
		GLES32.glCompileShader(fragmentShaderObject);
		iShaderCompiledStatus[0] = 0;//reinitialize
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog=null;//reinitialize

		GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_COMPILE_STATUS,iShaderCompiledStatus,0);
		if(iShaderCompiledStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetShaderiv(fragmentShaderObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("HAD: Fragment Shader Compilation Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Create Shader Program
		shaderProgramObject = GLES32.glCreateProgram();

		//Attach Vertex Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,vertexShaderObject);

		//Attach Fragment Shader to Program Object
		GLES32.glAttachShader(shaderProgramObject,fragmentShaderObject);

		//Pre-link binding of shader program object with vertex shader attributes
		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_VERTEX,"vPosition");

		GLES32.glBindAttribLocation(shaderProgramObject,GLESMacros.HAD_ATTRIBUTE_TEXTURE0,"vTexture0_Coord");

		//Link 2 shaders together to shader program object
		GLES32.glLinkProgram(shaderProgramObject);
		int[] iShaderProgramLinkStatus = new int[1];
		iInfoLogLength[0] = 0;//reinitialize
		szInfoLog = null;//reinitialize

		GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_LINK_STATUS,iShaderProgramLinkStatus,0);
		if(iShaderProgramLinkStatus[0] == GLES32.GL_FALSE)
		{
			GLES32.glGetProgramiv(shaderProgramObject,GLES32.GL_INFO_LOG_LENGTH,iInfoLogLength,0);
			if(iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				System.out.println("HAD: Shader Program Link Log = "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}

		//Get MVP uniform location
		mvpUniform = GLES32.glGetUniformLocation(shaderProgramObject,"u_mvp_matrix");
		texture0_sampler_uniform=GLES32.glGetUniformLocation(shaderProgramObject,"u_texture0_sampler");

		//System.out.println("HAD: Before LoadGLTexture");
		loadGLTexture();
		//System.out.println("HAD: After LoadGLTexture");

		//Vertices, Color,Shader Attribs, Vbo, Vao initializations
		/*final float squareVertices[] = new float[]
		{
			1.0f,1.0f,0.0f,
			-1.0f,1.0f,0.0f,
			-1.0f,-1.0f,0.0f,
			1.0f,-1.0f,0.0f
		};*/ 

		final float squareTexcoords[] =new float[]
		{
			1.0f,1.0f,
			0.0f,1.0f,
			0.0f,0.0f,
			1.0f,0.0f
		};

		/*****************Square*****************/
		GLES32.glGenVertexArrays(1,vao_square,0);
		GLES32.glBindVertexArray(vao_square[0]);

		/****************Square Position**************/
		GLES32.glGenBuffers(1,vbo_position,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_position[0]);

		/*ByteBuffer byteBuffer=ByteBuffer.allocateDirect(squareVertices.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer=byteBuffer.asFloatBuffer();
		verticesBuffer.put(squareVertices);
		verticesBuffer.position(0);*/

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,48,null,GLES32.GL_DYNAMIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_VERTEX,3,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_VERTEX);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		/****************Square Texture**************/
		GLES32.glGenBuffers(1,vbo_texture,0);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_texture[0]);

		ByteBuffer byteBuffer=ByteBuffer.allocateDirect(squareTexcoords.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer textureBuffer=byteBuffer.asFloatBuffer();
		textureBuffer.put(squareTexcoords);
		textureBuffer.position(0);

		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,squareTexcoords.length*4,textureBuffer,GLES32.GL_STATIC_DRAW);

		GLES32.glVertexAttribPointer(GLESMacros.HAD_ATTRIBUTE_TEXTURE0,2,GLES32.GL_FLOAT,false,0,0);

		GLES32.glEnableVertexAttribArray(GLESMacros.HAD_ATTRIBUTE_TEXTURE0);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);
		GLES32.glBindVertexArray(0);

		//Enable DepthTest
		GLES32.glEnable(GLES32.GL_DEPTH_TEST);

		//Specify Depth test to be done
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);

		//We will always cull the back faces for better performance
		GLES32.glEnable(GLES32.GL_CULL_FACE);

		//Set the background frame color
		GLES32.glClearColor(0.0f,0.0f,0.0f,1.0f);

		//Set ProjectionMatrix to identity matrix
		Matrix.setIdentityM(perspectiveProjectionMatrix,0);
	}

	private void loadGLTexture()
	{
		//BitmapFactory.Options options = new BitmapFactory.Options();

		//options.inScaled=false;

		//Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(),checkImage,options);

		//int[] texture = new int[1];

		MakeCheckImage();

		GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT,1);

		GLES32.glGenTextures(1,texture_checkerboard,0);

		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D,texture_checkerboard[0]);

		GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D,GLES32.GL_TEXTURE_WRAP_S,GLES32.GL_REPEAT);
		GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D,GLES32.GL_TEXTURE_WRAP_T,GLES32.GL_REPEAT);
		GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D,GLES32.GL_TEXTURE_MAG_FILTER,GLES32.GL_NEAREST);
		GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D,GLES32.GL_TEXTURE_MIN_FILTER,GLES32.GL_NEAREST);

		//GLUtils.texImage2D(GLES32.GL_TEXTURE_2D,0,bitmap,0);

		ByteBuffer byteBuffer = ByteBuffer.allocateDirect(checkImageWidth*checkImageHeight*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		byteBuffer.put(checkImage1d);
		byteBuffer.position(0);

		GLES32.glTexImage2D(GLES32.GL_TEXTURE_2D,0,GLES32.GL_RGBA,checkImageWidth,checkImageHeight,0,GLES32.GL_RGBA,GLES32.GL_UNSIGNED_BYTE,byteBuffer);

		GLES32.glGenerateMipmap(GLES32.GL_TEXTURE_2D);
	}

	private void MakeCheckImage()
	{
		int i,j,k,c;
 
		for(i=0;i<checkImageHeight;i++)
		{
			for(j=0;j<checkImageWidth;j++)
			{
				//a= i & 0x8;
				//b= j & 0x8;
				//c=(a==0)^(b==0);
				if((((i & 0x8)==0) ^ ((j & 0x8)==0))==true)
				{
					c=255;
				}
				else
				{
					c=0;
				}
				//c=((i & 0x8) ^ (j & 0x8));
				//c=c*255;
				/*checkImage[i][j][0]=c;
				checkImage[i][j][1]=c;
				checkImage[i][j][2]=c;
				checkImage[i][j][3]=255;*/
				checkImage1d[iterator]=(byte)c;
				checkImage1d[iterator+1]=(byte)c;
				checkImage1d[iterator+2]=(byte)c;
				checkImage1d[iterator+3]=(byte)255;
				iterator+=4;				
			}
		}

		/*for(i=0;i<checkImageHeight;i++)
		{
			for(j=0;j<checkImageWidth;j++)
			{
				for(k=0;k<4;k++)
				{
					checkImage1d[iterator]=(byte)checkImage[i][j][k];
					//System.out.println("HAD: checkImage 1D is " + checkImage1d[iterator]);
					iterator++;
				}
			}
		}*/
	}

	private void resize(int width,int height)
	{
		//Adjust the viewport based on geometry changes such as screen rotation
		GLES32.glViewport(0,0,width,height);

		//Perspective Projection
		Matrix.perspectiveM(perspectiveProjectionMatrix,0,45.0f,(float)width/(float)height,0.1f,100.0f);
	}

	public void draw()
	{
		float[] squareVertices = new float[12];
		//Draw background color
		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT|GLES32.GL_DEPTH_BUFFER_BIT);

		//Use Shader Program
		GLES32.glUseProgram(shaderProgramObject);

		float modelViewMatrix[] = new float[16];
		float modelViewProjectionMatrix[] = new float[16];
		float rotationMatrix[] = new float[16];
		float scaleMatrix[] = new float[16];

		//set ModelView and ModelViewProjection matrices to identity
		Matrix.setIdentityM(modelViewMatrix,0);
		Matrix.setIdentityM(modelViewProjectionMatrix,0);

		Matrix.translateM(modelViewMatrix,0,0.0f,0.0f,-6.0f);

		Matrix.multiplyMM(modelViewProjectionMatrix,0,perspectiveProjectionMatrix,0,modelViewMatrix,0);

		GLES32.glUniformMatrix4fv(mvpUniform,1,false,modelViewProjectionMatrix,0);

		GLES32.glBindVertexArray(vao_square[0]);

		squareVertices[0]=0.0f;
		squareVertices[1]=1.0f;
		squareVertices[2]=0.0f;
		squareVertices[3]=-2.0f;
		squareVertices[4]=1.0f;
		squareVertices[5]=0.0f;
		squareVertices[6]=-2.0f;
		squareVertices[7]=-1.0f;
		squareVertices[8]=0.0f;
		squareVertices[9]=0.0f;
		squareVertices[10]=-1.0f;
		squareVertices[11]=0.0f;

		ByteBuffer byteBuffer = ByteBuffer.allocateDirect(squareVertices.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		FloatBuffer verticesBuffer = byteBuffer.asFloatBuffer();
		verticesBuffer.put(squareVertices);
		verticesBuffer.position(0);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_position[0]);
		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,squareVertices.length*4,verticesBuffer,GLES32.GL_DYNAMIC_DRAW);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D,texture_checkerboard[0]);
		GLES32.glUniform1i(texture0_sampler_uniform,0);

		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,0,4);

		squareVertices[0]=2.41421f;
		squareVertices[1]=1.0f;
		squareVertices[2]=-1.41421f;
		squareVertices[3]=1.0f;
		squareVertices[4]=1.0f;
		squareVertices[5]=0.0f;
		squareVertices[6]=1.0f;
		squareVertices[7]=-1.0f;
		squareVertices[8]=0.0f;
		squareVertices[9]=2.41421f;
		squareVertices[10]=-1.0f;
		squareVertices[11]=-1.41421f;

		byteBuffer = ByteBuffer.allocateDirect(squareVertices.length*4);
		byteBuffer.order(ByteOrder.nativeOrder());
		verticesBuffer = byteBuffer.asFloatBuffer();
		verticesBuffer.put(squareVertices);
		verticesBuffer.position(0);

		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,vbo_position[0]);
		GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,squareVertices.length*4,verticesBuffer,GLES32.GL_DYNAMIC_DRAW);
		GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER,0);

		GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D,texture_checkerboard[0]);
		GLES32.glUniform1i(texture0_sampler_uniform,0);

		GLES32.glDrawArrays(GLES32.GL_TRIANGLE_FAN,0,4);

		GLES32.glBindVertexArray(0);

		//un-use shader program
		GLES32.glUseProgram(0);

		//Like SwapBuffers() in Windows
		requestRender();
	}

	public void uninitialize()
	{
		if(vao_square[0] != 0)
		{
			GLES32.glDeleteVertexArrays(1,vao_square,0);
			vao_square[0]=0;
		}

		if(vbo_position[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_position,0);
			vbo_position[0]=0;
		}

		if(vbo_texture[0] != 0)
		{
			GLES32.glDeleteBuffers(1,vbo_texture,0);
			vbo_texture[0]=0;
		}

		if(texture_checkerboard[0]!=0)
		{
			GLES32.glDeleteTextures(1,texture_checkerboard,0);
			texture_checkerboard[0]=0;
		}

		if(shaderProgramObject != 0)
		{
			if(vertexShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,vertexShaderObject);
				GLES32.glDeleteShader(vertexShaderObject);
				vertexShaderObject=0;
			}

			if(fragmentShaderObject != 0)
			{
				GLES32.glDetachShader(shaderProgramObject,fragmentShaderObject);
				GLES32.glDeleteShader(fragmentShaderObject);
				fragmentShaderObject=0;
			}
		}

		if(shaderProgramObject != 0)
		{
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject=0;
		}
	}
}