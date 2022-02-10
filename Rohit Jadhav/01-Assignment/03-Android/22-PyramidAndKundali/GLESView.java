package com.rohit_r_jadhav.pyramid_kundali;

import android.opengl.GLSurfaceView;
import android.opengl.GLES32;
import android.opengl.Matrix;

import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.opengl.GLUtils;

import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnDoubleTapListener;
import android.view.GestureDetector.OnGestureListener;
import android.content.Context;

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer, OnDoubleTapListener, OnGestureListener{

	GestureDetector gestureDetector;
	Context context;

	GLESView(Context drawingContext){
		super(drawingContext);
		context = drawingContext;

		gestureDetector = new GestureDetector(drawingContext, this, null, false);
		gestureDetector.setOnDoubleTapListener(this);

		setEGLContextClientVersion(3);
		setRenderer(this);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	@Override
	public boolean onTouchEvent(MotionEvent e){
		int action = e.getAction();
		if(!gestureDetector.onTouchEvent(e))
			super.onTouchEvent(e);
		return(true);
	}

	/********** Methods From OnDoubleTapListener **********/
	@Override
	public boolean onDoubleTap(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onDoubleTapEvent(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapConfirmed(MotionEvent e){
		return(true);
	}

	/********** Methods From OnGestureListener **********/
	@Override
	public boolean onDown(MotionEvent e){
		return(true);
	}

	@Override
	public boolean onSingleTapUp(MotionEvent e){
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
	public void onShowPress(MotionEvent e){

	}

	@Override
	public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY){
		uninitialize();
		System.exit(0);
		return(true);
	}



	/*********** Methods from GLSurfaceView.Renderer **********/
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config){

		String version = gl.glGetString(GL10.GL_VERSION);
		String glsl_version = gl.glGetString(GLES32.GL_SHADING_LANGUAGE_VERSION);
		String renderer = gl.glGetString(GL10.GL_RENDERER);
		String vendor = gl.glGetString(GL10.GL_VENDOR);

		System.out.println("RTR: OpenGL Version: " + version);
		System.out.println("RTR: OpenGLSL Version: " + glsl_version);
		System.out.println("RTR: Renderer: " + renderer);
		System.out.println("RTR: Vendor: " + vendor);

		initialize();
	}

	@Override 
	public void onSurfaceChanged(GL10 unused, int width, int height){
		resize(width, height);
	}

	@Override
	public void onDrawFrame(GL10 unused){
		update();
		display();
	}




	//For Pyramid
	private int vao_Pyramid[] = new int[1];
	private int vbo_Pyramid_Position[] = new int[1];
	private int vbo_Pyramid_TexCoord[] = new int[1];
	private int texture_stone[] = new int[1];
	private float angle_pyramid = 0.0f;

	//For Cube
	private int vao_Cube[] = new int[1];
	private int vbo_Cube_Position[] = new int[1];
	private int vbo_Cube_TexCoord[] = new int[1];
	private int texture_kundali[] = new int[1];
	private float angle_cube = 360.0f;

	//For Projection
	private float perspectiveProjectionMatrix[] = new float[4*4];

	//For Uniform
	private int mvpUniform;
	private int samplerUniform;

	//For Shaders
	private int vertexShaderObject;
	private int fragmentShaderObject;
	private int shaderProgramObject;

	private void initialize(){


		/********** Vertex Shader **********/
		vertexShaderObject = GLES32.glCreateShader(GLES32.GL_VERTEX_SHADER);

		final String vertexShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"in vec4 vPosition;" +
				"in vec2 vTexCoord;" +
				"out vec2 outTexCoord;" +
				"uniform mat4 u_mvp_matrix;" +
				"void main(void)" +
				"{" +
					"gl_Position = u_mvp_matrix * vPosition;" +
					"outTexCoord = vTexCoord;" +
				"}"
			);

		GLES32.glShaderSource(vertexShaderObject, vertexShaderSourceCode);

		GLES32.glCompileShader(vertexShaderObject);

		int iCompileStatus[] = new int[1];
		int iInfoLogLength[] = new int[1];
		String szInfoLog = null;

		GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_COMPILE_STATUS, iCompileStatus, 0);
		if(iCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(vertexShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(vertexShaderObject);
				System.out.println("RTR: Vertex Shader Compilation Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}



		/********** Fragment Shader **********/
		fragmentShaderObject = GLES32.glCreateShader(GLES32.GL_FRAGMENT_SHADER);

		final String fragmentShaderSourceCode = String.format(
				"#version 320 es" +
				"\n" +
				"precision highp float;" +
				"precision highp sampler2D;" +
				"in vec2 outTexCoord;" +
				"out vec4 FragColor;" +
				"uniform sampler2D u_sampler;" +
				"void main(void)" +
				"{" +
					"FragColor = texture(u_sampler, outTexCoord);" +
				"}"
			);
		
		GLES32.glShaderSource(fragmentShaderObject, fragmentShaderSourceCode);

		GLES32.glCompileShader(fragmentShaderObject);

		iCompileStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_COMPILE_STATUS, iCompileStatus, 0);
		if(iCompileStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetShaderiv(fragmentShaderObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				szInfoLog = GLES32.glGetShaderInfoLog(fragmentShaderObject);
				System.out.println("RTR: Fragment Shader Compilation Error: " + szInfoLog);
				szInfoLog = null;
				uninitialize();
				System.exit(0);
			}
		}



		/********** Program Object **********/
		shaderProgramObject = GLES32.glCreateProgram();

		GLES32.glAttachShader(shaderProgramObject, vertexShaderObject);
		GLES32.glAttachShader(shaderProgramObject, fragmentShaderObject);

		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_POSITION, "vPosition");
		GLES32.glBindAttribLocation(shaderProgramObject, GLESMacros.AMC_ATTRIBUTE_TEXCOORD0, "vTexCoord");

		GLES32.glLinkProgram(shaderProgramObject);

		int iProgramLinkStatus[] = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;

		GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_LINK_STATUS, iProgramLinkStatus, 0);
		if(iProgramLinkStatus[0] == GLES32.GL_FALSE){
			GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if(iInfoLogLength[0] > 0){
				 szInfoLog = GLES32.glGetProgramInfoLog(shaderProgramObject);
				 System.out.println("RTR: Program Linking Error: "+ szInfoLog);
				 szInfoLog = null;
				 uninitialize();
				 System.exit(0);
			}
		}

		mvpUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");
		samplerUniform = GLES32.glGetUniformLocation(shaderProgramObject, "u_sampler");




		/********** Positions **********/
		float pyramid_Position[] = new float[]{
			//Face
			0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			//Right
			0.0f, 1.0f, 0.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, -1.0f,
			//Back
			0.0f, 1.0f, 0.0f,
			1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			//Left
			0.0f, 1.0f, 0.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, 1.0f
		};

		float cube_Position[] = new float[]{
			//Top
			1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			//Bottom
			1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			//Front
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			//Back
			1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			1.0f, -1.0f, -1.0f,
			//Right
			1.0f, 1.0f, -1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, -1.0f,
			//Left
			-1.0f, 1.0f, 1.0f, 
			-1.0f, 1.0f, -1.0f, 
			-1.0f, -1.0f, -1.0f, 
			-1.0f, -1.0f, 1.0f
		};

		float pyramid_TexCoord[] = new float[] {
			0.5f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			0.5f, 0.0f,
			1.0f, 1.0f,
			0.0f, 1.0f,
			
			0.5f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			0.5f, 0.0f,
			1.0f, 1.0f,
			0.0f, 1.0f
			
		};

		float cube_TexCoord[] = new float[]{
			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f,

			1.0f, 0.0f,
			0.0f, 0.0f,
			0.0f, 1.0f,
			1.0f, 1.0f
		};


		

		/********** Pyramid **********/
		GLES32.glGenVertexArrays(1, vao_Pyramid, 0);
		GLES32.glBindVertexArray(vao_Pyramid[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Pyramid_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Pyramid_Position[0]);

				/********** For glBufferData **********/
				ByteBuffer Pyramid_positionByteBuffer = ByteBuffer.allocateDirect(pyramid_Position.length * 4);
				Pyramid_positionByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer Pyramid_positionBuffer = Pyramid_positionByteBuffer.asFloatBuffer();
				Pyramid_positionBuffer.put(pyramid_Position);
				Pyramid_positionBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							pyramid_Position.length * 4,
							Pyramid_positionBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,	
									3,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** TexCoord **********/
			GLES32.glGenBuffers(1, vbo_Pyramid_TexCoord, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Pyramid_TexCoord[0]);

				/********** For glBufferData **********/
				ByteBuffer pyramidTexCoord_ByteBuffer = ByteBuffer.allocateDirect(pyramid_TexCoord.length * 4);
				pyramidTexCoord_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer pyramidTexCoord_FloatBuffer = pyramidTexCoord_ByteBuffer.asFloatBuffer();
				pyramidTexCoord_FloatBuffer.put(pyramid_TexCoord);
				pyramidTexCoord_FloatBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							pyramid_TexCoord.length * 4,
							pyramidTexCoord_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0,
									2,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);



		/********** Cube **********/
		GLES32.glGenVertexArrays(1, vao_Cube, 0);
		GLES32.glBindVertexArray(vao_Cube[0]);

			/********** Position **********/
			GLES32.glGenBuffers(1, vbo_Cube_Position, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Cube_Position[0]);

				/********** For glBufferData **********/
				ByteBuffer Cube_positionByteBuffer = ByteBuffer.allocateDirect(cube_Position.length * 4);
				Cube_positionByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer Cube_positionBuffer = Cube_positionByteBuffer.asFloatBuffer();
				Cube_positionBuffer.put(cube_Position);
				Cube_positionBuffer.position(0);

			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER,
							cube_Position.length * 4,
							Cube_positionBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_POSITION,
									3,
									GLES32.GL_FLOAT,
									false,
									0,  0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_POSITION);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);


			/********** TexCoord **********/
			GLES32.glGenBuffers(1, vbo_Cube_TexCoord, 0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, vbo_Cube_TexCoord[0]);

				/********** For glBufferData **********/
				ByteBuffer cubeTexCoord_ByteBuffer = ByteBuffer.allocateDirect(cube_TexCoord.length * 4);
				cubeTexCoord_ByteBuffer.order(ByteOrder.nativeOrder());
				FloatBuffer cubeTexCoord_FloatBuffer = cubeTexCoord_ByteBuffer.asFloatBuffer();
				cubeTexCoord_FloatBuffer.put(cube_TexCoord);
				cubeTexCoord_FloatBuffer.position(0);


			GLES32.glBufferData(GLES32.GL_ARRAY_BUFFER, 
							cube_TexCoord.length * 4,
							cubeTexCoord_FloatBuffer,
							GLES32.GL_STATIC_DRAW);

			GLES32.glVertexAttribPointer(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0,
									2,
									GLES32.GL_FLOAT,
									false,
									0, 0);

			GLES32.glEnableVertexAttribArray(GLESMacros.AMC_ATTRIBUTE_TEXCOORD0);
			GLES32.glBindBuffer(GLES32.GL_ARRAY_BUFFER, 0);

		GLES32.glBindVertexArray(0);


		GLES32.glEnable(GLES32.GL_DEPTH_TEST);
		GLES32.glDepthFunc(GLES32.GL_LEQUAL);


			GLES32.glEnable(GLES32.GL_TEXTURE_2D);
			texture_stone[0] = loadTexture(R.raw.stone);
			texture_kundali[0] = loadTexture(R.raw.kundali);

			System.out.println("RTR: Stone Texture: " + texture_stone[0]);
			System.out.println("RTR: Kundali Texture: " + texture_kundali[0]);

		GLES32.glDisable(GLES32.GL_CULL_FACE);

		GLES32.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	}


	private int loadTexture(int imageFileResourceID){
		int texture[] = new int[1];

		BitmapFactory.Options options = new BitmapFactory.Options();

		options.inScaled = false;

		Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), imageFileResourceID, options);

		GLES32.glPixelStorei(GLES32.GL_UNPACK_ALIGNMENT, 4);
		GLES32.glGenTextures(1, texture, 0);
		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture[0]);

			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MAG_FILTER, GLES32.GL_LINEAR);
			GLES32.glTexParameteri(GLES32.GL_TEXTURE_2D, GLES32.GL_TEXTURE_MIN_FILTER, GLES32.GL_LINEAR_MIPMAP_LINEAR);

			GLUtils.texImage2D(GLES32.GL_TEXTURE_2D,
							0,
							bitmap,
							0);
			GLES32.glGenerateMipmap(GLES32.GL_TEXTURE_2D);

		GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, 0);
		return(texture[0]);

	}


	private void uninitialize(){

		if(texture_kundali[0] != 0){
			GLES32.glDeleteTextures(1, texture_kundali, 0);
			texture_kundali[0] = 0;
		}

		if(texture_stone[0] != 0){
			GLES32.glDeleteTextures(1, texture_stone, 0);
			texture_stone[0] = 0;
		}

		if(vbo_Cube_TexCoord[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Cube_TexCoord, 0);
			vbo_Cube_TexCoord[0] = 0;
		}

		if(vbo_Cube_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Cube_Position, 0);
			vbo_Cube_Position[0] = 0;
		}

		if(vao_Cube[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Cube, 0);
			vao_Cube[0] = 0;
		}

		if(vbo_Pyramid_TexCoord[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Pyramid_TexCoord, 0);
			vbo_Pyramid_TexCoord[0] = 0;
		}

		if(vbo_Pyramid_Position[0] != 0){
			GLES32.glDeleteBuffers(1, vbo_Pyramid_Position, 0);
			vbo_Pyramid_Position[0] = 0;
		}

		if(vao_Pyramid[0] != 0){
			GLES32.glDeleteVertexArrays(1, vao_Pyramid, 0);
			vao_Pyramid[0] = 0;
		}


		if(shaderProgramObject != 0){

			GLES32.glUseProgram(shaderProgramObject);

				if(fragmentShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, fragmentShaderObject);
					GLES32.glDeleteShader(fragmentShaderObject);
					fragmentShaderObject = 0;
					System.out.println("RTR: Fragment Shader Detached and Deleted!!");
				}

				if(vertexShaderObject != 0){
					GLES32.glDetachShader(shaderProgramObject, vertexShaderObject);
					GLES32.glDeleteShader(vertexShaderObject);
					vertexShaderObject = 0;
					System.out.println("RTR: Vertex Shader Detached and Deleted!!");
				}
				
			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;
		}

		/*if(shaderProgramObject != 0){

			int iShaderCount[] = new int[1];
			int iShaderNo;

			GLES32.glUseProgram(shaderProgramObject);
				GLES32.glGetProgramiv(shaderProgramObject, GLES32.GL_ATTACHED_SHADERS, iShaderCount, 0);
				System.out.println("RTR: ShaderCount: "+ iShaderCount[0]);
				int iShaders[] = new int[iShaderCount[0]];
				GLES32.glGetAttachedShaders(shaderProgramObject, iShaderCount[0],
										iShaderCount, 0,
										iShaders, 0);

				for(iShaderNo = 0; iShaderNo < iShaderCount[0] ; iShaderNo++){
					GLES32.glDetachShader(shaderProgramObject, iShaders[iShaderNo]);
					GLES32.glDeleteShader(iShaders[iShaderNo]);
					iShaders[iShaderNo] = 0;
					System.out.println("RTR: Shader Deleted!!");
				}
			GLES32.glUseProgram(0);
			GLES32.glDeleteProgram(shaderProgramObject);
			shaderProgramObject = 0;
		}*/
	}


	private void resize(int width, int height){
		if(height == 0)
			height = 1;

		GLES32.glViewport(0, 0, width, height);

		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
		Matrix.perspectiveM(perspectiveProjectionMatrix, 0,
						45.0f,
						(float)width / (float)height,
						0.1f,
						100.0f);
	}


	private void display(){

		float translateMatrix[] = new float[4 * 4];
		float rotateMatrix[] = new float[4 * 4];
		float scaleMatrix[] = new float[4 * 4];
		float modelViewMatrix[] = new float[4 * 4];
		float modelViewProjectionMatrix[] = new float[4 * 4];

		GLES32.glClear(GLES32.GL_COLOR_BUFFER_BIT | GLES32.GL_DEPTH_BUFFER_BIT);

		GLES32.glUseProgram(shaderProgramObject);

			/********** Pyramid **********/
			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(rotateMatrix, 0);
			Matrix.setIdentityM(scaleMatrix, 0);
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.translateM(translateMatrix, 0, -2.0f, 0.0f, -5.50f);
			Matrix.setRotateM(rotateMatrix, 0, angle_pyramid, 0.0f, 1.0f, 0.0f);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

			GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);


			GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_stone[0]);
			GLES32.glUniform1i(samplerUniform, 0);

			GLES32.glBindVertexArray(vao_Pyramid[0]);
				GLES32.glDrawArrays(GLES32.GL_TRIANGLES, 0, 12);
			GLES32.glBindVertexArray(0);


			/********** Cube **********/
			Matrix.setIdentityM(translateMatrix, 0);
			Matrix.setIdentityM(rotateMatrix, 0);
			Matrix.setIdentityM(scaleMatrix, 0);
			Matrix.setIdentityM(modelViewMatrix, 0);
			Matrix.setIdentityM(modelViewProjectionMatrix, 0);

			Matrix.translateM(translateMatrix, 0, 2.0f, 0.0f, -5.50f);
			Matrix.rotateM(rotateMatrix, 0, angle_cube, 1.0f, 1.0f, 1.0f);
			Matrix.scaleM(scaleMatrix, 0, 0.9f, 0.9f, 0.9f);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, translateMatrix, 0);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, rotateMatrix, 0);
			Matrix.multiplyMM(modelViewMatrix, 0, modelViewMatrix, 0, scaleMatrix, 0);
			Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);

			GLES32.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);

			GLES32.glActiveTexture(GLES32.GL_TEXTURE0);
			GLES32.glBindTexture(GLES32.GL_TEXTURE_2D, texture_kundali[0]);
			GLES32.glUniform1i(samplerUniform, 0);

			GLES32.glBindVertexArray(vao_Cube[0]);
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


	private void update(){
		angle_pyramid = angle_pyramid + 2.00f;
		angle_cube = angle_cube - 2.0f;

		if(angle_pyramid > 360.0f)
			angle_pyramid = 0.0f;

		if(angle_cube < 0.0f)
			angle_cube = 360.0f;
	}
}