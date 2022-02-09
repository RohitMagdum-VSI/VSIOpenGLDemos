package com.astromedicomp.tweak_smiley;

import android.content.Context;
import android.opengl.GLSurfaceView;//	for open gl surface view all related.
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.EGLConfig;
import android.opengl.GLES30;	//	Change this version as per requirement.
import android.view.MotionEvent;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.GestureDetector.OnDoubleTapListener;

//	For VBO
import java.nio.ByteBuffer;	//	nio - non-blocking I/O
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import android.opengl.Matrix;

import android.graphics.BitmapFactory;	//	Texture factory
import android.graphics.Bitmap;	//	for PNG image
import android.opengl.GLUtils;	//	for texImage2D()

public class GLESView extends GLSurfaceView implements GLSurfaceView.Renderer,OnGestureListener,OnDoubleTapListener{

	private GestureDetector gestureDetector;
	private final Context context;
	
	private int shaderObjectVertex;
	private int shaderObjectFragment;
	private int shaderObjectProgram;
	
	private int[] VAORect = new int[1];
	private int[] VBOPosition = new int[1];
	private int[] VBOTexture = new int[1];
	
	private int mvpUniform;
	private int texture0_sampler_uniform;
	
	private int[] texture_smiley = new int[1];
	private int[] g_iTextureWhite = new int[1];
	
	private int[] g_arrbyWhiteImage = new int [GLESMacros.IMAGE_HEIGHT * GLESMacros.IMAGE_WIDTH];
	
	private float perspectiveProjectionMatrix[] = new float[16];	//	4*4 Matrix.
	
	private int g_iKey = 0;
	
    public GLESView(Context drawingContext)
	{
		super(drawingContext);
		context = drawingContext;
		
		//
		//	Accordingly set EGLContext to supported version of OpenGL-ES
		//
		setEGLContextClientVersion(3);
		
		//
		//	Set renderer for drawing on the GLSurfaceView.
		//
		setRenderer(this);	//	Because of this OnSurfaceCreated() get called.
		
		//
		//	Render the view only when there is change in the drawing data.
		//
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
		
		
		gestureDetector = new GestureDetector(context,this, null, false);
		gestureDetector.setOnDoubleTapListener(this);	//	this means handler i.e) who is going to handle.
	}
	
	//---------------------------------------------------------------------------
	//+	Overriden methods of GLSurfaceView.Renderer
	
	//
	//	Init code
	//
	@Override
	public void onSurfaceCreated(GL10 gl, EGLConfig config)
	{
		//	OpenGL-ES version check.
		String version = gl.glGetString(GL10.GL_VERSION);
		System.out.println("RTR: "+version);
		//	Get GLSL version.
		String glslVersion = gl.glGetString(GLES30.GL_SHADING_LANGUAGE_VERSION);
		System.out.println("RTR: "+glslVersion);
		initialize(gl);
	}
	
	//
	//	Change size code.
	//
	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height)
	{
		resize(width, height);
	}
	
	//
	//	Rendering code.
	//
	@Override
	public void onDrawFrame(GL10 unused)
	{
		draw();
	}
	//-	Overriden methods of GLSurfaceView.Renderer
	//---------------------------------------------------------------------------
	
	//
	//	Handling 'onTouchEvent' is the most IMPORTANT.
	//	Because it triggers all gesture and tap events.
	//
	@Override
	public boolean onTouchEvent(MotionEvent event)
	{
		int eventAction = event.getAction();
		if (!gestureDetector.onTouchEvent(event))
		{
			super.onTouchEvent(event);
		}
		return(true);
	}
	
	//---------------------------------------------------------------------------
	//+	Abstract method from OnDoubleTapListener
	
	//
	//	onDoubleTap
	//
	@Override
	public boolean onDoubleTap(MotionEvent e)
	{
		System.out.println("RTR:Double Tap");
		g_iKey += 1;
		if (g_iKey > 4)
		{
			g_iKey = 0;
		}

		return(true);
	}
	
	//
	//	onDoubleTapEvent
	//
	@Override
	public boolean onDoubleTapEvent(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onDoubleTap'
		return true;
	}
	
	//
	//	onSingleTapConfirmed
	//
	@Override
	public boolean onSingleTapConfirmed(MotionEvent e)
	{
		System.out.println("RTR:Single Tap");
		return true;
	}
	
	//
	//	onDown
	//
	@Override
	public boolean onDown(MotionEvent e)
	{
		//	Do not write any code here bacause already written 'onSingleTapConfirmed'
		return true;
	}
	//-	Abstract method from OnDoubleTapListener
	//---------------------------------------------------------------------------
	
	//---------------------------------------------------------------------------
	//+	abstract method from OnGestureListener so must be implemented
	
	//
	//	onFling
	//
	@Override
	public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY)
	{
		System.out.println("RTR:Flingg");
		return true;
	}
	
	//
	//	onLongPress
	//
	@Override
	public void onLongPress(MotionEvent e)
	{
		System.out.println("RTR:Long press");
	}
	
	//
	//	onScroll
	//
	@Override
	public boolean onScroll(MotionEvent e1,MotionEvent e2, float distanceX, float distanceY)
	{
		System.out.println("RTR:Scroll");
		uninitialize();
		System.exit(0);
		return true;
	}
	
	//
	//	onShowPress
	//
	@Override
	public void onShowPress(MotionEvent e)
	{
		
	}
	
	//
	//	onSingleTapUp
	//
	@Override
	public boolean onSingleTapUp(MotionEvent e)
	{
		return true;
	}
	//-	abstract method from OnGestureListener so must be implemented
	//---------------------------------------------------------------------------
	
	//---------------------------------------------------------------------------
	//+	OpenGL methods
	
	private void initialize(GL10 gl)
	{
		//---------------------------------------------------------------------------
		//+	Vertex shader 
		shaderObjectVertex = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER);
		
		//	Vertex shader source code.
		final String vertexShaderSourceCode = String.format
		(
		"#version 300 es"+
		"\n"+
		"in vec4 vPosition;"+
		"in vec2 vTexture0_Coord;"+
		"out vec2 out_texture0_coord;"+
		"uniform mat4 u_mvp_matrix;"+
		"void main(void)"+
		"{"+
		"gl_Position = u_mvp_matrix * vPosition;"+
		"out_texture0_coord = vTexture0_Coord;"+
		"}"
		);
		
		//	Provide source code to shader
		GLES30.glShaderSource(shaderObjectVertex,vertexShaderSourceCode);
		
		//	Compile shader and check for errors.
		GLES30.glCompileShader(shaderObjectVertex);
		
		int[] iShaderCompiledStatus = new int[1];
		int[] iInfoLogLength = new int[1];
		String szInfoLog = null;
		
		GLES30.glGetShaderiv(shaderObjectVertex,GLES30.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
		if (GLES30.GL_FALSE == iShaderCompiledStatus[0])
		{
			GLES30.glGetShaderiv(shaderObjectVertex, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetShaderInfoLog(shaderObjectVertex);
				System.out.println("RTR: Vertex shader compilation status Log: "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Vertex shader 
		//---------------------------------------------------------------------------
		
		//---------------------------------------------------------------------------
		//+	Fragment shader 
		shaderObjectFragment = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER);
		
		//	Vertex shader source code.
		final String fragmentShaderSourceCode = String.format
		(
		"#version 300 es"+
		"\n"+
		"precision highp float;"+
		"in vec2 out_texture0_coord;"+
		"uniform highp sampler2D u_texture0_sampler;"+
		"out vec4 FragColor;"+
		"void main(void)"+
		"{"+
		"FragColor = texture(u_texture0_sampler, out_texture0_coord);"+
		"}"
		);
		
		//	Provide source code to shader
		GLES30.glShaderSource(shaderObjectFragment,fragmentShaderSourceCode);
		
		//	Compile shader and check for errors.
		GLES30.glCompileShader(shaderObjectFragment);
		
		iShaderCompiledStatus[0] = 0;
		iInfoLogLength[0] = 0;
		szInfoLog = null;
		
		GLES30.glGetShaderiv(shaderObjectFragment,GLES30.GL_COMPILE_STATUS, iShaderCompiledStatus, 0);
		if (GLES30.GL_FALSE == iShaderCompiledStatus[0])
		{
			GLES30.glGetShaderiv(shaderObjectFragment, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetShaderInfoLog(shaderObjectFragment);
				System.out.println("RTR: Fragment shader compilation status Log: "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Fragment shader 
		//---------------------------------------------------------------------------

		//---------------------------------------------------------------------------
		//+	Shader program
		shaderObjectProgram = GLES30.glCreateProgram();
		
		GLES30.glAttachShader(shaderObjectProgram,shaderObjectVertex);
		
		GLES30.glAttachShader(shaderObjectProgram,shaderObjectFragment);
		
		//	pre-link binding of shader program object with vertex shader attributes.
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_POSITION, "vPosition");
		GLES30.glBindAttribLocation(shaderObjectProgram, GLESMacros.RTR_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");
		
		//	Link the 2 shaders together to shader program
		GLES30.glLinkProgram(shaderObjectProgram);
		int[] iLinkStatus = new int[1];
		iInfoLogLength[0] = 0;
		szInfoLog = null;
		GLES30.glGetProgramiv(shaderObjectProgram, GLES30.GL_LINK_STATUS, iLinkStatus, 0);
		if (GLES30.GL_FALSE == iLinkStatus[0])
		{
			GLES30.glGetProgramiv(shaderObjectProgram, GLES30.GL_INFO_LOG_LENGTH, iInfoLogLength, 0);
			if (iInfoLogLength[0] > 0)
			{
				szInfoLog = GLES30.glGetProgramInfoLog(shaderObjectProgram);
				System.out.println("RTR: Shader program Link log "+szInfoLog);
				uninitialize();
				System.exit(0);
			}
		}
		//-	Shader program 
		//---------------------------------------------------------------------------
		
		//	Get MVP uniform location.
		mvpUniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_mvp_matrix");
		texture0_sampler_uniform = GLES30.glGetUniformLocation(shaderObjectProgram, "u_texture0_sampler");
		
		//
		//	Load textures.
		//
		texture_smiley[0] = loadGLTexture(R.raw.smiley);
		g_iTextureWhite[0] = loadGLWhiteTexture();				

		//	Vertices,colors,shader attributes,VBO,VAO initialization.
		final float arrRectVertices[] = new float[]
		{
			1.0f, 1.0f, 1.0f,	//	Right top
			-1.0f, 1.0f, 1.0f,	// Left Top 
			-1.0f, -1.0f, 1.0f,	//	Left bottom
			1.0f, -1.0f, 1.0f,	// Right bottom
		};
		
	
		final float arrRectTextCoord[] =
		{
			1.0f,1.0f,// Right top
			0.0f,1.0f,// Left top 
			0.0f,0.0f,// Left bottom
			1.0f,0.0f,// Right bottom			
		};
		
		ByteBuffer byBuffer;
		FloatBuffer fVerticesBuffer;

		///////////////////////////////////////////////////////////////////
		//+	VAO Rectangle
		
		GLES30.glGenVertexArrays(1, VAORect, 0);
		GLES30.glBindVertexArray(VAORect[0]);
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Position
		GLES30.glGenBuffers(1, VBOPosition, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOPosition[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(arrRectVertices.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(arrRectVertices);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrRectVertices.length * 4,
							fVerticesBuffer,
							GLES30.GL_STATIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_POSITION, 3, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_POSITION);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Position
		///////////////////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////////////////
		//+	VBO Texture
		GLES30.glGenBuffers(1, VBOTexture, 0);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOTexture[0]);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrRectTextCoord.length * 4,
							null,
							GLES30.GL_DYNAMIC_DRAW
							);
							
		GLES30.glVertexAttribPointer(GLESMacros.RTR_ATTRIBUTE_TEXTURE0, 2, GLES30.GL_FLOAT, false, 0, 0);
		
		GLES30.glEnableVertexAttribArray(GLESMacros.RTR_ATTRIBUTE_TEXTURE0);
		
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		//-	VBO Texture
		///////////////////////////////////////////////////////////////////

		GLES30.glBindVertexArray(0);
		
		//-	VAO Rectangle
		///////////////////////////////////////////////////////////////////
		
		//	Enable depth testing.
		GLES30.glEnable(GLES30.GL_DEPTH_TEST);
		//	dept test to do
		GLES30.glDepthFunc(GLES30.GL_LEQUAL);
		//	We will always cull backfaces for better performance
		//	Disable for animation
		//GLES30.glEnable(GLES30.GL_CULL_FACE);
		
		//	Set the background frame color.
		GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		
		//	Set Projection matrix to identity.
		Matrix.setIdentityM(perspectiveProjectionMatrix, 0);
	}
	
	private int loadGLTexture(int imageFileResourceID)
	{
		BitmapFactory.Options options = new BitmapFactory.Options();
		options.inScaled = false;	//	By default true , we need to make it false.
		
		//	Read in the resource
		Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), imageFileResourceID, options);
		
		int texture[] = new int[1];
		
		//	Create texture object to apply to model.
		GLES30.glGenTextures(1, texture, 0);
		
		//	Indicate that pixel rows are tightly packed.
		//	(defaults to stride of 4 which is kind of only good for RGBA or float data type)
		GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1);
		
		//	bind with the texture.
		GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture[0]);
		
		//	setup filter and wrap modes for this texture object.
		GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
		GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR_MIPMAP_LINEAR);
		
		//	load the bitmap into the bound texture.
		GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0);
		
		//	generate mipmap
		GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D);
		
		return texture[0];
	}
	
	private int loadGLWhiteTexture()
	{
		int texture[] = new int[1];
		ByteBuffer byBuffer;
		
		MakeCheckImage();
		
		Bitmap bitmap = Bitmap.createBitmap(g_arrbyWhiteImage, 0, GLESMacros.IMAGE_WIDTH, GLESMacros.IMAGE_WIDTH, GLESMacros.IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
		
		//	Create texture object to apply to model.
		GLES30.glGenTextures(1, texture, 0);
		
		//	Indicate that pixel rows are tightly packed.
		//	(defaults to stride of 4 which is kind of only good for RGBA or float data type)
		GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1);
		
		//	bind with the texture.
		GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture[0]);
		
		GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_REPEAT);
		GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_REPEAT);

		
		//	setup filter and wrap modes for this texture object.
		GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
		GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR_MIPMAP_LINEAR);
		
		// load the bitmap into the bound texture
		GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0);		
		
		// generate mipmap
		GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D);
		
		return texture[0];
	}
	
	
	void MakeCheckImage()
	{
		int i, j, c;

		for (i = 0; i < GLESMacros.IMAGE_HEIGHT; i++)
		{
			for (j = 0; j < GLESMacros.IMAGE_WIDTH; j++)
			{
				int r = 255;
				int g = 255;
				int b = 255;
				int a = 255;
				
				g_arrbyWhiteImage[i * GLESMacros.IMAGE_WIDTH + j] = (a << 24) | (r << 16) | (g << 8) | b;
				
				// c = 1 * 255;//(((i & 0x8) == 0) ^ ((j & 0x8) == 0)) * 255;
				// g_arrbyWhiteImage[i][j][0] = (byte)c;
				// g_arrbyWhiteImage[i][j][1] = (byte)c;
				// g_arrbyWhiteImage[i][j][2] = (byte)c;
				// g_arrbyWhiteImage[i][j][3] = (byte)c;
			}
		}
	}
	
	private void resize(int width, int height)
	{
		if (height == 0)
			height = 1;
		
		if (width == 0)
			width = 1;

		//	Adjust the viewport based on geometry changes, 
		//	such as screen rotation.
		GLES30.glViewport(0, 0, width, height);
		if (width > height)
		{
			Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)width/(float)height, 0.1f, 100.0f); 
		}
		else
		{
			Matrix.perspectiveM(perspectiveProjectionMatrix, 0, 45.0f, (float)height/(float)width, 0.1f, 100.0f); 
		}
	}
	
	public void draw()
	{
		//	Draw background color.
		GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT | GLES30.GL_DEPTH_BUFFER_BIT);
		
		//	Use shader program.
		GLES30.glUseProgram(shaderObjectProgram);
		
		//	OpenGL-ES drawing.
		float modelViewMatrix[] = new float[16];
		float modelViewProjectionMatrix[] = new float[16];

		////////////////////////////////////////////////////////////////////////////
		//+ Draw Pyramid
		Matrix.setIdentityM(modelViewMatrix, 0);
		Matrix.setIdentityM(modelViewProjectionMatrix, 0);
		
		Matrix.translateM(modelViewMatrix, 0, 0.0f,0.0f,-4.0f);
		
		//	Get modelViewProjectionMatrix
		//	MVPMatrix = ortho * mv matrix;
		Matrix.multiplyMM(modelViewProjectionMatrix, 0, perspectiveProjectionMatrix, 0, modelViewMatrix, 0);
		
		//	Pass mvp(above) matrix to vertex shader in 'u_mvp_matrix' shader variable.
		GLES30.glUniformMatrix4fv(mvpUniform, 1, false, modelViewProjectionMatrix, 0);
		
		//	Bind VAO
		GLES30.glBindVertexArray(VAORect[0]);
		
		float arrRectTextCoord[] = new float[8];
		
		ByteBuffer byBuffer;
		FloatBuffer fVerticesBuffer;
		
		if (1 == g_iKey)
		{
			arrRectTextCoord[0] = 0.5f;
			arrRectTextCoord[1] = 0.5f;
			
			arrRectTextCoord[2] = 0.0f;
			arrRectTextCoord[3] = 0.5f;
			
			arrRectTextCoord[4] = 0.0f;
			arrRectTextCoord[5] = 0.0f;
			
			arrRectTextCoord[6] = 0.5f;
			arrRectTextCoord[7] = 0.0f;
		}
		else if (2 == g_iKey)
		{
			arrRectTextCoord[0] = 1.0f;
			arrRectTextCoord[1] = 1.0f;
			
			arrRectTextCoord[2] = 0.0f;
			arrRectTextCoord[3] = 1.0f;
			
			arrRectTextCoord[4] = 0.0f;
			arrRectTextCoord[5] = 0.0f;
			
			arrRectTextCoord[6] = 1.0f;
			arrRectTextCoord[7] = 0.0f;
		}
		else if (3 == g_iKey)
		{
			arrRectTextCoord[0] = 2.0f;
			arrRectTextCoord[1] = 2.0f;
			
			arrRectTextCoord[2] = 0.0f;
			arrRectTextCoord[3] = 2.0f;
			
			arrRectTextCoord[4] = 0.0f;
			arrRectTextCoord[5] = 0.0f;
			
			arrRectTextCoord[6] = 2.0f;
			arrRectTextCoord[7] = 0.0f;
		}
		else if (4 == g_iKey)
		{
			arrRectTextCoord[0] = 0.5f;
			arrRectTextCoord[1] = 0.5f;
			
			arrRectTextCoord[2] = 0.5f;
			arrRectTextCoord[3] = 0.5f;
			
			arrRectTextCoord[4] = 0.5f;
			arrRectTextCoord[5] = 0.5f;
			
			arrRectTextCoord[6] = 0.5f;
			arrRectTextCoord[7] = 0.5f;
		}
		else
		{
			System.out.println("RTR:Display - Invalid key data");
		}

		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, VBOTexture[0]);
		
		//	Convert array to byte data.
		byBuffer = ByteBuffer.allocateDirect(arrRectTextCoord.length * 4);
		byBuffer.order(ByteOrder.nativeOrder());
		fVerticesBuffer = byBuffer.asFloatBuffer();
		fVerticesBuffer.put(arrRectTextCoord);
		fVerticesBuffer.position(0);
		
		GLES30.glBufferData(
							GLES30.GL_ARRAY_BUFFER,
							arrRectTextCoord.length * 4,
							fVerticesBuffer,
							GLES30.GL_DYNAMIC_DRAW
							);
		GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
		
		//	bind with pyramid texture.
		GLES30.glActiveTexture(GLES30.GL_TEXTURE0);
		if (0 == g_iKey)
		{
			GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, g_iTextureWhite[0]);
		}
		else
		{
			GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture_smiley[0]);
		}
		GLES30.glUniform1i(texture0_sampler_uniform, 0);
		
		GLES30.glDrawArrays(GLES30.GL_TRIANGLE_FAN, 0, 4);
		
		// unbind VAO
		GLES30.glBindVertexArray(0);
		
		//- Draw Rectangle
		////////////////////////////////////////////////////////////////////////////
		
		
		//	un-use program.
		GLES30.glUseProgram(0);
		
		update();
		
		//	render/flush
		requestRender();	//	Same a SwapBuffers()
	}
	
	private void update()
	{
	}
	
	void uninitialize()
	{
		System.out.println("RTR:Uninitialize-->");
		if (VAORect[0] != 0)
		{
			GLES30.glDeleteVertexArrays(1, VAORect, 0);
			VAORect[0] = 0;
		}
		
		if (VBOPosition[0] != 0)
		{
			GLES30.glDeleteBuffers(1, VBOPosition, 0);
			VBOPosition[0] = 0;
		}
		
		if (VBOTexture[0] != 0)
		{
			GLES30.glDeleteBuffers(1, VBOTexture, 0);
			VBOTexture[0] = 0;
		}
		
		if (shaderObjectProgram != 0)
		{
			if (shaderObjectVertex != 0)
			{
				GLES30.glDetachShader(shaderObjectProgram, shaderObjectVertex);
				GLES30.glDeleteShader(shaderObjectVertex);
				shaderObjectVertex = 0;
			}
			if (shaderObjectFragment != 0)
			{
				GLES30.glDetachShader(shaderObjectProgram, shaderObjectFragment);
				GLES30.glDeleteShader(shaderObjectFragment);
				shaderObjectFragment = 0;
			}
		}
		
		if (shaderObjectProgram != 0)
		{
			GLES30.glDeleteProgram(shaderObjectProgram);
			shaderObjectProgram = 0;
		}
		
		//	Delete texture objects
		if (texture_smiley[0] != 0)
		{
			GLES30.glDeleteTextures(1, texture_smiley, 0);
			texture_smiley[0] = 0;
		}
		
		//	Delete texture objects
		if (g_iTextureWhite[0] != 0)
		{
			GLES30.glDeleteTextures(1, g_iTextureWhite, 0);
			g_iTextureWhite[0] = 0;
		}
		
		System.out.println("RTR:Uninitialize<--");
	}
	//-	OpenGL methods
	//---------------------------------------------------------------------------
}
