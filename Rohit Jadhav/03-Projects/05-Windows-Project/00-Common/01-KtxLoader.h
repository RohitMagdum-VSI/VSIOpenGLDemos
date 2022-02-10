//For KTX File
struct Header{
	unsigned char identifier[12];
	unsigned int endianness;
	unsigned int glType;
	unsigned int glTypeSize;
	unsigned int glFormat;
	unsigned int glInternalFormat;
	unsigned int glBaseInternalFormat;
	unsigned int pixelWidth;
	unsigned int pixelHeight;
	unsigned int pixelDepth;
	unsigned int arrayElements;
	unsigned int faces;
	unsigned int mipLevels;
	unsigned int keyPairBytes;
};


void uninitialize(void);


GLuint LoadTexture(TCHAR imageResourceId[], GLint flag){

	HBITMAP hBitmap = NULL;
	BITMAP bmp;
	GLuint texture = 0;

	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);

	if (hBitmap) {

		GetObject(hBitmap, sizeof(BITMAP), &bmp);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		
		if(flag == 1){
			
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		}

		glTexImage2D(GL_TEXTURE_2D,
			0,
			GL_RGB,
			bmp.bmWidth, bmp.bmHeight, 0,
			GL_BGR_EXT,
			GL_UNSIGNED_BYTE,
			bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);
		glBindTexture(GL_TEXTURE_2D, 0);
		
	}
	return(texture);
}



int LoadKTXTexture(const char* filename, GLuint *texture){

	
	unsigned int swap32(unsigned int);
	unsigned int calculate_stride(struct Header&, unsigned int, unsigned int);
	unsigned int calculate_face_size(struct Header&);

	FILE *pFile = NULL;
	GLuint iTemp = 0;
	int iRetval = 0;

	struct Header h;
	size_t data_start, data_end, data_total_size;

	unsigned char *data = NULL;
	GLenum target = GL_NONE;


	unsigned char identifier[] = {
		0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 
		0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
	};


	pFile = fopen(filename, "rb");

	if(pFile == NULL){
		fprintf(gpFile, "LoadKTXTexture : %s file open failed\n", filename);
		uninitialize();
		DestroyWindow(ghwnd);
	}
	else
		fprintf(gpFile, "LoadKTXTexture : %s file open\n", filename);


	if(fread(&h, sizeof(struct Header), 1, pFile) != 1){
		goto fail_read;
	}


	if(memcmp(h.identifier, identifier, sizeof(identifier)) != 0){
		goto fail_header;
	}



	fprintf(gpFile, "LoadKTXTexture : Endianness : 0x%x\n", h.endianness);


	if(h.endianness == 0x04030201){
		// No Swap Needed
	}
	else if(h.endianness == 0x01020304){

		h.endianness = swap32(h.endianness);
		h.glType = swap32(h.glType);
		h.glTypeSize = swap32(h.glTypeSize);
		h.glFormat = swap32(h.glFormat);
		h.glInternalFormat = swap32(h.glInternalFormat);
		h.glBaseInternalFormat = swap32(h.glBaseInternalFormat);
		h.pixelWidth = swap32(h.pixelWidth);
		h.pixelHeight = swap32(h.pixelHeight);
		h.pixelDepth = swap32(h.pixelDepth);
		h.arrayElements = swap32(h.arrayElements);
		h.faces = swap32(h.faces);
		h.mipLevels = swap32(h.mipLevels);
		h.keyPairBytes = swap32(h.keyPairBytes);
	}
	else
		goto fail_header;




	if(h.pixelHeight == 0){
		
		// *** 1D Texture ***

		if(h.arrayElements == 0){
			target = GL_TEXTURE_1D;
		}
		else
			target = GL_TEXTURE_1D_ARRAY;

	}
	else if(h.pixelDepth == 0){

		// *** 2D Texture ***

		if(h.arrayElements == 0){
			
			if(h.faces == 0)
				target = GL_TEXTURE_2D;
			else
				target = GL_TEXTURE_CUBE_MAP;
		
		}
		else{
			
			if(h.faces == 0)
				target = GL_TEXTURE_2D_ARRAY;
			else
				target = GL_TEXTURE_CUBE_MAP_ARRAY;
		
		}
	}
	else{

		// *** 3D Texture ***

		target = GL_TEXTURE_3D;
	}




	// ***** Check for Insanity *****
	if(target == GL_NONE || (h.pixelWidth == 0) || (h.pixelHeight == 0 && h.pixelDepth != 0))
		goto fail_header;	



	if(*texture == 0){
		glGenTextures(1, texture);
	}


	glBindTexture(target, *texture);

	data_start = ftell(pFile) + h.keyPairBytes;
	fseek(pFile, 0, SEEK_END);
	data_end = ftell(pFile);

	data_total_size = data_end - data_start;


	fseek(pFile, data_start, SEEK_SET);

	data = (unsigned char*)malloc(sizeof(unsigned char) * data_total_size);
	if(data == NULL){
		fprintf(gpFile, "LoadKTXTexture : Memory Allocation Failed\n");
		uninitialize();
		DestroyWindow(ghwnd);
	}

	
	memset(data, 0, data_total_size);

	fread(data, 1, data_total_size, pFile);	


	if(h.mipLevels == 0)
		h.mipLevels = 1;


	switch(target){

		case GL_TEXTURE_1D:
			glTexStorage1D(GL_TEXTURE_1D, h.mipLevels, h.glInternalFormat, h.pixelWidth);
			glTexSubImage1D(GL_TEXTURE_1D, 0, 0, h.pixelWidth, h.glFormat, h.glInternalFormat, data);
			break;


		case GL_TEXTURE_2D:

			if(h.glType == GL_NONE){
				glCompressedTexImage2D(GL_TEXTURE_2D, 0, h.glInternalFormat, h.pixelWidth, h.pixelHeight, 0, 420*380 / 2, data);
			}
			else{

				glTexStorage2D(GL_TEXTURE_2D, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight);

				{
					unsigned char *ptr = data;
					unsigned int height = h.pixelHeight;
					unsigned int width = h.pixelWidth;

					glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
					for(unsigned int i = 0; i < h.mipLevels; i++){

						glTexSubImage2D(GL_TEXTURE_2D, i, 0, 0, width, height, h.glFormat, h.glType, ptr);

						ptr = ptr + height * calculate_stride(h, width, 1);

						height >>= 1;
						width >>= 1;

						if(!height)
							height = 1;
						if(!width)
							width = 1;
					}
				}

			}

			break;


		case GL_TEXTURE_3D:

			glTexStorage3D(GL_TEXTURE_3D, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight, h.pixelDepth);
			glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.pixelDepth, h.glFormat, h.glType, data);

			break;

		case GL_TEXTURE_1D_ARRAY:
			glTexStorage2D(GL_TEXTURE_1D_ARRAY, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.arrayElements);
			glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, 0, h.pixelWidth, h.arrayElements, h.glFormat, h.glType, data);
			break;


		case GL_TEXTURE_2D_ARRAY:
			glTexStorage3D(GL_TEXTURE_2D_ARRAY, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight, h.arrayElements);
			glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.arrayElements, h.glFormat, h.glType, data);
			break;

		case GL_TEXTURE_CUBE_MAP:
			glTexStorage2D(GL_TEXTURE_CUBE_MAP, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight);

			{
				unsigned int face_size = calculate_face_size(h);
				for(unsigned int i = 0; i < h.faces; i++){
					glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.glFormat, h.glType, data + face_size * i);
				}
			}
			break;

		case GL_TEXTURE_CUBE_MAP_ARRAY:

			glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, h.mipLevels, h.glInternalFormat, h.pixelWidth, h.pixelHeight, h.arrayElements);
			glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, h.pixelWidth, h.pixelHeight, h.faces * h.arrayElements, h.glFormat, h.glType, data);

			break;

		default:
			goto fail_target;
	}


	if(h.mipLevels == 1)
		glGenerateMipmap(target);

	iRetval = (int)(*texture);


fail_target:

	if(data){
		free(data);
		data = NULL;
	}


fail_header:
fail_read:

	if(pFile){
		fclose(pFile);
		pFile = NULL;
	}

	return(iRetval);
}


unsigned int swap32(unsigned int u32){

	union{
		unsigned int u32;
		unsigned char u8[4];
	}a, b;

	a.u32 = u32;

	b.u8[0] = a.u8[3];
	b.u8[1] = a.u8[2];
	b.u8[2] = a.u8[1];
	b.u8[3] = a.u8[0];

	return(b.u32);
}

unsigned int calculate_stride(struct Header &h, unsigned int width, unsigned int pad = 4){

	unsigned int channels = 0;

	switch(h.glInternalFormat){

		case GL_RED:
			channels = 1;
			break;

		case GL_RG:
			channels = 2;
			break;

		case GL_BGR:
		case GL_RGB:
			channels = 3;
			break;

		case GL_BGRA:
		case GL_RGBA:
			channels = 4;
			break;
	}

	unsigned int  stride = h.glTypeSize * channels * width;

	stride = (stride + (pad - 1)) & ~(pad - 1);

	return(stride);
}


unsigned int calculate_face_size(struct Header &h){

	unsigned int stride = calculate_stride(h, h.pixelWidth, 4);

	return(stride * h.pixelHeight);
}

