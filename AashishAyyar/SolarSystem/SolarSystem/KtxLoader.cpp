#include <windows.h>
#include <stdio.h>

#include <gl/glew.h>
#include <gl/GL.h>

typedef struct header 
{
	unsigned char identifier[12];
	unsigned int endianess;
	unsigned int gltype;
	unsigned int gltypesize;
	unsigned int glformat;
	unsigned int glinternalformat;
	unsigned int glbaseinternalformat;
	unsigned int pixelwidth;
	unsigned int pixelheight;
	unsigned int pixeldepth;
	unsigned int arrayelements;
	unsigned int faces;
	unsigned int miplevels;
	unsigned int keypairbytes;
};

union keyvaluepair 
{
	unsigned int size;
	unsigned char rawbytes[4];
};

static const unsigned char identifier[] = 
{
	0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A 
};

unsigned int swap32(const unsigned int u32);
unsigned int calculate_stride(const header &h, unsigned int width, unsigned int pad);
unsigned int calculate_face_size(const header &h);

unsigned int loadktx(const char *filename, unsigned int &tex)
{
	FILE *fp;
	GLuint temp = 0;
	GLuint retval = 0;
	header h;
	GLenum target = GL_NONE;

	size_t data_start, data_end;
	unsigned char *data = NULL;

	fopen_s(&fp,filename, "rb");
	if (!fp)
		return 0;

	if (fread(&h, sizeof(h), 1, fp) != 1)
	{
		fclose(fp);
		return 0;
	}

	if (memcmp(h.identifier, identifier, sizeof(identifier)) != 0)
	{
		fclose(fp);
		return 0;
	}

	if (h.endianess == 0x04030201)
	{
		// Correct endianess no swap needed
	}
	else if (h.endianess == 0x01020304) 
	{
		h.endianess = swap32(h.endianess);
		h.gltype = swap32(h.gltype);
		h.gltypesize = swap32(h.gltypesize);
		h.glformat = swap32(h.glformat);
		h.glinternalformat = swap32(h.glinternalformat);
		h.glbaseinternalformat = swap32(h.glbaseinternalformat);
		h.pixelwidth = swap32(h.pixelwidth);
		h.pixelheight = swap32(h.pixelheight);
		h.pixeldepth = swap32(h.pixeldepth);
		h.arrayelements = swap32(h.arrayelements);
		h.faces = swap32(h.faces);
		h.miplevels = swap32(h.miplevels);
		h.keypairbytes = swap32(h.keypairbytes);
	}
	else 
	{
		fclose(fp);
		return 0;
	}
	
	if (h.pixelheight == 0)
	{
		if (h.arrayelements == 0)
		{
			target = GL_TEXTURE_1D;
		}
		else
		{
			target = GL_TEXTURE_1D_ARRAY;
		}
	}
	else if(h.pixeldepth == 0)
	{
		if (h.arrayelements == 0)
		{
			if (h.faces == 0)
			{
				target = GL_TEXTURE_2D;
			}
			else
			{
				target = GL_TEXTURE_CUBE_MAP;
			}
		}
		else 
		{
			if (h.faces == 0) 
			{
				target = GL_TEXTURE_2D_ARRAY;
			}
			else
			{
				target = GL_TEXTURE_CUBE_MAP_ARRAY;
			}
		}
	}
	else 
	{
		target = GL_TEXTURE_3D;
	}

	if (
		(target == GL_NONE) ||
		(h.pixelwidth == 0) ||
		((h.pixelheight == 0) && (h.pixeldepth != 0))
		)
	{
		fclose(fp);
		return 0;
	}

	temp = tex;
	if (tex == 0)
		glGenTextures(1, &tex);
	
	glBindTexture(target, tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	data_start = ftell(fp) + h.keypairbytes;
	fseek(fp, 0, SEEK_END);
	data_end = ftell(fp);
	fseek(fp, data_start, SEEK_SET);
	
	data = new unsigned char[data_end - data_start];
	memset(data, 0, data_end - data_start);

	size_t size = fread(data, 1, data_end - data_start, fp);
	
	if (h.miplevels == 0)
		h.miplevels = 1;

	switch (target) 
	{
	case GL_TEXTURE_1D:
	{
		glTexStorage1D(GL_TEXTURE_1D, h.miplevels, h.glinternalformat, h.pixelwidth);
		glTexSubImage1D(GL_TEXTURE_1D, 0, 0, h.pixelwidth, h.glformat, h.glinternalformat, data);
	}
	break;
	case GL_TEXTURE_2D:
	{
		if (h.gltype == GL_NONE) 
		{
			glCompressedTexImage2D(GL_TEXTURE_2D, 0, h.glinternalformat, h.pixelwidth, h.pixelheight, 0, 420 * 380 / 2, data);
		}
		else 
		{
			glTexStorage2D(GL_TEXTURE_2D, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight);
			{
				unsigned char *ptr = data;
				unsigned int height = h.pixelheight;
				unsigned int width = h.pixelwidth;

				glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

				for (unsigned int i = 0; i < h.miplevels; i++) 
				{
					glTexSubImage2D(GL_TEXTURE_2D, i, 0, 0, width, height, h.glformat, h.gltype, ptr);
					ptr += height * calculate_stride(h, width, 1);

					height >>= 1;
					width >>= 1;

					if (!height)
						height = 1;
					if (!width)
						width = 1;
				}
			}
		}
	}
	break;
	case GL_TEXTURE_3D:
	{
		glTexStorage3D(GL_TEXTURE_3D, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.pixeldepth);
		glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.pixeldepth, h.glformat, h.gltype, data);
	}
	break;
	case GL_TEXTURE_1D_ARRAY:
	{
		glTexStorage2D(GL_TEXTURE_1D_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.arrayelements);
		glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, 0, h.pixelwidth, h.arrayelements, h.glformat, h.gltype, data);
	}
	break;
	case GL_TEXTURE_2D_ARRAY:
	{
		glTexStorage3D(GL_TEXTURE_2D_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.arrayelements);
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.arrayelements, h.glformat, h.gltype, data);
	}
	break;
	case GL_TEXTURE_CUBE_MAP:
	{
		glTexStorage2D(GL_TEXTURE_CUBE_MAP, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight);
		unsigned int face_size = calculate_face_size(h);
		for (unsigned int i = 0; i < h.faces; i++) 
		{
			glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, h.pixelwidth, h.pixelheight, h.glformat, h.gltype, data + face_size * i);
		}
	}
	break;
	case GL_TEXTURE_CUBE_MAP_ARRAY:
	{
		glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, h.miplevels, h.glinternalformat, h.pixelwidth, h.pixelheight, h.arrayelements);
	
		glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, 0, h.pixelwidth, h.pixelheight, h.faces * h.arrayelements, h.glformat, h.gltype, data);
	}
	break;
	default:
	{
		delete[] data;

		fclose(fp);
		return 0;
	}
	break;
	}

	if (h.miplevels == 1)
		glGenerateMipmap(target);

	retval = tex;

	delete[] data;
	fclose(fp);

	return TRUE;
}

unsigned int swap32(const unsigned int u32) 
{
	union 
	{
		unsigned int u32;
		unsigned int u8[4];
	} a, b;
	
	a.u32 = u32;
	b.u8[0] = a.u8[3];
	b.u8[1] = a.u8[2];
	b.u8[2] = a.u8[1];
	b.u8[3] = a.u8[0];

	return b.u32;
}

unsigned int calculate_stride(const header &h, unsigned int width, unsigned int pad = 4) 
{
	unsigned int channels = 0;

	switch (h.glbaseinternalformat) 
	{
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

	unsigned int stride = h.gltypesize * channels * width;
	
	stride = (stride + (pad - 1)) & ~(pad - 1);
	
	return stride;
}

unsigned int calculate_face_size(const header &h)
{
	unsigned int stride = calculate_stride(h, h.pixelwidth);
	
	return stride * h.pixelheight;
}


