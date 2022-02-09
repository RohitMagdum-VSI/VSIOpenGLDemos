#include<Windows.h>
#include<stdio.h>
#include "Tga_Loader.h"

Tga_Loader::Tga_Loader()
{

}

unsigned char * Tga_Loader::LoadTarga(char *filename, int &height, int &width)
{
    int error, bitsperpixel, imageSize, index, i, j, k;
    FILE *pFile;
    unsigned int count;
    TargaHeader tgaFileHeader;
    unsigned char *tgaImage;

    error = fopen_s(&pFile, filename, "rb");
    if(error != 0)
    {
        return false;
    }

    count = (unsigned int)fread(&tgaFileHeader, sizeof(TargaHeader), 1, pFile);
    if(count != 1)
    {
        return false;
    }

    height = (int)tgaFileHeader.height;
    width = (int)tgaFileHeader.width;
    bitsperpixel = (int)tgaFileHeader.bpp;

    if(bitsperpixel != 32)
    {
        return false;
    }

    imageSize = width * height * 4;

    tgaImage = new unsigned char[imageSize];
    if(!tgaImage)
    {
        return false;
    }

    count = (unsigned int)fread(tgaImage, 1, imageSize, pFile);
    if(count != imageSize)
    {
        return false;
    }

    error = fclose(pFile);
    if(error != 0)
    {
        return false;
    }

	unsigned char *tgaData = new unsigned char[imageSize];
    if(!tgaData)
    {
        return false;
    }

    // Initialize the index into the targa destination data array.
    index = 0;

    // Initialize the index into the targa image data.
    k = (width * height *4) - (width * 4);


    // Now copy the targa image data into the targa destination array in the correct order since the targa format is stored upside down.
    for(j=0; j<height; j++)
    {
        for(i=0; i<width; i++)
        {
            tgaData[index+0] = tgaImage[k+2];//Red
            tgaData[index+1] = tgaImage[k+1];//Green
            tgaData[index+2] = tgaImage[k+0];//Blue
            tgaData[index+3] = tgaImage[k+3];//Alpha

            
            // Increment the indexes into the targa data.
            k += 4;
            index += 4;
        }

		// Set the targa image data index back to the preceding row at the beginning of the column since its reading it in upside down.
        k -= (width * 8);
    }
    

    delete tgaImage;
    tgaImage = NULL;

    return tgaData;
}

Tga_Loader::~Tga_Loader()
{
    
}