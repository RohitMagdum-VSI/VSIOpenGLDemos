#include<Windows.h>

class Tga_Loader
{
private:
	struct TargaHeader
	{
		unsigned char data1[12];
		unsigned short width;
		unsigned short height;
		unsigned char bpp;
		unsigned char data2;
	};
public:
    Tga_Loader();
    unsigned char * LoadTarga(char*, int&, int&);
    ~Tga_Loader();
};