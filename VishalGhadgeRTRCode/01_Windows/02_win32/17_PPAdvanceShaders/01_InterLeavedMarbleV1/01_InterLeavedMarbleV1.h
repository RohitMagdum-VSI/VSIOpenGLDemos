#define IDBITMAP_STONE 101

#define INTERLEAVED_ARRAY	0
#define INTERLEAVED_STRUCT	1 

#pragma pack(push, 1)
typedef struct _CUBE_VERTEX
{
	float position[3];
	float color[3];
	float normal[3];
	float texture[2];

} CUBE_VERTEX;
#pragma pack(pop)
