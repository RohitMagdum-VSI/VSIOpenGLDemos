typedef struct _VBM_HEADER{
    unsigned int magic;
    unsigned int size;
    char name[64];
    unsigned int num_attribs;
    unsigned int num_frames;
    unsigned int num_vertices;
    unsigned int num_indices;
    unsigned int index_type;
}VBM_HEADER, *P_VBM_HEADER;


typedef struct _VBM_ATTRIB_HEADER{
    char name[64];
    unsigned int type;
    unsigned int components;
    unsigned int flags;
}VBM_ATTRIB_HEADER, *P_VBM_ATTRIB_HEADER;


typedef struct _VBM_FRAME_HEADER{
    unsigned int first;
    unsigned int count;
    unsigned int flags;
}VBM_FRAME_HEADER, *P_VBM_FRAME_HEADER;

void vbmLoader(char*name, int iVertexIndex, int iNormalIndex, int iTexcoord0Index);
