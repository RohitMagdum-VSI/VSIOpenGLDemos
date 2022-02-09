#include <Windows.h>
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include "../common/vmath.h"
#include "../common/Common.h"
#include "../common/Shapes.h"

#include "ShaderUtils.h"
#include "Metaballs.h"

extern FILE* gpFile;

//
// Globals
//
const GLuint guiTesselationLevel = 64;
GLfloat gfIsoSurfaceLevel = 12.0f;
unsigned int window_width = 256;
unsigned int window_height = 256;

const GLuint guiSamplesPerAxis = guiTesselationLevel;
const GLuint guiSamplesIn3DSpace = guiSamplesPerAxis * guiSamplesPerAxis * guiSamplesPerAxis;
const GLuint guiCellsPerAxis = guiSamplesPerAxis - 1;
const GLuint guiCellsIn3DSpace = guiCellsPerAxis * guiCellsPerAxis * guiCellsPerAxis;
const GLuint guiVerticesPerTriangle = 3;
const GLuint guiTrianglesPerCell = 5;
const GLuint guiMCVerticesPerCell = guiVerticesPerTriangle * guiTrianglesPerCell;
const GLuint guiMCCellsTypesCount = 256;

const int iNSpheres = 3;
const int iNSpherePositionComponents = 4;

//
// Stage 1
//
GLuint guiSphereUpdaterVSO = 0;
GLuint guiSphereUpdaterFSO = 0;
GLuint guiSphereUpdaterSPO = 0;
GLuint guiSphereUpdaterSpherePositionsBufferObjectID = 0;
GLuint guiSphereUpdaterTransformFeedbackObjectID = 0;
const GLchar *gpchSphereUpdaterUniformTimeName = "time";
GLuint guiSphereUpdaterUniformTimeID = 0;
const GLchar *gpchSpherePositionVaryingName = "sphere_position";

//
// Stage 2
//
GLuint guiScalarFieldVSO = 0;
GLuint guiScalarFieldFSO = 0;
GLuint guiScalarFieldSPO = 0;
GLuint guiScalarFieldBufferObjectID = 0;
GLuint guiScalarFieldTransformFeedbackObjectID = 0;
const GLchar *gpchScalarFieldUniformSamplesPerAxisName = "samples_per_axis";
GLuint guiScalarFieldUniformSamplesPerAxisNameID = 0;
const GLchar *gpchScalarFieldUniformSpheresName = "spheres_uniform_block";
GLuint guiScalarFieldUniformSpheresID = 0;
const GLchar *gpchScalarFieldValueVaryingName = "scalar_field_value";
GLuint guiScalarFieldTextureObjectID = 0;

//
// Stage 3
//
GLuint guiMarchingCubesCellsVSO = 0;
GLuint guiMarchingCubesCellsFSO = 0;
GLuint guiMarchingCubesCellsSPO = 0;
const GLchar *gpchMarchingCubesCellsUniformCellsPerAxisName = "cells_per_axis";
GLuint guiMarchingCubesCellsUniformCellsPerAxisID = 0;
const GLchar *gpchMarchingCubesCellsUniformIsoLevelName = "iso_level";
GLuint guiMarchingCubesCellsUniformIsoLevelID = 0;
const GLchar *gpchMarchingCubesCellsUniformScalarFieldSamplerName = "scalar_field";
GLuint guiMarchingCubesCellsUniformScalarFieldSamplerID = 0;
const GLchar *gpchMarchingCubesCellsVaryingName = "cell_type_index";
GLuint guiMarchingCubesCellsTransformFeedbackObjectID = 0;
GLuint guiMarchingCubesCellsTypesBufferID = 0;
GLuint guiMarchingCubesCellsTypesTextureObjectID = 0;

//
// Stage 4
//
GLuint guiMarchingCubesTrianglesVSO = 0;
GLuint guiMarchingCubesTrianglesFSO = 0;
GLuint guiMarchingCubesTrianglesSPO = 0;
const GLchar *gpchMarchingCubesTriangleUniformSamplesPerAxisName = "samples_per_axis";
GLuint guiMarchingCubesTriangleUniformSamplesPerAxisID = 0;
const GLchar* gpchMarchingCubesTriangleUniformIsoLevelName = "iso_level";
GLuint guiMarchingCubesTriangleUniformIsoLevelID = 0;
const GLchar* gpchMarchingCubesTriangleUniformTimeName = "time";
GLuint guiMarchingCubesTriangleUniformTimeID = 0;
const GLchar* gpchMarchingCubesTriangleUniformMVPName = "mvp";
GLuint guiMarchingCubesTriangleUniformMVPID = 0;
const GLchar* gpchMarchingCubesTriangleUniformCellTypesSamplerName = "cell_types";
GLuint guiMarchingCubesTriangleUniformCellTypesSamplerID = 0;
const GLchar* gpchMarchingCubesTriangleUniformScalarFieldSamplerName = "scalar_field";
GLuint guiMarchingCubesTriangleUniformScalarFieldSamplerID = 0;
const GLchar* gpchMarchingCubesTriangleUniformSpherePositionsName = "sphere_positions_uniform_block";
GLuint guiMarchingCubesTriangleUniformSpherePositionsID = 0;
const GLchar* gpchMarchingCubesTriangleUniformTriTableSamplerName = "tri_table";
GLuint guiMarchingCubesTriangleUniformTriTableSampleID = 0;
GLuint guiMarchingCubesTriangleUniformLookupTableTextureID = 0;
GLuint guiMarchingCubesTriangleUniformVAOID = 0;

const GLint tri_table[guiMCCellsTypesCount * guiMCVerticesPerCell] =
{
  -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
   0,  8,  3,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
   0,  1,  9,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
   1,  8,  3,     9,  8,  1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
   /* [tri_table chosen part for documentation] */
	  1,  2, 10,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  3,     1,  2, 10,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  9,  2, 10,     0,  2,  9,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  2,  8,  3,     2, 10,  8,    10,  9,  8,    -1, -1, -1,    -1, -1, -1,
	  3, 11,  2,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0, 11,  2,     8, 11,  0,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  9,  0,     2,  3, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1, 11,  2,     1,  9, 11,     9,  8, 11,    -1, -1, -1,    -1, -1, -1,
	  3, 10,  1,    11, 10,  3,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0, 10,  1,     0,  8, 10,     8, 11, 10,    -1, -1, -1,    -1, -1, -1,
	  3,  9,  0,     3, 11,  9,    11, 10,  9,    -1, -1, -1,    -1, -1, -1,
	  9,  8, 10,    10,  8, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  7,  8,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  3,  0,     7,  3,  4,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  1,  9,     8,  4,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  1,  9,     4,  7,  1,     7,  3,  1,    -1, -1, -1,    -1, -1, -1,
	  1,  2, 10,     8,  4,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  3,  4,  7,     3,  0,  4,     1,  2, 10,    -1, -1, -1,    -1, -1, -1,
	  9,  2, 10,     9,  0,  2,     8,  4,  7,    -1, -1, -1,    -1, -1, -1,
	  2, 10,  9,     2,  9,  7,     2,  7,  3,     7,  9,  4,    -1, -1, -1,
	  8,  4,  7,     3, 11,  2,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 11,  4,  7,    11,  2,  4,     2,  0,  4,    -1, -1, -1,    -1, -1, -1,
	  9,  0,  1,     8,  4,  7,     2,  3, 11,    -1, -1, -1,    -1, -1, -1,
	  4,  7, 11,     9,  4, 11,     9, 11,  2,     9,  2,  1,    -1, -1, -1,
	  3, 10,  1,     3, 11, 10,     7,  8,  4,    -1, -1, -1,    -1, -1, -1,
	  1, 11, 10,     1,  4, 11,     1,  0,  4,     7, 11,  4,    -1, -1, -1,
	  4,  7,  8,     9,  0, 11,     9, 11, 10,    11,  0,  3,    -1, -1, -1,
	  4,  7, 11,     4, 11,  9,     9, 11, 10,    -1, -1, -1,    -1, -1, -1,
	  9,  5,  4,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  9,  5,  4,     0,  8,  3,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  5,  4,     1,  5,  0,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  8,  5,  4,     8,  3,  5,     3,  1,  5,    -1, -1, -1,    -1, -1, -1,
	  1,  2, 10,     9,  5,  4,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  3,  0,  8,     1,  2, 10,     4,  9,  5,    -1, -1, -1,    -1, -1, -1,
	  5,  2, 10,     5,  4,  2,     4,  0,  2,    -1, -1, -1,    -1, -1, -1,
	  2, 10,  5,     3,  2,  5,     3,  5,  4,     3,  4,  8,    -1, -1, -1,
	  9,  5,  4,     2,  3, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0, 11,  2,     0,  8, 11,     4,  9,  5,    -1, -1, -1,    -1, -1, -1,
	  0,  5,  4,     0,  1,  5,     2,  3, 11,    -1, -1, -1,    -1, -1, -1,
	  2,  1,  5,     2,  5,  8,     2,  8, 11,     4,  8,  5,    -1, -1, -1,
	 10,  3, 11,    10,  1,  3,     9,  5,  4,    -1, -1, -1,    -1, -1, -1,
	  4,  9,  5,     0,  8,  1,     8, 10,  1,     8, 11, 10,    -1, -1, -1,
	  5,  4,  0,     5,  0, 11,     5, 11, 10,    11,  0,  3,    -1, -1, -1,
	  5,  4,  8,     5,  8, 10,    10,  8, 11,    -1, -1, -1,    -1, -1, -1,
	  9,  7,  8,     5,  7,  9,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  9,  3,  0,     9,  5,  3,     5,  7,  3,    -1, -1, -1,    -1, -1, -1,
	  0,  7,  8,     0,  1,  7,     1,  5,  7,    -1, -1, -1,    -1, -1, -1,
	  1,  5,  3,     3,  5,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  9,  7,  8,     9,  5,  7,    10,  1,  2,    -1, -1, -1,    -1, -1, -1,
	 10,  1,  2,     9,  5,  0,     5,  3,  0,     5,  7,  3,    -1, -1, -1,
	  8,  0,  2,     8,  2,  5,     8,  5,  7,    10,  5,  2,    -1, -1, -1,
	  2, 10,  5,     2,  5,  3,     3,  5,  7,    -1, -1, -1,    -1, -1, -1,
	  7,  9,  5,     7,  8,  9,     3, 11,  2,    -1, -1, -1,    -1, -1, -1,
	  9,  5,  7,     9,  7,  2,     9,  2,  0,     2,  7, 11,    -1, -1, -1,
	  2,  3, 11,     0,  1,  8,     1,  7,  8,     1,  5,  7,    -1, -1, -1,
	 11,  2,  1,    11,  1,  7,     7,  1,  5,    -1, -1, -1,    -1, -1, -1,
	  9,  5,  8,     8,  5,  7,    10,  1,  3,    10,  3, 11,    -1, -1, -1,
	  5,  7,  0,     5,  0,  9,     7, 11,  0,     1,  0, 10,    11, 10,  0,
	 11, 10,  0,    11,  0,  3,    10,  5,  0,     8,  0,  7,     5,  7,  0,
	 11, 10,  5,     7, 11,  5,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 10,  6,  5,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  3,     5, 10,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  9,  0,  1,     5, 10,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  8,  3,     1,  9,  8,     5, 10,  6,    -1, -1, -1,    -1, -1, -1,
	  1,  6,  5,     2,  6,  1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  6,  5,     1,  2,  6,     3,  0,  8,    -1, -1, -1,    -1, -1, -1,
	  9,  6,  5,     9,  0,  6,     0,  2,  6,    -1, -1, -1,    -1, -1, -1,
	  5,  9,  8,     5,  8,  2,     5,  2,  6,     3,  2,  8,    -1, -1, -1,
	  2,  3, 11,    10,  6,  5,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 11,  0,  8,    11,  2,  0,    10,  6,  5,    -1, -1, -1,    -1, -1, -1,
	  0,  1,  9,     2,  3, 11,     5, 10,  6,    -1, -1, -1,    -1, -1, -1,
	  5, 10,  6,     1,  9,  2,     9, 11,  2,     9,  8, 11,    -1, -1, -1,
	  6,  3, 11,     6,  5,  3,     5,  1,  3,    -1, -1, -1,    -1, -1, -1,
	  0,  8, 11,     0, 11,  5,     0,  5,  1,     5, 11,  6,    -1, -1, -1,
	  3, 11,  6,     0,  3,  6,     0,  6,  5,     0,  5,  9,    -1, -1, -1,
	  6,  5,  9,     6,  9, 11,    11,  9,  8,    -1, -1, -1,    -1, -1, -1,
	  5, 10,  6,     4,  7,  8,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  3,  0,     4,  7,  3,     6,  5, 10,    -1, -1, -1,    -1, -1, -1,
	  1,  9,  0,     5, 10,  6,     8,  4,  7,    -1, -1, -1,    -1, -1, -1,
	 10,  6,  5,     1,  9,  7,     1,  7,  3,     7,  9,  4,    -1, -1, -1,
	  6,  1,  2,     6,  5,  1,     4,  7,  8,    -1, -1, -1,    -1, -1, -1,
	  1,  2,  5,     5,  2,  6,     3,  0,  4,     3,  4,  7,    -1, -1, -1,
	  8,  4,  7,     9,  0,  5,     0,  6,  5,     0,  2,  6,    -1, -1, -1,
	  7,  3,  9,     7,  9,  4,     3,  2,  9,     5,  9,  6,     2,  6,  9,
	  3, 11,  2,     7,  8,  4,    10,  6,  5,    -1, -1, -1,    -1, -1, -1,
	  5, 10,  6,     4,  7,  2,     4,  2,  0,     2,  7, 11,    -1, -1, -1,
	  0,  1,  9,     4,  7,  8,     2,  3, 11,     5, 10,  6,    -1, -1, -1,
	  9,  2,  1,     9, 11,  2,     9,  4, 11,     7, 11,  4,     5, 10,  6,
	  8,  4,  7,     3, 11,  5,     3,  5,  1,     5, 11,  6,    -1, -1, -1,
	  5,  1, 11,     5, 11,  6,     1,  0, 11,     7, 11,  4,     0,  4, 11,
	  0,  5,  9,     0,  6,  5,     0,  3,  6,    11,  6,  3,     8,  4,  7,
	  6,  5,  9,     6,  9, 11,     4,  7,  9,     7, 11,  9,    -1, -1, -1,
	 10,  4,  9,     6,  4, 10,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4, 10,  6,     4,  9, 10,     0,  8,  3,    -1, -1, -1,    -1, -1, -1,
	 10,  0,  1,    10,  6,  0,     6,  4,  0,    -1, -1, -1,    -1, -1, -1,
	  8,  3,  1,     8,  1,  6,     8,  6,  4,     6,  1, 10,    -1, -1, -1,
	  1,  4,  9,     1,  2,  4,     2,  6,  4,    -1, -1, -1,    -1, -1, -1,
	  3,  0,  8,     1,  2,  9,     2,  4,  9,     2,  6,  4,    -1, -1, -1,
	  0,  2,  4,     4,  2,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  8,  3,  2,     8,  2,  4,     4,  2,  6,    -1, -1, -1,    -1, -1, -1,
	 10,  4,  9,    10,  6,  4,    11,  2,  3,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  2,     2,  8, 11,     4,  9, 10,     4, 10,  6,    -1, -1, -1,
	  3, 11,  2,     0,  1,  6,     0,  6,  4,     6,  1, 10,    -1, -1, -1,
	  6,  4,  1,     6,  1, 10,     4,  8,  1,     2,  1, 11,     8, 11,  1,
	  9,  6,  4,     9,  3,  6,     9,  1,  3,    11,  6,  3,    -1, -1, -1,
	  8, 11,  1,     8,  1,  0,    11,  6,  1,     9,  1,  4,     6,  4,  1,
	  3, 11,  6,     3,  6,  0,     0,  6,  4,    -1, -1, -1,    -1, -1, -1,
	  6,  4,  8,    11,  6,  8,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  7, 10,  6,     7,  8, 10,     8,  9, 10,    -1, -1, -1,    -1, -1, -1,
	  0,  7,  3,     0, 10,  7,     0,  9, 10,     6,  7, 10,    -1, -1, -1,
	 10,  6,  7,     1, 10,  7,     1,  7,  8,     1,  8,  0,    -1, -1, -1,
	 10,  6,  7,    10,  7,  1,     1,  7,  3,    -1, -1, -1,    -1, -1, -1,
	  1,  2,  6,     1,  6,  8,     1,  8,  9,     8,  6,  7,    -1, -1, -1,
	  2,  6,  9,     2,  9,  1,     6,  7,  9,     0,  9,  3,     7,  3,  9,
	  7,  8,  0,     7,  0,  6,     6,  0,  2,    -1, -1, -1,    -1, -1, -1,
	  7,  3,  2,     6,  7,  2,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  2,  3, 11,    10,  6,  8,    10,  8,  9,     8,  6,  7,    -1, -1, -1,
	  2,  0,  7,     2,  7, 11,     0,  9,  7,     6,  7, 10,     9, 10,  7,
	  1,  8,  0,     1,  7,  8,     1, 10,  7,     6,  7, 10,     2,  3, 11,
	 11,  2,  1,    11,  1,  7,    10,  6,  1,     6,  7,  1,    -1, -1, -1,
	  8,  9,  6,     8,  6,  7,     9,  1,  6,    11,  6,  3,     1,  3,  6,
	  0,  9,  1,    11,  6,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  7,  8,  0,     7,  0,  6,     3, 11,  0,    11,  6,  0,    -1, -1, -1,
	  7, 11,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  7,  6, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  3,  0,  8,    11,  7,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  1,  9,    11,  7,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  8,  1,  9,     8,  3,  1,    11,  7,  6,    -1, -1, -1,    -1, -1, -1,
	 10,  1,  2,     6, 11,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  2, 10,     3,  0,  8,     6, 11,  7,    -1, -1, -1,    -1, -1, -1,
	  2,  9,  0,     2, 10,  9,     6, 11,  7,    -1, -1, -1,    -1, -1, -1,
	  6, 11,  7,     2, 10,  3,    10,  8,  3,    10,  9,  8,    -1, -1, -1,
	  7,  2,  3,     6,  2,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  7,  0,  8,     7,  6,  0,     6,  2,  0,    -1, -1, -1,    -1, -1, -1,
	  2,  7,  6,     2,  3,  7,     0,  1,  9,    -1, -1, -1,    -1, -1, -1,
	  1,  6,  2,     1,  8,  6,     1,  9,  8,     8,  7,  6,    -1, -1, -1,
	 10,  7,  6,    10,  1,  7,     1,  3,  7,    -1, -1, -1,    -1, -1, -1,
	 10,  7,  6,     1,  7, 10,     1,  8,  7,     1,  0,  8,    -1, -1, -1,
	  0,  3,  7,     0,  7, 10,     0, 10,  9,     6, 10,  7,    -1, -1, -1,
	  7,  6, 10,     7, 10,  8,     8, 10,  9,    -1, -1, -1,    -1, -1, -1,
	  6,  8,  4,    11,  8,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  3,  6, 11,     3,  0,  6,     0,  4,  6,    -1, -1, -1,    -1, -1, -1,
	  8,  6, 11,     8,  4,  6,     9,  0,  1,    -1, -1, -1,    -1, -1, -1,
	  9,  4,  6,     9,  6,  3,     9,  3,  1,    11,  3,  6,    -1, -1, -1,
	  6,  8,  4,     6, 11,  8,     2, 10,  1,    -1, -1, -1,    -1, -1, -1,
	  1,  2, 10,     3,  0, 11,     0,  6, 11,     0,  4,  6,    -1, -1, -1,
	  4, 11,  8,     4,  6, 11,     0,  2,  9,     2, 10,  9,    -1, -1, -1,
	 10,  9,  3,    10,  3,  2,     9,  4,  3,    11,  3,  6,     4,  6,  3,
	  8,  2,  3,     8,  4,  2,     4,  6,  2,    -1, -1, -1,    -1, -1, -1,
	  0,  4,  2,     4,  6,  2,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  9,  0,     2,  3,  4,     2,  4,  6,     4,  3,  8,    -1, -1, -1,
	  1,  9,  4,     1,  4,  2,     2,  4,  6,    -1, -1, -1,    -1, -1, -1,
	  8,  1,  3,     8,  6,  1,     8,  4,  6,     6, 10,  1,    -1, -1, -1,
	 10,  1,  0,    10,  0,  6,     6,  0,  4,    -1, -1, -1,    -1, -1, -1,
	  4,  6,  3,     4,  3,  8,     6, 10,  3,     0,  3,  9,    10,  9,  3,
	 10,  9,  4,     6, 10,  4,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  9,  5,     7,  6, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  3,     4,  9,  5,    11,  7,  6,    -1, -1, -1,    -1, -1, -1,
	  5,  0,  1,     5,  4,  0,     7,  6, 11,    -1, -1, -1,    -1, -1, -1,
	 11,  7,  6,     8,  3,  4,     3,  5,  4,     3,  1,  5,    -1, -1, -1,
	  9,  5,  4,    10,  1,  2,     7,  6, 11,    -1, -1, -1,    -1, -1, -1,
	  6, 11,  7,     1,  2, 10,     0,  8,  3,     4,  9,  5,    -1, -1, -1,
	  7,  6, 11,     5,  4, 10,     4,  2, 10,     4,  0,  2,    -1, -1, -1,
	  3,  4,  8,     3,  5,  4,     3,  2,  5,    10,  5,  2,    11,  7,  6,
	  7,  2,  3,     7,  6,  2,     5,  4,  9,    -1, -1, -1,    -1, -1, -1,
	  9,  5,  4,     0,  8,  6,     0,  6,  2,     6,  8,  7,    -1, -1, -1,
	  3,  6,  2,     3,  7,  6,     1,  5,  0,     5,  4,  0,    -1, -1, -1,
	  6,  2,  8,     6,  8,  7,     2,  1,  8,     4,  8,  5,     1,  5,  8,
	  9,  5,  4,    10,  1,  6,     1,  7,  6,     1,  3,  7,    -1, -1, -1,
	  1,  6, 10,     1,  7,  6,     1,  0,  7,     8,  7,  0,     9,  5,  4,
	  4,  0, 10,     4, 10,  5,     0,  3, 10,     6, 10,  7,     3,  7, 10,
	  7,  6, 10,     7, 10,  8,     5,  4, 10,     4,  8, 10,    -1, -1, -1,
	  6,  9,  5,     6, 11,  9,    11,  8,  9,    -1, -1, -1,    -1, -1, -1,
	  3,  6, 11,     0,  6,  3,     0,  5,  6,     0,  9,  5,    -1, -1, -1,
	  0, 11,  8,     0,  5, 11,     0,  1,  5,     5,  6, 11,    -1, -1, -1,
	  6, 11,  3,     6,  3,  5,     5,  3,  1,    -1, -1, -1,    -1, -1, -1,
	  1,  2, 10,     9,  5, 11,     9, 11,  8,    11,  5,  6,    -1, -1, -1,
	  0, 11,  3,     0,  6, 11,     0,  9,  6,     5,  6,  9,     1,  2, 10,
	 11,  8,  5,    11,  5,  6,     8,  0,  5,    10,  5,  2,     0,  2,  5,
	  6, 11,  3,     6,  3,  5,     2, 10,  3,    10,  5,  3,    -1, -1, -1,
	  5,  8,  9,     5,  2,  8,     5,  6,  2,     3,  8,  2,    -1, -1, -1,
	  9,  5,  6,     9,  6,  0,     0,  6,  2,    -1, -1, -1,    -1, -1, -1,
	  1,  5,  8,     1,  8,  0,     5,  6,  8,     3,  8,  2,     6,  2,  8,
	  1,  5,  6,     2,  1,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  3,  6,     1,  6, 10,     3,  8,  6,     5,  6,  9,     8,  9,  6,
	 10,  1,  0,    10,  0,  6,     9,  5,  0,     5,  6,  0,    -1, -1, -1,
	  0,  3,  8,     5,  6, 10,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 10,  5,  6,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 11,  5, 10,     7,  5, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 11,  5, 10,    11,  7,  5,     8,  3,  0,    -1, -1, -1,    -1, -1, -1,
	  5, 11,  7,     5, 10, 11,     1,  9,  0,    -1, -1, -1,    -1, -1, -1,
	 10,  7,  5,    10, 11,  7,     9,  8,  1,     8,  3,  1,    -1, -1, -1,
	 11,  1,  2,    11,  7,  1,     7,  5,  1,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  3,     1,  2,  7,     1,  7,  5,     7,  2, 11,    -1, -1, -1,
	  9,  7,  5,     9,  2,  7,     9,  0,  2,     2, 11,  7,    -1, -1, -1,
	  7,  5,  2,     7,  2, 11,     5,  9,  2,     3,  2,  8,     9,  8,  2,
	  2,  5, 10,     2,  3,  5,     3,  7,  5,    -1, -1, -1,    -1, -1, -1,
	  8,  2,  0,     8,  5,  2,     8,  7,  5,    10,  2,  5,    -1, -1, -1,
	  9,  0,  1,     5, 10,  3,     5,  3,  7,     3, 10,  2,    -1, -1, -1,
	  9,  8,  2,     9,  2,  1,     8,  7,  2,    10,  2,  5,     7,  5,  2,
	  1,  3,  5,     3,  7,  5,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  7,     0,  7,  1,     1,  7,  5,    -1, -1, -1,    -1, -1, -1,
	  9,  0,  3,     9,  3,  5,     5,  3,  7,    -1, -1, -1,    -1, -1, -1,
	  9,  8,  7,     5,  9,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  5,  8,  4,     5, 10,  8,    10, 11,  8,    -1, -1, -1,    -1, -1, -1,
	  5,  0,  4,     5, 11,  0,     5, 10, 11,    11,  3,  0,    -1, -1, -1,
	  0,  1,  9,     8,  4, 10,     8, 10, 11,    10,  4,  5,    -1, -1, -1,
	 10, 11,  4,    10,  4,  5,    11,  3,  4,     9,  4,  1,     3,  1,  4,
	  2,  5,  1,     2,  8,  5,     2, 11,  8,     4,  5,  8,    -1, -1, -1,
	  0,  4, 11,     0, 11,  3,     4,  5, 11,     2, 11,  1,     5,  1, 11,
	  0,  2,  5,     0,  5,  9,     2, 11,  5,     4,  5,  8,    11,  8,  5,
	  9,  4,  5,     2, 11,  3,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  2,  5, 10,     3,  5,  2,     3,  4,  5,     3,  8,  4,    -1, -1, -1,
	  5, 10,  2,     5,  2,  4,     4,  2,  0,    -1, -1, -1,    -1, -1, -1,
	  3, 10,  2,     3,  5, 10,     3,  8,  5,     4,  5,  8,     0,  1,  9,
	  5, 10,  2,     5,  2,  4,     1,  9,  2,     9,  4,  2,    -1, -1, -1,
	  8,  4,  5,     8,  5,  3,     3,  5,  1,    -1, -1, -1,    -1, -1, -1,
	  0,  4,  5,     1,  0,  5,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  8,  4,  5,     8,  5,  3,     9,  0,  5,     0,  3,  5,    -1, -1, -1,
	  9,  4,  5,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4, 11,  7,     4,  9, 11,     9, 10, 11,    -1, -1, -1,    -1, -1, -1,
	  0,  8,  3,     4,  9,  7,     9, 11,  7,     9, 10, 11,    -1, -1, -1,
	  1, 10, 11,     1, 11,  4,     1,  4,  0,     7,  4, 11,    -1, -1, -1,
	  3,  1,  4,     3,  4,  8,     1, 10,  4,     7,  4, 11,    10, 11,  4,
	  4, 11,  7,     9, 11,  4,     9,  2, 11,     9,  1,  2,    -1, -1, -1,
	  9,  7,  4,     9, 11,  7,     9,  1, 11,     2, 11,  1,     0,  8,  3,
	 11,  7,  4,    11,  4,  2,     2,  4,  0,    -1, -1, -1,    -1, -1, -1,
	 11,  7,  4,    11,  4,  2,     8,  3,  4,     3,  2,  4,    -1, -1, -1,
	  2,  9, 10,     2,  7,  9,     2,  3,  7,     7,  4,  9,    -1, -1, -1,
	  9, 10,  7,     9,  7,  4,    10,  2,  7,     8,  7,  0,     2,  0,  7,
	  3,  7, 10,     3, 10,  2,     7,  4, 10,     1, 10,  0,     4,  0, 10,
	  1, 10,  2,     8,  7,  4,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  9,  1,     4,  1,  7,     7,  1,  3,    -1, -1, -1,    -1, -1, -1,
	  4,  9,  1,     4,  1,  7,     0,  8,  1,     8,  7,  1,    -1, -1, -1,
	  4,  0,  3,     7,  4,  3,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  4,  8,  7,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  9, 10,  8,    10, 11,  8,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  3,  0,  9,     3,  9, 11,    11,  9, 10,    -1, -1, -1,    -1, -1, -1,
	  0,  1, 10,     0, 10,  8,     8, 10, 11,    -1, -1, -1,    -1, -1, -1,
	  3,  1, 10,    11,  3, 10,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  2, 11,     1, 11,  9,     9, 11,  8,    -1, -1, -1,    -1, -1, -1,
	  3,  0,  9,     3,  9, 11,     1,  2,  9,     2, 11,  9,    -1, -1, -1,
	  0,  2, 11,     8,  0, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  3,  2, 11,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  2,  3,  8,     2,  8, 10,    10,  8,  9,    -1, -1, -1,    -1, -1, -1,
	  9, 10,  2,     0,  9,  2,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  2,  3,  8,     2,  8, 10,     0,  1,  8,     1, 10,  8,    -1, -1, -1,
	  1, 10,  2,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  1,  3,  8,     9,  1,  8,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  9,  1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	  0,  3,  8,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,
	 -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1,    -1, -1, -1
};

/*
*	Coefficients for Lassajou Equations
*	v(t) = start_center + lissajou_amplitude * sin(lissajou_frequency * t + lissajou_phase) 
*/

const GLchar* pchSpheresUpdaterVertexShaderSource =
"#version 430 core\n" \
"\n" \
/*Sphere Data*/ \
"struct sphere_descriptor\n" \
"{\n" \
"	vec3 start_center;\n" \
"	vec3 lissajou_amplitude;\n" \
"	vec3 lissajou_frequency;\n" \
"	vec3 lissajou_phase;\n" \
"	float size;\n" \
"}\n" \
"uniform float time;\n" \
"out vec4 sphere_position;\n" \
"void main(void)\n" \
"{\n" \
"	sphere_descriptor spheres[] = sphere_descriptor[]\n" \
"	(\n" \
"		sphere_descriptor(vec3(0.50, 0.50, 0.50), vec3(0.20, 0.25, 0.25), vec3(11.0, 21.0, 31.0), vec3(30.0, 45.0, 90.0), 0.100),\n" \
"		sphere_descriptor( vec3(0.50, 0.50, 0.50), vec3(0.25, 0.20, 0.25), vec3(22.0, 32.0, 12.0), vec3(45.0, 90.0, 120.0), 0.050),\n" \
"		sphere_descriptor( vec3(0.50, 0.50, 0.50), vec3(0.25, 0.25, 0.20), vec3(33.0, 13.0, 23.0), vec3(90.0, 120.0, 150.0), 0.250)\n" \
"	);\n" \
/*Calculate new xyz coordinates of shpere*/ \
"	vec3 sphere_position3 = spheres[gl_VertexID].start_center\n" \
"							  + spheres[gl_VertexID].lissajou_amplitude\n" \
"							  * sin(radians(spheres[gl_VertexID].lissajou_frequency)\n" \
"							  * time + radians(spheres[gl_VertexID].lissajou_phase));\n" \
/*update sphere position coordinate, w coordinate is spheres weight/charge*/ \
"	sphere_position = vec4(sphere_position3, spheres[gl_VertexID].size);\n" \
"}\n"; \

const GLchar* pchSpheresUpdaterFragmentShaderSource =
"#version 430 core\n" \
"\n" \
"void main(void)\n" \
"{\n" \
"}\n"; \

const GLchar* pchScalarFieldVertexShaderSource =
"#version 430 core\n" \
"#define EPSILON 0.000001f\n" \
"#define N_SPHERES 3\n" \
"uniform int samples_per_axis;\n" \
"uniform spheres_uniform_block;\n" \
"{\n" \
"	vec4 input_spheres[N_SPHERES];\n" \
"}\n" \
"out float scalar_field_value;\n" \
"\n" \
"ivec3 decode_space_positions(in int vertex_index)\n" \
"{\n" \
"	int encoded_position = vertex_index;\n" \
"	ivec3 space_position;\n" \
"	space_position.x = encoded_position % samples_per_axis;\n" \
"	encoded_position = encoded_position / samples_per_axis;\n" \
"	space_position.y = encoded_position % samples_per_axis;\n" \
"	encoded_position = encoded_position / samples_per_axis;\n" \
"	space_position.z = encoded_position;\n" \
"	return space_position;\n" \
"}\n" \
"\n" \
"vec3 normalize_space_position_coordinates(in ivec3 space_position)\n" \
"{\n" \
"	vec3 normalized_space_position = vec3(space_position) / float(samples_per_axis - 1);\n" \
"	return normalized_space_position;\n" \
"}\n" \
"\n" \
"float calculate_scalar_field_value(in vec3 position)\n" \
"{\n" \
"	float field_value = 0.0f;\n" \
"	for(int i = 0; i < N_SPHERES; i++)\n" \
"	{\n" \
"		vec3 sphere_position = input_spheres[i].xyz;\n" \
"		float vertex_sphere_distance = length(distance(sphere_position, position));\n" \
"		field_value += input_spheres[i].w / pow(max(EPSILON, vertex_sphere_distance), 2.0);\n" \
"	}\n" \
"	return field_value;\n" \
"}\n" \
"\n" \
"void main()\n" \
"{\n" \
"	ivec3 space_position = decode_space_position(gl_VertexID);\n" \
"	vec3 normalized_position = normalize_space_position_coordinates(space_position);\n" \
"	scalar_field_value = calculate_scalar_field_value(normalized_position);\n" \
"}\n"; \

const GLchar* pchScalarFieldFragmentShaderSource =
"#version 430 core\n" \
"\n" \
"void main(void)\n" \
"{\n" \
"}\n"; \

const GLchar* pchMarchingCubesCellsVertexShaderSource =
"#version 430 core\n" \
"\n" \
"precision lowp sampler3D;\n" \
"uniform sampler3D scalar_field;\n" \
"uniform int cells_per_axis;\n" \
"uniform float iso_level;\n" \
"flat out int cell_type_index;\n" \
"\n"
"int get_cell_type_index(in float cell_corner_field_values[8], in float isolevel)\n" \
"{\n" \
"	int cell_type_index = 0;\n" \
"	for(int i = 0; i < 8; i++)\n" \
"	{\n" \
"		if(cell_corner_field_values[i] < isolevel)\n" \
"		{\n" \
"			cell_type_index |= (1 << i);\n" \
"		}\n" \
"	}\n" \
"	return cell_type_index;\n" \
"}\n" \
"\n" \
"ivec3 decode_space_position(in int cell_index)\n" \
"{\n" \
"	int encoded_position = cell_index;\n" \
"	ivec3 space_position;\n" \
"	space_position.x = encoded_position % cells_per_axis;\n" \
"	encoded_position = encoded_position / cells_per_axis;\n" \
"	space_position.y = encoded_position % cells_per_axis;\n" \
"	encoded_position = encoded_position / cells_per_axis;\n" \
"	space_position.z = encoded_position;\n" \
"	return space_position;\n" \
"}\n" \
"\n" \
"void main(void)\n" \
"{\n" \
"	const int corner_in_cell = 8;\n" \
"	const ivec3 cell_corners_offsets[corner_in_cell] = ivec3[]\n" \
"	(\n" \
"		ivec3(0, 0, 0),\n" \
"		ivec3(1, 0, 0),\n" \
"		ivec3(1, 0, 1),\n" \
"		ivec3(0, 0, 1),\n" \
"		ivec3(0, 1, 0),\n" \
"		ivec3(1, 1, 0),\n" \
"		ivec3(1, 1, 1),\n" \
"		ivec3(0, 1, 1),\n" \
"	);\n" \
"	vec3 scalar_field_normalizers = vec3(textureSize(scalar_field, 0)) - vec3(1, 1, 1);\n" \
"	float scalar_field_in_cell_corners[8];\n" \
"	ivec3 space_position = decode_space_position(gl_VertexID);\n" \
"	for(int i = 0; i < corner_in_cell; i++)\n" \
"	{\n" \
"		ivec3 cell_corner = space_position + cell_corners_offsets[i];\n" \
"		vec3 normalized_cell_corner = vec3(cell_corner) / scalar_field_normalizers;\n" \
"		scalar_field_in_cell_corners[i] = textureLod(scalar_field, normalized_cell_corner, 0.0).r;\n" \
"	}\n" \
"	cell_type_index = get_cell_type_index(scalar_field_in_cell_corners, iso_level);\n" \
"}\n"; \

const GLchar* pchMarchingCubesCellFragmentShaderSource =
"#version 430 core\n" \
"\n" \
"void main(void)\n" \
"{\n" \
"}\n"; \

const GLchar* pchMarchingCubesTrianglesVertexShaderSource =
"#version 430 core\n" \
"\n" \
"precision highp isampler2D;\n" \
"precision highp isampler3D;\n" \
"precision highp sampler2D;\n" \
"precision highp sampler3D;\n" \
"\n" \
"#define EPSILON 0.000001f\n" \
"#define CELLS_PER_AXIS (samples_per_axis - 1)\n" \
"\n" \
"const int mc_vertices_per_cell = 15;\n" \
"uniform int samples_per_axis;\n" \
"uniform isampler3D cell_types;\n" \
"uniform sampler3D scalar_field;\n" \
"uniform isampler2D tri_table;\n" \
"uniform mat4 u_mvp;\n" \
"uniform float iso_level;\n" \
"out vec4 phong_vertex_position;\n" \
"out vec3 phong_vertex_normal_vector;\n" \
"out vec3 phong_vertex_color;\n" \
"\n" \
"float calc_partial_derivative(vec3 begin_vertex, vec3 end_vertex)\n" \
"{\n" \
"	float field_value_begin = textureLod(scalar_field, begin_vertex, 0.0).r;\n" \
"	float field_value_end = textureLod(scalar_field, end_vertex, 0.0).r;\n" \
"	return (field_value_end - field_value_begin) / distance(begin_vertex, end_vertex);\n" \
"}\n" \
"\n" \
"vec3 calc_cell_corner_normal(in vec3 p1)\n" \
"{\n" \
"	vec3 result;\n" \
"	vec3 delta;\n" \
"	delta = vec3(1.0 / float(samples_per_axis - 1), 0, 0);\n" \
"	result.x = calc_partial_derivative(p1 - delta, p1 + delta);\n" \
"	delta = vec3(0.0, 1.0 / float(samples_per_axis - 1), 0.0);\n" \
"	result.y = calc_partial_derivative(p1 - delta, p1 + delta);\n" \
"	delta = vec3(0.0, 0.0, 1.0 / float(samples_per_axis - 1));\n" \
"	result.z = calc_partial_derivative(p1 - delta, p1 + delta);\n"\
"	return result;\n" \
"}\n" \
"\n" \
"vec3 calc_phong_normal(in float start_vertex_portion, in vec3 edge_start, in vec3 edge_end)\n" \
"{\n" \
"	vec3 edge_start_normal = calc_cell_corner_normal(edge_start);\n" \
"	vec3 edge_end_normal = calc_cell_corner_normal(edge_end);\n" \
"	return mix(edge_end_normal, edge_start_normal, start_vertex, portion)\n" \
"}\n" \
"\n" \
"ivec4 decode_cell_position(in int encoded_position_argument)\n" \
"{\n" \
"	ivec4 cell_position;\n" \
"	int encoded_position = encoded_position_argument;\n" \
"	cell_position.w = encoded_position % mc_vertices_per_cell;\n" \
"	encoded_position = encoded_position / mc_vertices_per_cell;\n" \
"	cell_position.x = encoded_position % CELLS_PER_AXIS;\n" \
"	encoded_position = encoded_position / CELLS_PER_AXIS;\n" \
"	cell_position.y = encoded_position % CELLS_PER_AXIS;\n" \
"	encoded_position = encoded_position / CELLS_PER_AXIS;\n" \
"	cell_position.z = encoded_position;\n" \
"	return cell_position;\n" \
"}\n" \
"\n" \
"int get_cell_type(in ivec3 cell_position)\n" \
"{\n" \
"	vec3 cell_position_normalized = vec3(cell_position) / float(CELLS_PER_AXIS - 1);\n" \
"	int cell_type_index = textureLod(cell_types, cell_position_normalized, 0.0).r;\n" \
"	return cell_type_index;\n" \
"}\n" \
"\n" \
"int get_edge_number(in int cell_type_index, in int combined_triangle_no_and_vertex_no)\n" \
"{\n" \
"	vec2 tri_table_index = vec2(float(combined_triangle_no_and_vertex_no) / 14.0, float(cell_type_index) / 255.0);\n" \
"	return textureLod(tri_table, tri_table_index, 0.0).r;\n" \
"}\n" \
"\n" \
"vec3 get_edge_coordinates(in vec3 cell_origin_corner_coordinates, in int edge_number, in bool is_edge_start_vertex)\n" \
"{\n" \
"	const int edge_begins_in_cell_corners[12] = int[] (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3);\n" \
"	const int edge_ends_in_cell_corners[12] = int[] (1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7);\n" \
"	const ivec3 cell_corners_offsets[8] = ivec3[8]\n" \
"	(\n" \
"		ivec3(0, 0, 0),\n" \
"		ivec3(1, 0, 0),\n" \
"		ivec3(1, 0, 1),\n" \
"		ivec3(0, 0, 1),\n" \
"		ivec3(0, 1, 0),\n" \
"		ivec3(1, 1, 0),\n" \
"		ivec3(1, 1, 1),\n" \
"		ivec3(0, 1, 1)\n" \
"	);\n" \
"	int edge_corner_no;\n" \
"	if(is_edge_start_vertex)\n" \
"	{\n" \
"		edge_corner_no = edge_begins_in_cell_corners[edge_number];\n" \
"	}\n" \
"	else\n" \
"	{\n" \
"		edge_corner_no = edge_ends_in_cell_corners[edge_number];\n" \
"	}\n" \
"	vec3 normalized_corner_offsets = vec3(cell_corners_offsets[edge_corner_no]) / float(samples_per_axis - 1);\n" \
"	vec3 edge_corner = cell_origin_corner_coordinates + normalized_corner_offsets;\n" \
"	return edge_corner;\n" \
"}\n" \
"\n" \
"float get_start_corner_portion(in vec3 start_corner, in vec3 end_corner, in float iso_level)\n" \
"{\n" \
"	float result;\n" \
"	float start_field_value = textureLod(scalar_field, start_corner, 0.0).r;\n" \
"	float end_field_value = textureLod(scalar_field, end_corner, 0.0).r;\n" \
"	float field_delta = abs(start_field_value - end_field_value);\n" \
"	if(field_delta > EPSILON)\n" \
"	{\n" \
"		result = abs(end_field_value - iso_level) / field_delta;\n" \
"	}\n" \
"	return result;\n" \
"}\n" \
"\n" \
"void main(void)\n" \
"{\n" \
"	ivec4 cell_position_and_vertex_no = decode_cell_position(gl_VertexID);\n" \
"	ivec3 cell_position = cell_position_and_vertex_no.xyz;\n" \
"	int triangle_and_vertex_number = cell_position_and_vertex_no.w;\n" \
"	int cell_type_index = get_cell_type(cell_position);\n" \
"	int edge_number = get_edge_number(cell_type_index, triangle_and_vertex_number);\n" \
"	if(edge_number != -1)\n" \
"	{\n" \
"		vec3 cell_origin_corner = vec3(cell_position) / float(samples_per_axis - 1);\n" \
"		vec3 start_corner = get_edge_coordinates(cell_origin_corner, edge_number, true);\n" \
"		vec3 end_corner = get_edge_coordinates(cell_origin_corner, edge_number, false);\n" \
"		float start_vertex_portion = get_start_corner_portion(start_corner, end_corner, iso_level);\n" \
"		vec3 edge_middle_vertex = mix(end_corner, start_corner, start_vertex_portion);\n" \
"		vec3 vertex_normal = calc_phong_normal(start_vertex_portion, start_corner, end_corner);\n" \
"		gl_Position = u_mvp * vec4(edge_middle_vertex, 1.0);\n" \
"		phong_vertex_position = gl_Position;\n" \
"		phong_vertex_normal_vector = vertex_normal;\n" \
"		phong_vertex_color = vec3(0.7);\n" \
"	}\n" \
"	else\n" \
"	{\n" \
"		gl_Position = vec4(0.0);\n" \
"		phong_vertex_position = gl_Position;\n" \
"		phong_vertex_normal_vector = vec3(0);\n" \
"		phong_vertex_color = vec3(0.0);\n" \
"	}\n" \
"}\n"; \

const GLchar* pchMarchingCubesTrianglesFragmentShaderSource =
"#version 430 core\n" \
"\n" \
"precision lowp float;\n" \
"uniform float time;\n" \
"in vec4 phong_vertex_position;\n" \
"in vec4 phong_vertex_normal_vector;\n" \
"in vec4 phong_vertex_color;\n" \
"out vec4 FragColor;\n" \
"void main(void)\n" \
"{\n" \
"	const float light_distance = 5.0;\n" \
"	float theta = float(time);\n" \
"	float phi = float(time) / 3.0;\n" \
"	vec3 light_position = vec3\n" \
"	(\n" \
"		light_distance * cos(theta) * sin(phi),\n" \
"		light_distance * cos(theta) * cos(phi),\n" \
"		light_distance * sin(theta)\n" \
"	);\n" \
"	const vec3 ambient_color = vec3(0.1, 0.1, 0.1);\n" \
"	const float attenuation = 1.0;\n" \
"	const float shininess = 3.0;\n" \
"	vec3 normal_direction = normalize(phong_vertex_normal_vector);\n" \
"	vec3 view_direction = normalize(vec3(vec4(0.0, 0.0, 1.0, 0.0) - phong_vertex_position));\n" \
"	vec3 light_direction = normalize(light_location);\n" \
"	vec3 ambient_lighting = ambient_color * phong_vertex_color;\n" \
"	vec3 diffuse_reflection = attenuation * phong_vertex_color * max(0.0, dot(normal_direction, light_direction));\n" \
"	vec3 specular_reflection = vec3(0.0, 0.0, 0.0);\n" \
"	if(dot(normal_direction, light_direction) >= 0.0)\n" \
"	{\n" \
"		specular_reflection = attenuation * phong_vertex_color * pow(max(0.0, dot(reflect(-light_direction, normal_direction), view_direction)), shininess);\n" \
"	}\n" \
"	FragColor = vec4(ambient_lighting + diffuse_reflection + specular_reflection, 1.0);\n" \
"}\n"; \

bool InitMetaballs(void)
{
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	//
	// Stage 1
	//
	guiSphereUpdaterSPO = glCreateProgram();
	guiSphereUpdaterVSO = glCreateShader(GL_VERTEX_SHADER);
	guiSphereUpdaterFSO = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(guiSphereUpdaterVSO, 1, (const GLchar**)&pchSpheresUpdaterVertexShaderSource, NULL);
	glCompileShader(guiSphereUpdaterVSO);
	if (!CheckCompileStatus(guiSphereUpdaterVSO))
	{
		fprintf(gpFile, "InitMetaBalls Sphere Updater VSO Failed to compile\n");
		return false;
	}

	glShaderSource(guiSphereUpdaterFSO, 1, (const GLchar**)&pchSpheresUpdaterFragmentShaderSource, NULL);
	glCompileShader(guiSphereUpdaterFSO);
	if (!CheckCompileStatus(guiSphereUpdaterFSO))
	{
		fprintf(gpFile, "InitMetaBalls Sphere Updater FSO Failed to compile\n");
		return false;
	}

	glAttachShader(guiSphereUpdaterSPO, guiSphereUpdaterVSO);
	glAttachShader(guiSphereUpdaterSPO, guiSphereUpdaterFSO);

	glTransformFeedbackVaryings(guiSphereUpdaterSPO, 1, &gpchSpherePositionVaryingName, GL_SEPARATE_ATTRIBS);

	glLinkProgram(guiSphereUpdaterSPO);
	if (!CheckLinkStatus(guiSphereUpdaterSPO))
	{
		fprintf(gpFile, "InitMetaBalls Sphere Updater SPO Failed to Link\n");
		return false;
	}

	// Uniforms
	guiSphereUpdaterUniformTimeID = glGetUniformLocation(guiSphereUpdaterSPO, gpchSphereUpdaterUniformTimeName);

	glUseProgram(guiSphereUpdaterSPO);

	glGenBuffers(1, &guiSphereUpdaterSpherePositionsBufferObjectID);
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, guiSphereUpdaterSpherePositionsBufferObjectID);
	glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, iNSpheres * iNSpherePositionComponents * sizeof(GLfloat), NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

	glGenTransformFeedbacks(1, &guiSphereUpdaterTransformFeedbackObjectID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, guiSphereUpdaterTransformFeedbackObjectID);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, guiSphereUpdaterSpherePositionsBufferObjectID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	//
	// Stage 2
	//
	guiScalarFieldSPO = glCreateProgram();
	guiScalarFieldVSO = glCreateShader(GL_VERTEX_SHADER);
	guiScalarFieldFSO = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(guiScalarFieldVSO, 1, (const GLchar**)&pchScalarFieldVertexShaderSource, NULL);
	glCompileShader(guiScalarFieldVSO);
	if (!CheckCompileStatus(guiScalarFieldVSO))
	{
		fprintf(gpFile, "InitMetaBalls Scalar Field VSO Failed to Compile\n");
		return false;
	}

	glShaderSource(guiScalarFieldFSO, 1, (const GLchar**)&pchScalarFieldFragmentShaderSource, NULL);
	glCompileShader(guiScalarFieldFSO);
	if (!CheckCompileStatus(guiScalarFieldFSO))
	{
		fprintf(gpFile, "InitMetaBalls Scalar Field FSO Failed to Compile\n");
		return false;
	}

	glAttachShader(guiScalarFieldSPO, guiScalarFieldVSO);
	glAttachShader(guiScalarFieldSPO, guiScalarFieldFSO);

	glTransformFeedbackVaryings(guiScalarFieldSPO, 1, &gpchScalarFieldValueVaryingName, GL_SEPARATE_ATTRIBS);

	glLinkProgram(guiScalarFieldSPO);
	if (!CheckLinkStatus(guiScalarFieldSPO))
	{
		fprintf(gpFile, "InitMetaBalls Scalar Field SPO Failed to Link\n");
		return false;
	}

	guiScalarFieldUniformSamplesPerAxisNameID = glGetUniformLocation(guiScalarFieldSPO, gpchScalarFieldUniformSamplesPerAxisName);
	guiScalarFieldUniformSpheresID = glGetUniformLocation(guiScalarFieldSPO, gpchScalarFieldUniformSpheresName);

	glUseProgram(guiScalarFieldSPO);

	glUniform1i(guiScalarFieldUniformSamplesPerAxisNameID, guiSamplesPerAxis);
	
	glUniformBlockBinding(guiScalarFieldSPO, guiScalarFieldUniformSpheresID, 0);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, guiSphereUpdaterSpherePositionsBufferObjectID);

	glGenBuffers(1, &guiScalarFieldBufferObjectID);
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, guiScalarFieldBufferObjectID);
	glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, guiSamplesIn3DSpace * sizeof(GLfloat), NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

	glGenTransformFeedbacks(1, &guiScalarFieldTransformFeedbackObjectID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, guiScalarFieldTransformFeedbackObjectID);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, guiScalarFieldBufferObjectID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	glGenTextures(1, &guiScalarFieldTextureObjectID);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, guiScalarFieldTextureObjectID);
	glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32F, guiSamplesPerAxis, guiSamplesPerAxis, guiSamplesPerAxis);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	//
	// Stage 3
	//


}