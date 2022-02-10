#pragma once


//Sphere-> Radius, stack, slices, pos[], tex[], nor[], elem[]
extern int CreateSphere_RRJ(float, int, int, float[], float[], float[], unsigned short[]);

//Semisphere-> Radius, stack, slices, pos[], tex[], nor[], elem[]
extern int CreateSemiSphere_RRJ(float, int, int, float[], float[], float[], unsigned short[]);

//Cylinder-> Radius, length, slices, pos[], nor[], elem[]
extern int CreateCylinder_RRJ(float, float, int, float[], float[], unsigned short[]);

//Frustum-> Radius1, Radius2, Length, slices, pos[], nor[], elem[]
extern int CreateFrustum_RRJ(float, float, float, int, float[], float[], unsigned short[]);



