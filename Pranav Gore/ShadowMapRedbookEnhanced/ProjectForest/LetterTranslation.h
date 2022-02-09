#pragma once

#define LETTER_PICKING_MIN_DISTANCE (10.0f)
#define LETTER_HEIGHT 5.0f

// For ASTROMEDICOMP LETTERS
glm::vec3 LetterA_Translation = glm::vec3(-390.000000f, LETTER_HEIGHT, -220.000000f);
glm::vec3 LetterS_Translation = glm::vec3(-206.000000f, LETTER_HEIGHT, 230.000000f);
glm::vec3 LetterT_Translation = glm::vec3(-29.0f, LETTER_HEIGHT, -49.0f);
glm::vec3 LetterR_Translation = glm::vec3(288.0f, LETTER_HEIGHT, 251.0f);
glm::vec3 LetterO_Translation = glm::vec3(67.0f, LETTER_HEIGHT, -97.0f);
glm::vec3 LetterM_Translation = glm::vec3(-72.0f, LETTER_HEIGHT, 80.0f);
glm::vec3 LetterE_Translation = glm::vec3(868.000000f, LETTER_HEIGHT, -130.000000f);
glm::vec3 LetterD_Translation = glm::vec3(0.000000f, LETTER_HEIGHT, 5.000000f);
glm::vec3 LetterI_Translation = glm::vec3(556.0, LETTER_HEIGHT, 133.0f);
glm::vec3 LetterC_Translation = glm::vec3(207.0f, LETTER_HEIGHT+1.5f, 1.0f);
glm::vec3 LetterO_2_Translation = glm::vec3(197.336868f, LETTER_HEIGHT+1.5f, -251.474136f);
glm::vec3 LetterM_2_Translation = glm::vec3(216.818634f, LETTER_HEIGHT+1.5f, 641.704773f);
glm::vec3 LetterP_Translation = glm::vec3(617.725342f, LETTER_HEIGHT+2.0f, 430.937653f);


// For Final Translations

glm::vec3 LetterA_Final_Translation = glm::vec3(170.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterS_Final_Translation = glm::vec3(165.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterT_Final_Translation = glm::vec3(160.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterR_Final_Translation = glm::vec3(155.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterO_Final_Translation = glm::vec3(151.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterM_Final_Translation = glm::vec3(145.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterE_Final_Translation = glm::vec3(140.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterD_Final_Translation = glm::vec3(135.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterI_Final_Translation = glm::vec3(131.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterC_Final_Translation = glm::vec3(125.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterO_2_Final_Translation = glm::vec3(120.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterM_2_Final_Translation = glm::vec3(115.0f, LETTER_HEIGHT, 470.0f);
glm::vec3 LetterP_Final_Translation = glm::vec3(110.0f, LETTER_HEIGHT, 470.0f);

// for RASTER letters
//glm::vec3 LetterR_RG_Translation = glm::vec3(110.0f, 2.0f, -20.0f);
//glm::vec3 LetterA_RG_Translation = glm::vec3(112.0f, 2.0f, -20.0f);
//glm::vec3 LetterS_RG_Translation = glm::vec3(114.0f, 2.0f, -20.0f);
//glm::vec3 LetterT_RG_Translation = glm::vec3(116.0f, 2.0f, -20.0f);
//glm::vec3 LetterE_RG_Translation = glm::vec3(118.0f, 2.0f, -20.0f);
//glm::vec3 LetterR_RG2_Translation = glm::vec3(120.0f, 2.0f, -20.0f);

// For Final translations
glm::vec3 LetterR_RG_Translation = glm::vec3(160.0f, 18.0f, 420.0f);
glm::vec3 LetterA_RG_Translation = glm::vec3(155.0f, 17.5f, 420.0f);
glm::vec3 LetterS_RG_Translation = glm::vec3(150.0f, 18.0f, 420.0f);
glm::vec3 LetterT_RG_Translation = glm::vec3(145.0f, 18.5f, 420.0f);
glm::vec3 LetterE_RG_Translation = glm::vec3(140.0f, 18.0f, 420.0f);
glm::vec3 LetterR_RG2_Translation = glm::vec3(135.0f, 18.0f, 420.0f);

// Scale Constant
glm::vec3 LetterScaleConstant = glm::vec3(15.0f, 15.0f, 15.0f);

// For Bouncing Letters
GLfloat LetterBounceFactor = 0.005f;

// For Model Colors
glm::vec4 RedColor = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
glm::vec4 GreenColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
glm::vec4 BlueColor = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
glm::vec4 CyanColor = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
glm::vec4 MagentaColor = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
glm::vec4 YellowColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
glm::vec4 BlackColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
glm::vec4 WhiteColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
glm::vec4 GreyColor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

GLfloat gfLetterRotationAngle = 0.0f;
GLfloat gfRasterRotateAngle = 0.0f;

BOOL gbLetterA_SetFlag = FALSE;
BOOL gbLetterS_SetFlag = FALSE;
BOOL gbLetterT_SetFlag = FALSE;
BOOL gbLetterR_SetFlag = FALSE;
BOOL gbLetterO_SetFlag = FALSE;
BOOL gbLetterM_SetFlag = FALSE;
BOOL gbLetterE_SetFlag = FALSE;
BOOL gbLetterD_SetFlag = FALSE;
BOOL gbLetterI_SetFlag = FALSE;
BOOL gbLetterC_SetFlag = FALSE;
BOOL gbLetterO_2_SetFlag = FALSE;
BOOL gbLetterM_2_SetFlag = FALSE;
BOOL gbLetterP_SetFlag = FALSE;
BOOL gbUniversalLetterMoveFlag = FALSE;

//Individual Letter reached final position
BOOL gbLetterA_ReachedFinalPosFlag = FALSE;
BOOL gbLetterS_ReachedFinalPosFlag = FALSE;
BOOL gbLetterT_ReachedFinalPosFlag = FALSE;
BOOL gbLetterR_ReachedFinalPosFlag = FALSE;
BOOL gbLetterO_ReachedFinalPosFlag = FALSE;
BOOL gbLetterM_ReachedFinalPosFlag = FALSE;
BOOL gbLetterE_ReachedFinalPosFlag = FALSE;
BOOL gbLetterD_ReachedFinalPosFlag = FALSE;
BOOL gbLetterI_ReachedFinalPosFlag = FALSE;
BOOL gbLetterC_ReachedFinalPosFlag = FALSE;
BOOL gbLetterO_2_ReachedFinalPosFlag = FALSE;
BOOL gbLetterM_2_ReachedFinalPosFlag = FALSE;
BOOL gbLetterP_ReachedFinalPosFlag = FALSE;
BOOL gbAllLettersReachedFinalPosFlag = FALSE; //All ASTROMEDICOM Letter reached final position

BOOL gbLetterGameOver = FALSE;

