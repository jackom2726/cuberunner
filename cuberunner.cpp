////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <stdexcept>
#if __GNUG__
#   include <tr1/memory>
#endif

#ifdef __MAC__
#   include <OpenGL/gl3.h>
#   include <GLUT/glut.h>
#else
#   include <GL/glew.h>
#   include <GL/glut.h>
#endif

#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "quat.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"

using namespace std; // for string, vector, iostream, and other standard C++ stuff
using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.5.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.5. Use GLSL 1.5 unless your system does not support it.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
static const bool g_Gl2Compatible = false;

// window dimensions
static int g_windowWidth = 1280;
static int g_windowHeight = 512;

static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundSize = 10.0;   // half the ground length
static const float g_groundY = -.05;      // y coordinate of the ground
static const float g_cubeSideLength = .22;
static const float g_runnerZ = 3.5;
static float g_groundX = 0; // x coordinate of ground

// gameplay variables
//static int g_numCubes = 0;
static float g_cubeFieldLeftSide = -2;
static float g_cubeFieldWidth = g_windowHeight / 128.0;
static const float g_furthestCubeZ = -3.0;
static const float g_nearestCubeZ = 1.0;
static const float g_zRange = g_nearestCubeZ - g_furthestCubeZ;
static bool g_leftDown = false;
static bool g_rightDown = false;
static const float g_xTranslationAmount = .05;
static const float g_maxRotationAngle = 35; // max screen tilt angle in degrees
static const float g_sinHalfMaxRotationAngle = sin(0.5 * g_maxRotationAngle * CS175_PI/180);

// simulation constants
static const int g_simulationsPerSecond = 40;
static const int g_simRateOriginal = 5;
static const int g_simRateLowBound = 2;
static const float g_cubeIncrDisMin = .06;
static const float g_cubeIncrDisMax = .1;

// simulation variables
static int g_simCount = -1; // counts simulations (cycles through 3*g_simulationsPerCubeGen*g_simulationsPerSecond)
static int g_simulationsPerCubeGen = g_simRateOriginal; // number of simulations that pass for each generated cube
static float g_secondsPerLevel = 5.0; // number of seconds that pass before speed increases in tutorial mode or color changes in normal gameplay mode
static float g_cubeIncrDis = g_cubeIncrDisMin; // the distance each cube moves for each simulaiton (each call to runCubes)
static bool g_gameOn = true; // indicates whether the game has been paused after a collision
static bool g_gamePaused = false; // indicates whether the game has been paused after the 'p' key is pressed

// game modes
static bool g_tutorialMode = true; // tutorial mode (where the game begins)
static bool g_rgbCubesMode = false; // normal gameplay mode
static bool g_deathMode = false; // death mode

static bool g_autonomous = false; // AI plays game

// jump vairables
static const float g_jumpPeak = g_cubeSideLength + .5; // high enough to jump over a cube
static float g_jumpHeight = 0.0; // indicates current jump height
static float g_jumpAmount = .1;
static bool g_jumpInProgress = false;
static bool g_jumpPeakReached = false;

static int g_cyclesRequiredToClearCube = ceil((.5 * g_cubeSideLength) / g_xTranslationAmount);
static int g_minCyclesRequiredToJumpCube = ceil(g_cubeSideLength / g_jumpAmount);
static int g_maxCyclesRequiredToJumpCube = ceil(g_jumpPeak / g_jumpAmount);

// mouse controls
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event

static int g_activeShader = 0;

time_t start_time; // start time of round
time_t pause_begin; // start time of paused time
double pause_time = 0; // number of seconds that have elapsed in paused time

struct ShaderState {
  GlProgram program;

  // Handles to uniform variables
  GLint h_uLight, h_uLight2;
  GLint h_uProjMatrix;
  GLint h_uModelViewMatrix;
  GLint h_uNormalMatrix;
  GLint h_uColor;

  // Handles to vertex attributes
  GLint h_aPosition;
  GLint h_aNormal;

  ShaderState(const char* vsfn, const char* fsfn) {
    readAndCompileShader(program, vsfn, fsfn);

    const GLuint h = program; // short hand

    // Retrieve handles to uniform variables
    h_uLight = safe_glGetUniformLocation(h, "uLight");
    h_uLight2 = safe_glGetUniformLocation(h, "uLight2");
    h_uProjMatrix = safe_glGetUniformLocation(h, "uProjMatrix");
    h_uModelViewMatrix = safe_glGetUniformLocation(h, "uModelViewMatrix");
    h_uNormalMatrix = safe_glGetUniformLocation(h, "uNormalMatrix");
    h_uColor = safe_glGetUniformLocation(h, "uColor");

    // Retrieve handles to vertex attributes
    h_aPosition = safe_glGetAttribLocation(h, "aPosition");
    h_aNormal = safe_glGetAttribLocation(h, "aNormal");

    if (!g_Gl2Compatible)
      glBindFragDataLocation(h, 0, "fragColor");
    checkGlErrors();
  }

};

static const int g_numShaders = 2;
static const char * const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"}
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
  {"./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"}
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
  Cvec3f p, n;

  VertexPN() {}
  VertexPN(float x, float y, float z,
           float nx, float ny, float nz)
    : p(x,y,z), n(nx, ny, nz)
  {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* functions from geometrymaker.h
  VertexPN(const GenericVertex& v) {
    *this = v;
  }

  VertexPN& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  GlArrayObject vao;
  int vboLen, iboLen;

  Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // bind the object's VAO
    glBindVertexArray(vao);

    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aNormal);

    // bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
    safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

    // bind ibo
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aNormal);

    // disable VAO
    glBindVertexArray(NULL);
  }
};


// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_runner, g_cube;

// --------- Scene
static float g_light1X = 0.0;
static float g_light2X = 0.0;
static const Cvec3 g_light1(g_light1X, 3.0, 14.0), g_light2(g_light2X, 3.0, -1.0);  // define two lights positions in world space
static const RigTForm g_originalSkyRbt = RigTForm(Cvec3(0.0, 0.25, 4.0));
static RigTForm g_skyRbt = g_originalSkyRbt; // camera
static RigTForm g_runnerRbt = RigTForm(); // runner

static const int g_numLayers = 7;
static deque<RigTForm> g_cubeRbt[g_numLayers]; // holds cubes
static deque<Cvec3f> g_cubeColors[g_numLayers]; // holds cube colros

static Cvec3f g_runnerColor; // runner color

///////////////// END OF G L O B A L S //////////////////////////////////////////////////

static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPN vtx[4] = {
    VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN(g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN(g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initRunner() {
    // A small triangle appearing at the bottom of the screen
    VertexPN vtx[3] = {
        VertexPN(-.03, 0, g_runnerZ, 0, 1, 0),
        VertexPN(.03, 0,  g_runnerZ, 0, 1, 0),
        VertexPN(0, .03, g_runnerZ, 0, 1, 0),
    };
    unsigned short idx[] = {0, 1, 2};
    g_runner.reset(new Geometry(&vtx[0], &idx[0], 3, 6));
}

///////////////// HELPER FUNCTIONS //////////////////////////////////////////////////

static void printCubeXValues() {
    for (int layer = 0; layer < g_numLayers; layer++) {
        cout << "LAYER " << layer << ":" << endl;
        for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
            cout << "\t" << g_cubeRbt[layer][i].getTranslation()[0] << endl;
        }
    }
}

static void changeColors() {
    // RGB CUBES MODE
    if (g_rgbCubesMode) {
        glClearColor(255/255., 255/255., 255/255., 0.); // white sky
        g_runnerColor = Cvec3f(108/255.0, 91/255.0, 5/255.0); // gold runner
    }
    // DEATH MODE
    else if (g_deathMode) {
        glClearColor(0/255., 0/255., 0/255., 0.); // black sky
        g_runnerColor = Cvec3f(245/255.0, 42/255.0, 76/255.0); // red runner
    }
    // NORMAL MODE
    else {
        glClearColor(42/255., 148/255., 253/255., 0.); // blue sky
        g_runnerColor = Cvec3f(0/255.0, 60/255.0, 255/255.0); // blue runner
    }
}

// increases cube speed as tutorial progresses
static void setCubeIncrDis() {
    g_cubeIncrDis = g_cubeIncrDisMax - ( (g_cubeIncrDisMax - g_cubeIncrDisMin) / (g_simRateOriginal - g_simRateLowBound))*(g_simulationsPerCubeGen - g_simRateLowBound);
}

// Jump functions:
// raises camera and runner to jumpPeak
static void jump() {
    if (abs(g_jumpPeak - g_jumpHeight) > g_jumpAmount) {
        g_jumpHeight = g_jumpHeight + g_jumpAmount;
    }
    else {
        g_jumpPeakReached = true;
    }
}
// lowers camera and runner after reaching jumpPeak
static void descend() {
    if (g_jumpHeight >= g_jumpAmount) {
        g_jumpHeight = g_jumpHeight - g_jumpAmount;
    }
    else {
        g_jumpHeight = 0.0;
        g_jumpInProgress = false;
        g_jumpPeakReached = false;
    }
}
static void handleJump() {
    if(g_jumpPeakReached) {
        descend();
        g_skyRbt = (g_originalSkyRbt * RigTForm(Cvec3(0, -.1, 0)) * inv(g_originalSkyRbt)) * g_skyRbt;
        g_runnerRbt = (g_originalSkyRbt * RigTForm(Cvec3(0, -.1, 0)) * inv(g_originalSkyRbt)) * g_runnerRbt;
    }
    else {
        jump();
        g_skyRbt = (g_originalSkyRbt * RigTForm(Cvec3(0, .1, 0)) * inv(g_originalSkyRbt)) * g_skyRbt;
        g_runnerRbt = (g_originalSkyRbt * RigTForm(Cvec3(0, .1, 0)) * inv(g_originalSkyRbt)) * g_runnerRbt;
    }
}

// removes cubes from plane
static void clearCubes() {
    //g_numCubes = 0;
    for (int layer = 0; layer < g_numLayers; layer++) {
        g_cubeRbt[layer].clear();
        g_cubeColors[layer].clear();
    }
}

// adds cubes to the plane
static void addCubes() {
    g_simCount = (g_simCount + 1) % (int)(g_secondsPerLevel * g_simulationsPerSecond * 3);
    if (g_simCount % g_simulationsPerCubeGen == 0) {
        float x = ((double) rand() / (RAND_MAX));
        float z = ((double) rand() / (RAND_MAX));
        float current_cubeZ = g_furthestCubeZ + g_zRange*z;
        RigTForm current_cube = RigTForm(Cvec3(g_cubeFieldLeftSide + g_cubeFieldWidth*x,g_groundY + .5*g_cubeSideLength, current_cubeZ));
        //int current_layer = (int) (current_cubeZ - g_furthestCubeZ) / ( ((int) g_zRange / g_numLayers) + 1);
        int current_layer = (int) (x*g_numLayers);
        g_cubeRbt[current_layer].push_back(current_cube);
        
        // SETTING COLOR
        if (g_rgbCubesMode) {
            // RED-GREEN-BLUE LEVELS MODE
            float c = ((double) rand() / (RAND_MAX));
            c = (.9 * c) + .1;
            if(g_simCount < g_secondsPerLevel * g_simulationsPerSecond) {
                g_cubeColors[current_layer].push_back(Cvec3f(c,0,0));
            }
            else if (g_simCount < 2*g_secondsPerLevel * g_simulationsPerSecond) {
                g_cubeColors[current_layer].push_back(Cvec3f(0,c,0));
            }
            else {
                g_cubeColors[current_layer].push_back(Cvec3f(0,0,c));
            }
        }
        else if(g_deathMode) {
            g_cubeColors[current_layer].push_back(Cvec3f(.1,.1,.1));
        }
        else {
            // RANDOM COLORS MODE
            float r = ((double) rand() / (RAND_MAX));
            float g = ((double) rand() / (RAND_MAX));
            float b = ((double) rand() / (RAND_MAX));
            g_cubeColors[current_layer].push_back(Cvec3f(r,g,b));
        }
        
        //g_numCubes++;
    }
}

static void advanceCube(int layer, int i, Cvec3 current_position) {
    // if the cube isn't already behind the camera, move it forward and spin it
    if (current_position[2] < g_skyRbt.getTranslation()[2] + g_cubeSideLength) {
        g_cubeRbt[layer][i].setTranslation((Cvec3(current_position[0], current_position[1], current_position[2] + g_cubeIncrDis)));
        g_cubeRbt[layer][i] = (g_cubeRbt[layer][i] * RigTForm(Quat::makeYRotation(100)) * inv(g_cubeRbt[layer][i])) * g_cubeRbt[layer][i];
    }
    else {
        g_cubeRbt[layer].erase(g_cubeRbt[layer].begin() + i);
        g_cubeColors[layer].erase(g_cubeColors[layer].begin() + i);
    }
}

static void moveCubesForward() {
    for (int layer = 0; layer < g_numLayers; layer++) {
        for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
            Cvec3 current_position = g_cubeRbt[layer][i].getTranslation();
            g_cubeRbt[layer][i].setTranslation((Cvec3(current_position[0], current_position[1], current_position[2] + g_cubeIncrDis)));
            g_cubeRbt[layer][i] = (g_cubeRbt[layer][i] * RigTForm(Quat::makeYRotation(100)) * inv(g_cubeRbt[layer][i])) * g_cubeRbt[layer][i];
        }
    }
}

static void moveCubesBack() {
    for (int layer = 0; layer < g_numLayers; layer++) {
        for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
            Cvec3 current_position = g_cubeRbt[layer][i].getTranslation();
            g_cubeRbt[layer][i].setTranslation((Cvec3(current_position[0], current_position[1], current_position[2] - g_cubeIncrDis)));
            g_cubeRbt[layer][i] = (g_cubeRbt[layer][i] * RigTForm(Quat::makeYRotation(-100)) * inv(g_cubeRbt[layer][i])) * g_cubeRbt[layer][i];
        }
    }
}

static void detectCollision(Cvec3 current_position) {
    // if the runner point ever falls inside a cube, we have a collision
    if (abs(g_skyRbt.getTranslation()[0] - current_position[0]) < (sqrt(2.0)/2.0)*g_cubeSideLength &&
        abs(g_runnerRbt.getTranslation()[1] - current_position[1]) < .5*g_cubeSideLength &&
        abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength ) {
        
        cout << endl << "COLLISION!" << endl;
        cout << "Time: " << (float)(difftime(time(0), start_time) - pause_time) << " seconds" << endl;
        cout << "Press the up arrow key to continue playing" << endl;
        
        // if we were in tutorial mode, restart the tutorial
        if(g_tutorialMode) {
            g_simCount = -1;
            g_simulationsPerCubeGen = g_simRateOriginal;
            setCubeIncrDis();
        }
        
        //pause the game
        g_gameOn = false;
    }
}

// moves camera and runner left while tilting screen clockwise
static void moveLeft() {
    if (g_skyRbt.getRotation()[3] < g_sinHalfMaxRotationAngle) {
        // rotates the camera left
        g_skyRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(1)) * inv(g_skyRbt)) * g_skyRbt;
        g_runnerRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(1)) * inv(g_skyRbt)) * g_runnerRbt;
    }
    // translates the camera left
    g_skyRbt = (g_originalSkyRbt * RigTForm(Cvec3(-g_xTranslationAmount, 0, 0)) * inv(g_originalSkyRbt)) * g_skyRbt;
    g_runnerRbt = (g_originalSkyRbt * RigTForm(Cvec3(-g_xTranslationAmount, 0, 0)) * inv(g_originalSkyRbt)) * g_runnerRbt;
    
    // shifts rest of the screen right
    g_cubeFieldLeftSide = g_cubeFieldLeftSide - g_xTranslationAmount;
    g_groundX = g_groundX - g_xTranslationAmount;
    g_light1X = g_light1X - g_xTranslationAmount;
    g_light2X = g_light2X - g_xTranslationAmount;
}

// moves camera and runner right while tilting screen counter-clockwise
static void moveRight() {
    if (g_skyRbt.getRotation()[3] > -g_sinHalfMaxRotationAngle) {
        // rotates the camera left
        g_skyRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(-1)) * inv(g_skyRbt)) * g_skyRbt;
        g_runnerRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(-1)) * inv(g_skyRbt)) * g_runnerRbt;
    }
    // translates the camera left
    g_skyRbt = (g_originalSkyRbt * RigTForm(Cvec3(g_xTranslationAmount, 0, 0)) * inv(g_originalSkyRbt)) * g_skyRbt;
    g_runnerRbt = (g_originalSkyRbt * RigTForm(Cvec3(g_xTranslationAmount, 0, 0)) * inv(g_originalSkyRbt)) * g_runnerRbt;
    
    // shifts rest of the screen left
    g_cubeFieldLeftSide = g_cubeFieldLeftSide + g_xTranslationAmount;
    g_groundX = g_groundX + g_xTranslationAmount;
    g_light1X = g_light1X + g_xTranslationAmount;
    g_light2X = g_light2X + g_xTranslationAmount;
}

// undo any tilting to the screen
static void resetScreenRotation() {
    if (abs(g_skyRbt.getRotation()[3]) <.01) {
        // reset rotation
        g_skyRbt.setRotation(Quat());
        
        g_runnerRbt.setRotation(Quat());
        g_runnerRbt.setTranslation(Cvec3(g_skyRbt.getTranslation()[0], g_runnerRbt.getTranslation()[1], g_runnerRbt.getTranslation()[2]));
    }
    // if we're tilted left, tilt right
    // if we're tilted right, tilt left
    else if (g_skyRbt.getRotation()[3] < 0) {
        g_skyRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(3)) * inv(g_skyRbt)) * g_skyRbt;
        
        g_runnerRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(3)) * inv(g_skyRbt)) * g_runnerRbt;
    }
    else if (g_skyRbt.getRotation()[3] > 0) {
        g_skyRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(-3)) * inv(g_skyRbt)) * g_skyRbt;
        
        g_runnerRbt = (g_skyRbt * RigTForm(Quat::makeZRotation(-3)) * inv(g_skyRbt)) * g_runnerRbt;
    }
}

///////////////// END OF HELPER FUNCTIONS //////////////////////////////////////////////////

static void runCubes(int dontCare) {
    
    addCubes();
    
    // when in tutorial mode, speed up every 5 seconds and increase cube-generation rate
    // until you reach the normal gameplay speed, at which point switch to normal gameplay
    if (g_simulationsPerCubeGen > g_simRateLowBound && g_simCount > 0 && g_simCount%(int)(g_secondsPerLevel * g_simulationsPerSecond) == 0 && g_tutorialMode) {
        g_simulationsPerCubeGen--;
        if(g_simulationsPerCubeGen == g_simRateLowBound) {
            cout << endl << "Normal Gameplay Mode" << endl;
            g_rgbCubesMode = true;
            g_tutorialMode = false;
            changeColors();
        }
        setCubeIncrDis();
        cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
    }
    
    // move cubes and detect collisions
    for (int layer = 0; layer < g_numLayers; layer++) {
        for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
            
            Cvec3 current_position = g_cubeRbt[layer][i].getTranslation();
            
            advanceCube(layer, i, current_position);
            
            detectCollision(current_position);
        }
    }
    // AI plays game by choosing the least crowded paths and jumping when necessary
    if (g_autonomous) {
        
        
        int middle_layer = (int) (.5 * g_numLayers);
        
        if (g_cubeRbt[middle_layer].size() > g_cubeRbt[middle_layer - 1].size() || g_cubeRbt[middle_layer].size() > g_cubeRbt[middle_layer + 1].size()) {
            if (g_cubeRbt[middle_layer - 1].size() < g_cubeRbt[middle_layer + 1].size()) {
                g_leftDown = true;
                g_rightDown = false;
            }
            else {
                g_rightDown = true;
                g_leftDown = false;
            }
        }
        else {
            g_leftDown = false;
            g_rightDown = false;
        }
         
        
        
        g_cyclesRequiredToClearCube = ceil((.5 * g_cubeSideLength) / g_xTranslationAmount);
        g_minCyclesRequiredToJumpCube = ceil(g_cubeSideLength / g_jumpAmount);
        g_maxCyclesRequiredToJumpCube = ceil(g_jumpPeak / g_jumpAmount);
        
        for (int layer = 0; layer < g_numLayers; layer++) {
            for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
                Cvec3 current_position = g_cubeRbt[layer][i].getTranslation();
         
                /*
                 g_rightDown = true;
                 if(!g_jumpInProgress) {
                     // if we're not jumping, and we'll hit a cube soon, swerve accordingly
                     if (abs(g_skyRbt.getTranslation()[0] - current_position[0]) < (sqrt(2.0)/2.0)*g_cubeSideLength &&
                         abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength  + g_cubeIncrDis*g_cyclesRequiredToClearCube ) {
                         
                         g_jumpInProgress = true;
                         break;
                     }
                 }
                 // if we're in the middle of a jump, but we're going to move left to avoid it
                 // remember that momentum will make the runner land at the same point it would land if it hadn't jumped!
                 else {
                     if (abs(g_skyRbt.getTranslation()[0] - current_position[0]) < (sqrt(2.0)/2.0)*g_cubeSideLength &&
                         abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength  + g_cubeIncrDis*g_cyclesRequiredToClearCube ) {
                         
                         g_rightDown = false;
                         break;
                     }
                 }
                 */
                
                
                if(!g_jumpInProgress) {
                    // if we're not jumping, and we'll hit a cube soon, swerve accordingly
                    if (abs(g_skyRbt.getTranslation()[0] - current_position[0]) < (sqrt(2.0)/2.0)*g_cubeSideLength &&
                        abs(g_runnerRbt.getTranslation()[1] - current_position[1]) < .5*g_cubeSideLength &&
                        abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength  + g_cubeIncrDis*g_cyclesRequiredToClearCube &&
                        abs(g_runnerZ - current_position[2]) > .5*g_cubeSideLength  + g_cubeIncrDis*(g_cyclesRequiredToClearCube - 1)) {
                        
                        // if the we will crash into the left side of the cube, swerve left
                        if (g_skyRbt.getTranslation()[0] - current_position[0] > 0) {
                            g_rightDown = true;
                            break;
                        }
                        else {
                            g_leftDown = true;
                            break;
                        }
                    }
                    // no time to swerve? jump!
                    else if (abs(g_skyRbt.getTranslation()[0] - current_position[0]) < (sqrt(2.0)/2.0)*g_cubeSideLength &&
                             abs(g_runnerRbt.getTranslation()[1] - current_position[1]) < .5*g_cubeSideLength &&
                             abs(g_runnerZ - current_position[2]) - .5*g_cubeSideLength >= g_cubeIncrDis*g_minCyclesRequiredToJumpCube &&
                             abs(g_runnerZ - current_position[2]) + .5*g_cubeSideLength <= g_cubeIncrDis*g_maxCyclesRequiredToJumpCube) {
                        
                        g_jumpInProgress = true;
                        break;
                    }
                }
                // if we're in the middle of a jump, but we're going to hit a cube, move to avoid it
                // remember that momentum will make the runner land at the same point it would land if it hadn't jumped!
                else {
                    if (abs(g_skyRbt.getTranslation()[0] - current_position[0]) < (sqrt(2.0)/2.0)*g_cubeSideLength &&
                        abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength  + g_cubeIncrDis*g_cyclesRequiredToClearCube ) {
                        
                        // if the we will crash into the left side of the cube, swerve left
                        if (g_skyRbt.getTranslation()[0] - current_position[0] > 0) {
                            g_rightDown = true;
                            break;
                        }
                        else {
                            g_leftDown = true;
                            break;
                        }
                    }
                }
                
            }
        }
        
        // accounts for swerving into things immediately to your left/right
        for (int layer = 0; layer < g_numLayers; layer++) {
            for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
                Cvec3 current_position = g_cubeRbt[layer][i].getTranslation();
                
                if(g_rightDown) {
                    if (current_position[0] - g_skyRbt.getTranslation()[0] < (sqrt(2.0)/2.0)*g_cubeSideLength + g_xTranslationAmount*g_minCyclesRequiredToJumpCube &&
                        current_position[0] - g_skyRbt.getTranslation()[0] > 0 &&
                        abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength  + g_cubeIncrDis*g_cyclesRequiredToClearCube ) {
                        
                        if (!g_jumpInProgress) {
                           g_jumpInProgress = true;
                        }
                        else {
                            g_rightDown = false;
                            g_leftDown = false;
                        }
                    }
                }
                else if(g_leftDown) {
                    if (current_position[0] - g_skyRbt.getTranslation()[0] > -((sqrt(2.0)/2.0)*g_cubeSideLength + g_xTranslationAmount*g_minCyclesRequiredToJumpCube) &&
                        current_position[0] - g_skyRbt.getTranslation()[0] < 0 &&
                        abs(g_runnerZ - current_position[2]) < .5*g_cubeSideLength  + g_cubeIncrDis*g_cyclesRequiredToClearCube ) {

                        if (!g_jumpInProgress) {
                            g_jumpInProgress = true;
                        }
                        else {
                            g_rightDown = false;
                            g_leftDown = false;
                        }
                    }
                }
                
            }
        }
        
        
        
    }
    
    // continue tilting/moving camera as long as arrow keys are still held down
    if(g_leftDown) {
        moveLeft();
    }
    else if(g_rightDown) {
        moveRight();
    }
    // if arrow keys aren't being held, bring the screen rotation back to 0
    else {
        resetScreenRotation();
    }
    
    if (g_autonomous) {
        g_rightDown = false;
        g_leftDown = false;
    }
    
    // jump
    if(g_jumpInProgress) {
        handleJump();
    }
    
    // schedule this function to be called again
    if (g_gameOn && !g_gamePaused) {
        glutTimerFunc(1000/g_simulationsPerSecond, runCubes, 0);
    }
    glutPostRedisplay(); // signal redisplaying
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(g_cubeSideLength, vtx.begin(), idx.begin());
  g_cube.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
   
  start_time = time(0);
  pause_time = 0;
  cout << endl << "New Game Started" << endl;
    
  // Begin running the cubes
  runCubes(0);
  cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
}

// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}

// takes MVM and its normal matrix to the shaders
static void sendModelViewNormalMatrix(const ShaderState& curSS, const Matrix4& MVM, const Matrix4& NMVM) {
  GLfloat glmatrix[16];
  MVM.writeToColumnMajorMatrix(glmatrix); // send MVM
  safe_glUniformMatrix4fv(curSS.h_uModelViewMatrix, glmatrix);

  NMVM.writeToColumnMajorMatrix(glmatrix); // send NMVM
  safe_glUniformMatrix4fv(curSS.h_uNormalMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}


static void drawStuff() {
  // short hand for current shader state
  const ShaderState& curSS = *g_shaderStates[g_activeShader];

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(curSS, projmat);
    
  // use the skyRbt as the eyeRbt
  const RigTForm invSkyRbt = inv(g_skyRbt);

  const Cvec3 eyeLight1 = Cvec3(invSkyRbt * Cvec4(g_light1X, g_light1[1], g_light1[2], 1)); // g_light1 position in sky coordinates
  const Cvec3 eyeLight2 = Cvec3(invSkyRbt * Cvec4(g_light2X, g_light2[1], g_light2[2], 1)); // g_light2 position in sky coordinates
  safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
  safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);

  // draw ground
  // ===========
  //
  const RigTForm groundRbt = RigTForm(Cvec3(g_groundX,0,0));
  Matrix4 MVM = rigTFormToMatrix(invSkyRbt * groundRbt);
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, .9, .9, .9); // set color
  g_ground->draw(curSS);
    
  // draw runner
  // ===========
  //
  MVM = rigTFormToMatrix(invSkyRbt * g_runnerRbt);
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, g_runnerColor[0], g_runnerColor[1], g_runnerColor[2]);
  g_runner->draw(curSS);

  // draw cubes
  // ==========
  for (int layer = 0; layer < g_numLayers; layer++) {
      for (int i = 0; i < g_cubeRbt[layer].size(); i++) {
          MVM = rigTFormToMatrix(invSkyRbt * g_cubeRbt[layer][i]);
          NMVM = normalMatrix(MVM);
          sendModelViewNormalMatrix(curSS, MVM, NMVM);
          safe_glUniform3f(curSS.h_uColor, g_cubeColors[layer][i][0], g_cubeColors[layer][i][1], g_cubeColors[layer][i][2]);
          g_cube->draw(curSS);
      }
  }
}

static void display() {
  glUseProgram(g_shaderStates[g_activeShader]->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  drawStuff();
  glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_cubeFieldWidth = g_windowHeight / 128.0;
  if (g_cubeFieldWidth < 1) {
    g_cubeFieldWidth = 1;
  }
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  //cerr << "Size of window is now " << w << "x" << h << endl;
  updateFrustFovY();
  glutPostRedisplay();
}


static void motion(const int x, const int y) {
}


static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
    
  glutPostRedisplay();
}

static void keyboardUp(const unsigned char key, const int x, const int y) {
  switch (key) {
  }
  glutPostRedisplay();
}

// new  special keyboard callback, for arrow keys
static void specialKeyboardUp(const int key, const int x, const int y) {
    if (!g_autonomous) {
        switch (key) {
            case GLUT_KEY_RIGHT:
                g_rightDown = false;
                break;
            case GLUT_KEY_LEFT:
                g_leftDown = false;
                break;
        }
    }
    glutPostRedisplay();
}

static void keyboard(const unsigned char key, const int x, const int y) {
    switch (key) {
        // ESC closes window
        case 27:
            exit(0);
        case 'h':
            cout << " ============== H E L P ==============\n\n"
            << "h\t\t\t\thelp menu\n"
            << "s\t\t\t\tsave screenshot\n"
            << "f\t\t\t\tToggle flat shading on/off.\n"
            << "left\t\t\tMove left\n"
            << "right\t\t\tMove right\n"
            << "up\t\t\tResume game after collision\n"
            << "space\t\t\tJump\n"
            << "p\t\t\tPause game\n"
            << "r\t\t\t\tIncrease speed\n"
            << "v\t\t\t\tDecrease speed\n"
            << "q\t\t\t\tIncrease cube generation rate\n"
            << "z\t\t\t\tDecrease cube generation rate\n"
            << "t\t\t\t\tEnter tutorial mode\n"
            << "e\t\t\t\tEnter normal gameplay mode\n"
            << "d\t\t\t\tEnter death mode\n"
            << endl;
            break;
        case 's':
            glFlush();
            writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
            break;
        case 'f':
            g_activeShader ^= 1;
            break;
        // triggers jump
        case ' ':
            if(!g_gamePaused && !g_autonomous) {
                g_jumpInProgress = true;
            }
            break;
        // increases the distance the cubes move each time, effectively making them faster (not in tutorial mode)
        case 'R':
        case 'r':
            if (g_rgbCubesMode || g_deathMode) {
                g_cubeIncrDis += .02;
                cout << "Increasing speed!" << endl;
            }
            break;
        // decreases the distance the cubes move each time, effectively making them slower (not in tutorial mode)
        case 'V':
        case 'v':
            if (g_rgbCubesMode || g_deathMode) {
                if (g_cubeIncrDis > .02) {
                    g_cubeIncrDis -= .02;
                    cout << "Decreasing speed" << endl;
                }
            }
            break;
        // increases the rate at which the cubes are generated (not in tutorial mode)
        case 'Q':
        case 'q':
            if (g_rgbCubesMode || g_deathMode) {
                if (g_simulationsPerCubeGen > 1) {
                    g_simulationsPerCubeGen -= 1;
                }
                if (g_simulationsPerCubeGen == 1) {
                    cout << "Reached Max Cube-Generation Rate" << endl;
                }
                else {
                    cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
                }
            }
            break;
        // decreases the rate at which the cubes are generated (not in tutorial mode)
        case 'Z':
        case 'z':
            if (g_rgbCubesMode || g_deathMode) {
                if (g_simulationsPerCubeGen <= g_simulationsPerSecond) {
                    g_simulationsPerCubeGen += 1;
                    cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
                }
                else {
                    cout << "Reached Min Cube-Generation Rate" << endl;
                }
            }
            break;
        // enters tutorial mode
        case 'T':
        case 't':
            if (!g_gameOn) {
                g_gameOn = true;
                runCubes(0);
            }
            g_simulationsPerCubeGen = g_simRateOriginal;
            setCubeIncrDis();
            g_tutorialMode = true;
            g_rgbCubesMode = false;
            g_deathMode = false;
            clearCubes();
            changeColors();
            start_time = time(0);
            pause_time = 0;
            cout << endl << "Tutorial Mode" << endl;
            cout << "Resetting clock" << endl;
            cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
            break;
        // enters normal gameplay mode
        case 'E':
        case 'e':
            if (!g_gameOn) {
                g_gameOn = true;
                runCubes(0);
            }
            g_simulationsPerCubeGen = g_simRateLowBound;
            setCubeIncrDis();
            g_tutorialMode = false;
            g_rgbCubesMode = true;
            g_deathMode = true;
            clearCubes();
            changeColors();
            start_time = time(0);
            pause_time = 0;
            cout << endl << "Normal Gameplay Mode" << endl;
            cout << "Resetting clock" << endl;
            cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
            break;
        // enters death mode
        case 'D':
        case 'd':
            if (!g_gameOn) {
                g_gameOn = true;
                runCubes(0);
            }
            g_simulationsPerCubeGen = 1;
            g_cubeIncrDis = g_cubeIncrDisMax;
            g_tutorialMode = false;
            g_rgbCubesMode = false;
            g_deathMode = true;
            clearCubes();
            changeColors();
            start_time = time(0);
            pause_time = 0;
            cout << endl << "Death Mode" << endl;
            cout << "Resetting clock" << endl;
            cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
            break;
        case 'P':
        case 'p':
            if (g_gameOn) {
                g_gamePaused = !g_gamePaused;
                if (!g_gamePaused) {
                    cout << "GAME RESUMED" <<endl;
                    pause_time = difftime(time(0), pause_begin);
                    runCubes(0);
                }
                else {
                    cout << endl << "GAME PAUSED" << endl;
                    cout << "Press 'p' to resume" << endl <<endl;
                    pause_begin = time(0);
                    
                    printCubeXValues();
                }
            }
            break;
        case '1':
            g_autonomous = !g_autonomous;
            break;
        case ',':
            if (!g_gameOn || g_gamePaused) {
                moveCubesBack();
            }
            break;
        case '.':
            if (!g_gameOn || g_gamePaused) {
                moveCubesForward();
            }
            break;
  }
  glutPostRedisplay();
}

// new  special keyboard callback, for arrow keys
static void specialKeyboard(const int key, const int x, const int y) {
        switch (key) {
            // move right
            case GLUT_KEY_RIGHT:
                if (!g_autonomous) {
                    g_rightDown = true;
                    break;
                }
            // move left
            case GLUT_KEY_LEFT:
                if (!g_autonomous) {
                    g_leftDown = true;
                    break;
                }
            // resumes game after loss
            case GLUT_KEY_UP:
                if(!g_gameOn) {
                    if (g_tutorialMode) {
                        cout << endl << "Restarting Tutorial" << endl;
                        cout << "Cubes generated per second: " << (int)(g_simulationsPerSecond/g_simulationsPerCubeGen) << endl;
                    }
                    clearCubes();
                    start_time = time(0);
                    pause_time = 0;
                    g_gameOn = true;
                    runCubes(0);
                }
                break;
    }
    glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
#ifdef __MAC__
  glutInitDisplayMode(GLUT_3_2_CORE_PROFILE|GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH); // core profile flag is required for GL 3.2 on Mac
#else
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
#endif
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("CUBERUNNER");                       // title the window

  glutIgnoreKeyRepeat(true);                              // avoids repeated keyboard calls when holding space to emulate middle mouse

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
  glutKeyboardUpFunc(keyboardUp);
  glutSpecialFunc(specialKeyboard);                       // special keyboard callback
  glutSpecialUpFunc(specialKeyboardUp);
    
  glutBitmapCharacter(GLUT_STROKE_ROMAN, 35);
}

static void initGLState() {
  //glClearColor(128./255., 200./255., 255./255., 0.);
  changeColors();
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    if (g_Gl2Compatible)
      g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
    else
      g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  initRunner();
  initCubes();
}

int main(int argc, char * argv[]) {
    
  srand((unsigned int)time(0)); // seeds rand
    
  try {
    initGlutState(argc,argv);

    // on Mac, we shouldn't use GLEW.

#ifndef __MAC__
    glewInit(); // load the OpenGL extensions
#endif

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.5") << endl;

#ifndef __MAC__
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");
#endif

    initGLState();
    initShaders();
    initGeometry();

    glutMainLoop();

    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
