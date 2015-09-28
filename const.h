#ifndef __const__
#define __const__


#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define SQUARE(X) ((X)*(X))

#define NBCHAR 200

#define GREY_LEVEL 255
#define PI 3.141592654

#define WHITE 255
#define BLACK 0

#define ROUND(a) ((int)(a+0.5))

#define NOTUSED 0
#define USED 1


/*---------------------------------------------------------------*/
/*     nfa computation provided by rafael grompone von gioi      */
/*---------------------------------------------------------------*/

#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))
// #define M_LN10 2.30258509299404568402


/*----------------------------------------------------------*/
/*    Point structrue :  coordinate x and y                 */
/*----------------------------------------------------------*/
typedef struct{
  int x,y;
} point;


/*----------------------------------------------------------*/
/*    Rectangle structure: line segment with width          */
/*----------------------------------------------------------*/
typedef struct {
  float x1,y1,x2,y2;  /* first and second point of the line segment */
  float width;        /* rectangle width */
  float x,y;          /* center of the rectangle */
  float theta;        /* angle */
  float dx,dy;        /* vector with the line segment angle */
  float prec;         /* tolerance angle */
  float p;            /* probability of a point with angle within 'prec' */
} rect;


/*----------------------------------------------------------*/
/*    Data structrue :  include two elements x1 and x2      */
/*----------------------------------------------------------*/
typedef struct {
  float x1,x2;
} data;


/*----------------------------------------------------*/
/*     rectangle iterator                             */
/*----------------------------------------------------*/
typedef struct {
  float vx[4];  
  float vy[4];  
  float ys,ye;  
  int x,y;       
} rect_iter;


typedef struct {
  float norm;
  int x;
  int y;
} norm_list;


#endif




