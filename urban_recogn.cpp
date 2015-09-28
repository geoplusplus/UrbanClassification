#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cxcore.h"
#include "ml.h"
#include "highgui.h"
#include "cv.h"
#include "cvaux.h"

#include "const.h"
#include "func.h"

using namespace cv;
using namespace std;

/*-------------------------------------------------------------------*/
/*                        main function                              */
/*-------------------------------------------------------------------*/
int main(int argc, char *argv[]) 
{
   data mean0, mean1;
   mean0.x1 = 0, mean0.x2 = 0;
   mean1.x1 = 0, mean1.x2 = 0;
   Mat var0 = Mat::zeros(2,2,CV_32F);
   Mat var1 = Mat::zeros(2,2,CV_32F);
   float p0 = 0, p1 = 0; 


   printf("--------------------------start training---------------------------\n");
   get_training(mean0, mean1, var0, var1, p0, p1);
   
   printf("-------------------------start classifier--------------------------\n");
   get_classifier(mean0, mean1, var0, var1, p0, p1);

   return 0;
}



/*-------------------------------------------------------*/
/*                     Recal                             */
/*-------------------------------------------------------*/
void recal(Mat mat,int lgth,int wdth)
{
  int i,j;
  float max,min;

  min=mat.at<float>(0,0);
  
  for(i=0;i<lgth;i++) for(j=0;j<wdth;j++)
    if (mat.at<float>(i,j)<min) min=mat.at<float>(i,j);

  for(i=0;i<lgth;i++) for(j=0;j<wdth;j++)
    mat.at<float>(i,j)-=min;

  max=mat.at<float>(0,0);
  
  for(i=0;i<lgth;i++) for(j=0;j<wdth;j++) 
    if (mat.at<float>(i,j)>max) max=mat.at<float>(i,j);

  for(i=0;i<lgth;i++) for(j=0;j<wdth;j++)
    mat.at<float>(i,j)*=(GREY_LEVEL/max);      
}


int compare (const void * a, const void * b)
{
   const norm_list* pa = (norm_list*)a;
   const norm_list* pb = (norm_list*)b;
   return ( pa->norm - pb->norm );
}



void sort(Mat grad, norm_list list[], int length, int width)
{
  
  int i,j;
  for (i=0; i<length; i++){
    for (j=0; j<width; j++){
      list[i*width+j].norm = grad.at<float>(i,j);
      list[i*width+j].x = i;
      list[i*width+j].y = j;
    } 
  }

  qsort(list, length*width, sizeof(norm_list), compare);
}

/*--------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.     */
/*--------------------------------------------------------------------------*/
float get_theta( point* reg, int &reg_size, int x, int y, Mat grad, float &reg_angle, float prec)
{
  float lambda,theta,weight;
  float Ixx = 0.0;
  float Iyy = 0.0;
  float Ixy = 0.0;
  int i;
  float sum = 0.0;

  for(i=0; i<reg_size; i++){
    weight = grad.at<float>(reg[i].x, reg[i].y);
    sum += weight;
    Ixx += ( reg[i].y - y ) * ( reg[i].y - y ) * weight;
    Iyy += ( reg[i].x - x ) * ( reg[i].x - x ) * weight;
    Ixy -= ( reg[i].x - x ) * ( reg[i].y - y ) * weight;
    
  }

  Ixx = Ixx / sum;
  Iyy = Iyy / sum;
  Ixy = Ixy / sum;
  

  lambda = 0.5 * ( Ixx + Iyy - sqrt( (Ixx-Iyy)*(Ixx-Iyy) + 4.0*Ixy*Ixy ) );

  theta = fabs(Ixx)>fabs(Iyy) ? atan2(lambda-Ixx,Ixy) : atan2(Ixy,lambda-Iyy);

  if( fabs(reg_angle-theta) > prec ) theta += PI;

  return theta;
}


/*----------------------------------------------------------*/
/**Computes a rectangle that covers a region of points.     */
/*--------------------------------------------------------- */
void region2rect( point* reg, int &reg_size, Mat grad, float &reg_angle, float prec, rect &rec, float p)
{

  float x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
  int i;

  x = y = sum = 0.0;
  for(i=0; i<reg_size; i++){
    weight = grad.at<float>(reg[i].x, reg[i].y);
    x += reg[i].x * weight;
    y += reg[i].y * weight;
    sum += weight;
  }

  x /= sum;
  y /= sum;
 
  theta = get_theta (reg, reg_size, x, y, grad, reg_angle, prec);

  dx = cos(theta);
  dy = sin(theta);
  l_min = l_max = w_min = w_max = 0.0;
  for(i=0; i<reg_size; i++){

    l =  ( reg[i].x - x) * dx + ( reg[i].y - y) * dy;
    w = -( reg[i].x - x) * dy + ( reg[i].y - y) * dx;

    if( l > l_max ) l_max = l;
    if( l < l_min ) l_min = l;
    if( w > w_max ) w_max = w;
    if( w < w_min ) w_min = w;
  }

  rec.y1 = x + l_min * dx;
  rec.x1 = y + l_min * dy;
  rec.y2 = x + l_max * dx;
  rec.x2 = y + l_max * dy;
  rec.width = w_max - w_min;
  rec.y = x;
  rec.x = y;
  rec.theta = theta;
  rec.dy = dx;
  rec.dx = dy;
  rec.prec = prec;
  rec.p = p;

  if( rec.width < 1.0 ) rec.width = 1.0;
}


/*--------------------------------------------------------------------------------*/
/* find regions where adjacent pixels share the common gradient orientation       */
/*--------------------------------------------------------------------------------*/
void grow_region(Mat grad, Mat angl, Mat label, int x, int y, point* reg, 
                          int &reg_size, float &reg_angle, float prec, int length, int width)
{
  
  float sumdx,sumdy;
   
  reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;

  reg_angle = angl.at<float>(reg[0].x, reg[0].y);
  sumdx = cos(angl.at<float>(reg[0].x, reg[0].y));
  sumdy = sin(angl.at<float>(reg[0].x, reg[0].y));

  label.at<float>(reg[0].x, reg[0].y) = USED;
  
  for (int i=0; i<reg_size; i++){
    for (int x=reg[i].x-1; x<=reg[i].x+1; x++){
      for (int y=reg[i].y-1; y<=reg[i].y+1; y++){
        if ((x>=0&&y>=0) && (x<length&&y<width) && (label.at<float>(x,y)!=USED) && 
                   (fabs(reg_angle-angl.at<float>(x,y)) <= prec) && (grad.at<float>(x,y)!=0)){  
          label.at<float>(x, y) = USED;
          reg[reg_size].x = x;
          reg[reg_size].y = y;
          ++(reg_size);
          sumdx += cos(angl.at<float>(x,y));
          sumdy += sin(angl.at<float>(x,y));
          reg_angle = atan2(sumdy, sumdx);
        }
      }
    }
  }

  if(reg_angle < 0) reg_angle += PI;
}

/*---------------------------------------------------------------*/
/*     nfa computation provided by rafael grompone von gioi      */
/*---------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using the Lanczos approximation.
    See http://www.rskey.org/gamma.htm
  
    The formula used is
    @f[
      \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                  (x+5.5)^{x+0.5} e^{-(x+5.5)}
    @f]
    so
    @f[
      \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                      + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
    @f]
    and
      q0 = 75122.6331530,
      q1 = 80916.6278952,
      q2 = 36308.2951477,
      q3 = 8687.24529705,
      q4 = 1168.92649479,
      q5 = 83.8676043424,
      q6 = 2.50662827511.  
 */

float log_gamma_lanczos(float x)
{
  float q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                 8687.24529705, 1168.92649479, 83.8676043424,
                 2.50662827511 };

  float a = (x+0.5) * log(x+5.5) - (x+5.5);
  float b = 0.0;
  int n;

  for(n=0;n<7;n++){
    a -= log( x + (float) n );
    b += q[n] * pow( x, (float) n );
  }
  return a + log(b);
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using Windschitl method.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
        \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                    \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
    @f]
    so
    @f[
        \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                      + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
    @f]
    This formula is a good approximation when x > 15.
 */
float log_gamma_windschitl(float x)
{
  return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}
    @f]

    The value -log10(NFA) is equivalent but more intuitive than NFA:
    - -1 corresponds to 10 mean false alarms
    -  0 corresponds to 1 mean false alarm
    -  1 corresponds to 0.1 mean false alarms
    -  2 corresponds to 0.01 mean false alarms
    -  ...

    Used this way, the bigger the value, better the detection,
    and a logarithmic scale is used.

    @param n,k,p binomial parameters.
    @param logNT logarithm of Number of Tests

    The computation is based in the gamma function by the following
    relation:
    @f[
        \left(\begin{array}{c}n\\k\end{array}\right)
        = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
    @f]
    We use efficient algorithms to compute the logarithm of
    the gamma function.

    To make the computation faster, not all the sum is computed, part
    of the terms are neglected based on a bound to the error obtained
    (an error of 10% in the result is accepted).
 */
float nfa(int n, int k, float p, float logNT)
{
  float inv[100000];   
  float tolerance = 0.01;      
  float log1term,term,bin_term,mult_term,bin_tail,err,p_term;
  int i;

  if( n==0 || k==0 ) return -logNT;
  if( n==k ) return -logNT - (float) n * log10(p);

  p_term = p / (1.0-p);

  log1term = log_gamma( (float) n + 1.0 ) - log_gamma( (float) k + 1.0 )
           - log_gamma( (float) (n-k) + 1.0 )
           + (float) k * log(p) + (float) (n-k) * log(1.0-p);

  term = exp(log1term);

  if( (term-0.0) <= 0.001 ) {
    if( (float) k > (float) n * p )    
      return -log1term / M_LN10 - logNT;  
    else
      return -logNT;                      
  }

  
  bin_tail = term;
  for(i=k+1;i<=n;i++){  
    bin_term = (float) (n-i+1) * ( i<100000 ?
               ( inv[i]!=0.0 ? inv[i] : ( inv[i] = 1.0 / (float) i ) ) :
               1.0 / (float) i );

    mult_term = bin_term * p_term;
    term *= mult_term;
    bin_tail += term;
    if(bin_term<1.0){
       err = term * ( ( 1.0 - pow( mult_term, (float) (n-i+1) ) ) / (1.0-mult_term) - 1.0 );   
       if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
    }
  }


  return -log10(bin_tail) - logNT;
}


/*----------------------------------------------------*/
/*     rectangle iterator                             */
/*----------------------------------------------------*/
float rect_low(float x, float x1, float y1, float x2, float y2)
{
  if( x1==x2 && y1<y2 ) return y1;
  else if( x1==x2 && y1>y2 ) return y2;
  else if( x1<=x2 && x1<=x && x<=x2 ) return y1 + (x-x1) * (y2-y1) / (x2-x1);
  else return 0;
}


float rect_hi(float x, float x1, float y1, float x2, float y2)
{
  if( x1==x2 && y1<y2 ) return y2;
  else if( x1==x2 && y1>y2 ) return y1;
  else if( x1<=x2 && x1<=x && x<=x2 ) return y1 + (x-x1) * (y2-y1) / (x2-x1);
  else return 0;
}


int rect_end (rect_iter &i)
{ 
  return (float)(i.x) > i.vx[2];
}


void rect_inc (rect_iter &i)
{   
  if( !rect_end(i) ) i.y++;

  while( (float) (i.y) > i.ye && !rect_end(i) ){
      i.x++;
      if( rect_end(i) ) return;

      if( (float)i.x < i.vx[3] )
        i.ys = rect_low((float)i.x, i.vx[0], i.vy[0], i.vx[3], i.vy[3]);
      else
        i.ys = rect_low((float)i.x, i.vx[3], i.vy[3], i.vx[2], i.vy[2]);

      if( (float)i.x < i.vx[1] )
        i.ye = rect_hi((float)i.x, i.vx[0], i.vy[0], i.vx[1], i.vy[1]);
      else
        i.ye = rect_hi((float)i.x, i.vx[1], i.vy[1], i.vx[2], i.vy[2]);

      i.y = (int) ceil(i.ys);
  }
}


rect_iter rect_ini (rect &r)
{
  float vx[4],vy[4];
  int n,offset;
  rect_iter i;

  vx[0] = r.x1 - r.dy * r.width / 2.0;
  vy[0] = r.y1 + r.dx * r.width / 2.0;
  vx[1] = r.x2 - r.dy * r.width / 2.0;
  vy[1] = r.y2 + r.dx * r.width / 2.0;
  vx[2] = r.x2 + r.dy * r.width / 2.0;
  vy[2] = r.y2 - r.dx * r.width / 2.0;
  vx[3] = r.x1 + r.dy * r.width / 2.0;
  vy[3] = r.y1 - r.dx * r.width / 2.0;

  if( r.x1 < r.x2 && r.y1 <= r.y2 ) offset = 0;
  else if( r.x1 >= r.x2 && r.y1 < r.y2 ) offset = 1;
  else if( r.x1 > r.x2 && r.y1 >= r.y2 ) offset = 2;
  else offset = 3;

  
  for(n=0; n<4; n++){
    i.vx[n] = vx[(offset+n)%4];
    i.vy[n] = vy[(offset+n)%4];
  }
 
  i.x = (int) ceil(i.vx[0]) - 1;
  i.y = (int) ceil(i.vy[0]);
  i.ys = i.ye = 0;
 
  rect_inc(i);

  return i;
}



/*--------------------------------------------------------------------------*/
/*            Copy one rectangle structure to another.                      */
/*--------------------------------------------------------------------------*/

void rect_copy(rect &in, rect &out)
{
  out.x1 = in.x1;
  out.y1 = in.y1;
  out.x2 = in.x2;
  out.y2 = in.y2;
  out.width = in.width;
  out.x = in.x;
  out.y = in.y;
  out.theta = in.theta;
  out.dx = in.dx;
  out.dy = in.dy;
  out.prec = in.prec;
  out.p = in.p;
}

/*--------------------------------------------------------------*/
/*               Compute a rectangle's NFA value.               */
/*--------------------------------------------------------------*/

float rect_nfa(rect &rec, Mat angl, float logNT, int length, int width)
{
  rect_iter i;
  int pts = 0;
  int alg = 0;

  for( i=rect_ini(rec); !rect_end(i); rect_inc(i)){
    if( i.x >= 0 && i.y >= 0 && i.x < length && i.y < width ){
      ++pts; 
      if( fabs(rec.theta-angl.at<float>(i.x,i.y)) < rec.prec ) ++alg; 
    }
  }
   
  
  return nfa(pts,alg,rec.p,logNT);

}

/*--------------------------------------------------------------*/
/*                   Compute CLSR value                         */
/*--------------------------------------------------------------*/

float get_clsr(rect &rec, Mat tmp_dx, Mat tmp_dy, int length, int width)
{
  rect_iter i;
  int pts = 0;
  float* direction_max;
  direction_max = new float[length*width];
  float max;

  for (int j = 0; j < length*width; j++){
    direction_max[j] = 0.0;    
  }

  for( i=rect_ini(rec); !rect_end(i); rect_inc(i)){
    if( i.x >= 0 && i.y >= 0 && i.x < length && i.y < width ){ 
      if( fabs(tmp_dx.at<float>(i.x, i.y)) > fabs(tmp_dy.at<float>(i.x, i.y)) ) {
        // printf("%f\n", direction_max[pts]);
        direction_max[pts] = fabs(tmp_dx.at<float>(i.x, i.y)); 
      }
      else direction_max[pts] = fabs(tmp_dy.at<float>(i.x, i.y)); 
      pts++;
    }
  }
   
  max = direction_max[0];
  for( int j=1; j<pts; j++){
    if( direction_max[j]>max) max = direction_max[j];    
  }

  delete[] direction_max;
 
  return max;
  
}

/*---------------------------------------------------------------------------*/
/*    Try some rectangles variations to improve NFA value. Only if the       */
/*    rectangle is not meaningful (i.e., log_nfa <= eps).                    */
/*---------------------------------------------------------------------------*/

float rect_improve( rect &rec, Mat angl, float logNT, float eps, int length, int width )
{
  rect r;
  float log_nfa,log_nfa_new;
  float delta = 0.5;
  float delta_2 = delta / 2.0;
  int n;

  log_nfa = rect_nfa(rec, angl, logNT, length, width);

  if( log_nfa > eps ) return log_nfa;

  
  rect_copy(rec,r);
  for(n=0; n<5; n++){
    r.p /= 2.0;
    r.prec = r.p * PI;
    log_nfa_new = rect_nfa(r, angl, logNT, length, width);
    if( log_nfa_new > log_nfa ){
      log_nfa = log_nfa_new;
      rect_copy(r,rec);
    }
  }

  if( log_nfa > eps ) return log_nfa;

  rect_copy(rec,r);
  for(n=0; n<5; n++){
    if( (r.width - delta) >= 0.5 ){
      r.width -= delta;
      log_nfa_new = rect_nfa(r, angl, logNT, length, width);
      if( log_nfa_new > log_nfa ){
        rect_copy(r,rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if( log_nfa > eps ) return log_nfa;

  rect_copy(rec,r);
  for(n=0; n<5; n++){
    if( (r.width - delta) >= 0.5 ){
      r.x1 += -r.dy * delta_2;
      r.y1 +=  r.dx * delta_2;
      r.x2 += -r.dy * delta_2;
      r.y2 +=  r.dx * delta_2;
      r.width -= delta;
      log_nfa_new = rect_nfa(r, angl, logNT, length, width);
      if( log_nfa_new > log_nfa ){
        rect_copy(r,rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if( log_nfa > eps ) return log_nfa;

  rect_copy(rec,r);
  for(n=0; n<5; n++){
    if( (r.width - delta) >= 0.5 ){
      r.x1 -= -r.dy * delta_2;
      r.y1 -=  r.dx * delta_2;
      r.x2 -= -r.dy * delta_2;
      r.y2 -=  r.dx * delta_2;
      r.width -= delta;
      log_nfa_new = rect_nfa(r, angl, logNT, length, width);
      if( log_nfa_new > log_nfa ){
        rect_copy(r,rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if( log_nfa > eps ) return log_nfa;

  rect_copy(rec,r);
  for(n=0; n<5; n++){
    r.p /= 2.0;
    r.prec = r.p * PI;
    log_nfa_new = rect_nfa(r, angl, logNT, length, width);
    if( log_nfa_new > log_nfa ){
      log_nfa = log_nfa_new;
      rect_copy(r,rec);
    }
  }

  return log_nfa;
}

/*----------------------------------------------------------------------------*/
/*  Reduce the region size, by elimination the points far from the            */
/*  starting point, until that leads to rectangle with the right              */
/*  density of region points or to discard the region if too small.           */
/*--------------------------------------------------------------------------- */

int reduce_region_radius( point* reg, int &reg_size, Mat grad, float &reg_angle, 
                      float prec, float p, rect &rec, Mat label, Mat angl, float density_th )
{

  float density,rad1,rad2,rad,xc,yc;
  int i;

  density = reg_size / sqrt (pow(rec.x1-rec.x2,2) + pow(rec.y1-rec.y2,2)) * rec.width;

  if( density >= density_th ) return TRUE;

  xc = reg[0].x;
  yc = reg[0].y;
  rad1 = sqrt(pow(xc-rec.x1,2) + pow(yc-rec.y1,2));
  rad2 = sqrt(pow(xc-rec.x2,2) + pow(yc-rec.y2,2));
  rad = rad1 > rad2 ? rad1 : rad2;

  while( density < density_th ){
      rad *= 0.75;
      for(i=0; i<reg_size; i++){
        if( sqrt( pow(xc-reg[i].x,2) + pow(yc-reg[i].y,2)) > rad ){
            label.at<float>(reg[i].x, reg[i].y) = NOTUSED;
            reg[i].x = reg[reg_size-1].x; 
            reg[i].y = reg[reg_size-1].y;
            --reg_size;
            --i; 
        }
      }
      
      if( reg_size < 2 ) return FALSE;

      region2rect (reg, reg_size, grad, reg_angle, prec, rec, p); 

      density = reg_size / sqrt (pow(rec.x1-rec.x2,2) + pow(rec.y1-rec.y2,2)) * rec.width;                      
  }

  return TRUE;
}

/*----------------------------------------------------*/
/*          Signed angle difference.                  */         
/*--------------------------------------------------- */
float angle_diff_signed(float a, float b)
{
  a -= b;
  while( a <= -PI ) a += 2*PI;
  while( a >   PI ) a -= 2*PI;
  return a;
}


/*----------------------------------------------------*/
/*          Refine a rectangle.                       */
/*----------------------------------------------------*/

int refine ( point* reg, int &reg_size, Mat grad, float &reg_angle, float prec, 
                       float p, rect &rec, Mat label, Mat angl, float density_th, int length, int width)
{
  float density, xc, yc, mean_angle, new_angle, ang_c, ang_d, sum, s_sum;
  int i,n;
  
  density = reg_size / sqrt (pow(rec.x1-rec.x2,2) + pow(rec.y1-rec.y2,2)) * rec.width;
 
  if( density >= density_th ) return TRUE;

  xc = reg[0].x;
  yc = reg[0].y;
  ang_c = angl.at<float>(reg[0].x, reg[0].y);
  sum = s_sum = 0.0;
  n = 0;
  for(i=0; i<reg_size; i++){
    label.at<float>(reg[i].x, reg[i].y) = NOTUSED;
    if( sqrt( pow(xc-reg[i].x,2) + pow(yc-reg[i].y,2)) < rec.width ) {
      new_angle = angl.at<float>(reg[i].x, reg[i].y);
      ang_d = angle_diff_signed (new_angle, ang_c);
      sum += ang_d;
      s_sum += ang_d * ang_d;
      ++n;
    }
  }

  mean_angle = sum / n;

  float tau = 2.0 * sqrt( (s_sum - 2.0 * mean_angle * sum) /  n
                         + mean_angle * mean_angle ); 
  
  grow_region (grad, angl, label, reg[0].x, reg[0].y, reg, reg_size, reg_angle, tau, length, width);
  
  if( reg_size < 2 ) return FALSE;

  region2rect (reg, reg_size, grad, reg_angle, prec, rec, p);

  density = reg_size / sqrt (pow(rec.x1-rec.x2,2) + pow(rec.y1-rec.y2,2)) * rec.width;

  if( density < density_th )
    return reduce_region_radius( reg, reg_size, grad, reg_angle, prec, p, rec, label, angl, density_th );

  return TRUE;
}

/*-------------------------------------------------------------------------------------*/
/*        Compute PDF of Gaussian  provided by http://code.google.com/p/opencvx/       */
/*-------------------------------------------------------------------------------------*/
void cvMatGaussPdf( CvMat* samples, CvMat* mean, CvMat* cov, CvMat* probs, 
               bool normalize CV_DEFAULT(true), bool logprob CV_DEFAULT(false) )
{
    int D = samples->rows;
    int N = samples->cols;
    int type = samples->type;
    
    CvMat *invcov = cvCreateMat( D, D, type );
    cvInvert( cov, invcov, CV_SVD );

    CvMat *sample = cvCreateMat( D, 1, type );
    CvMat *subsample   = cvCreateMat( D, 1, type );
    CvMat *subsample_T = cvCreateMat( 1, D, type );
    CvMat *value       = cvCreateMat( 1, 1, type );

    double prob;
    for( int n = 0; n < N; n++ )
    {
        cvGetCol( samples, sample, n );

        cvSub( sample, mean, subsample );
        cvTranspose( subsample, subsample_T );
        cvMatMul( subsample_T, invcov, subsample_T );
        cvMatMul( subsample_T, subsample, value );
        prob = -0.5 * cvmGet(value, 0, 0);
        if( !logprob ) prob = exp( prob );

        cvmSet( probs, 0, n, prob );
    }
    if( normalize )
    {
        double norm = pow( 2* M_PI, D/2.0 ) * sqrt( cvDet( cov ) );
        if( logprob ) cvSubS( probs, cvScalar( log( norm ) ), probs );
        else cvConvertScale( probs, probs, 1.0 / norm );
    }
    
    cvReleaseMat( &invcov );
    cvReleaseMat( &sample );
    cvReleaseMat( &subsample );
    cvReleaseMat( &subsample_T );
    cvReleaseMat( &value );

}



float cvGaussPdf( CvMat* sample, CvMat* mean, CvMat* cov, 
            bool normalize CV_DEFAULT(true), bool logprob CV_DEFAULT(false) )
{
    float prob;
    CvMat *_probs  = cvCreateMat( 1, 1, sample->type );

    cvMatGaussPdf( sample, mean, cov, _probs, normalize, logprob );
    prob = cvmGet(_probs, 0, 0);

    cvReleaseMat( &_probs );
    return prob;
}


/*----------------------------------------------------------*/
/*        Estimate parametres of mixtured Gaussian          */
/*----------------------------------------------------------*/

void paramEstim (data* samples, data &mean0, data &mean1, 
                    Mat var0, Mat var1, float &p0, float &p1, float* label, int sum_nb )
{  

  //k-means
     
  mean0.x1 = -8,  mean0.x2 = 0.05;
  mean1.x1 = 18,  mean1.x2 = 0.01;
  
  data sum0, sum1;
  data l, k;

  int count0, count1;

  do{ 

     sum0.x1 = 0, sum0.x2 = 0; 
     sum1.x1 = 0, sum1.x2 = 0;
     count0 = 0 , count1 = 0;
     
     l = mean0 , k = mean1; 

     for(int i=0; i<sum_nb; i++){
        if(sqrt(pow(samples[i].x1-mean0.x1,2)+pow(samples[i].x2-mean0.x2,2))
            <sqrt(pow(samples[i].x1-mean1.x1,2)+pow(samples[i].x2-mean1.x2,2))){
            label[i] = 0;    
        }
        else{
            label[i] = 1;
        }
     }               
     
     for(int i=0; i<sum_nb; i++){
   if(label[i] == 0){
      sum0.x1 = sum0.x1 + samples[i].x1;
           sum0.x2 = sum0.x2 + samples[i].x2;
           count0++;
        }
        else{
           sum1.x1 = sum1.x1 + samples[i].x1;
           sum1.x2 = sum1.x2 + samples[i].x2;
           count1++;
        }         
     }         
  
     mean0.x1 = sum0.x1/count0;
     mean0.x2 = sum0.x2/count0;
     mean1.x1 = sum1.x1/count1;
     mean1.x2 = sum1.x2/count1;    
     
  }while(mean0.x1!=l.x1||mean0.x2!=l.x2||
            mean1.x1!=k.x1||mean1.x2!=k.x2);


  Mat temp0 = Mat(2,2,CV_32F); 
  Mat temp1 = Mat(2,2,CV_32F);

  for(int i=0; i<sum_nb; i++){
     if(label[i] == 0){
       temp0.at<float>(0,0) = samples[i].x1 - mean0.x1;
       temp0.at<float>(1,0) = samples[i].x2 - mean0.x2;
       temp0.at<float>(0,1) = 0;
       temp0.at<float>(1,1) = 0;
       var0 = var0 + temp0 * temp0.t();
     }else{
       temp1.at<float>(0,0) = samples[i].x1 - mean1.x1;
       temp1.at<float>(1,0) = samples[i].x2 - mean1.x2;
       temp1.at<float>(0,1) = 0;
       temp1.at<float>(1,1) = 0;
       var1 = var1 + temp1 * temp1.t();
     }
  }

  var0 = var0 / count0;
  var1 = var1 / count1;
 
  p0 = (float)count0 / (count0 + count1);
  p1 = 1-p0;


  //EM algorithem
  
  float karma_n0, karma_n1;
  float prob0, prob1, prob0new, prob1new;
  float sum, sum_new;  
  float n0,n1;
  float p0old, p1old;
 
  data mean0old, mean1old;
    
  Mat var0old;
  Mat var1old;
 
  do{
 
    mean0old = mean0;
    mean1old = mean1;
    var0old = var0;
    var1old = var1;
    p0old = p0;
    p1old = p1;

    sum = 0, sum_new = 0;
    n0 = 0, n1 = 0;
    sum0.x1 = 0, sum0.x2 = 0;
    sum1.x1 = 0, sum1.x2 = 0;    

    for(int n=0; n<sum_nb; n++){

       float v[] = { samples[n].x1, samples[n].x2 };
       float m0[] = { mean0old.x1, mean0old.x2 };
       float m1[] = { mean1old.x1, mean1old.x2 };
       float a0[] = { var0old.at<float>(0,0),var0old.at<float>(0,1), 
                        var0old.at<float>(1,0), var0old.at<float>(1,1) };
       float a1[] = { var1old.at<float>(0,0),var1old.at<float>(0,1), 
                        var1old.at<float>(1,0), var1old.at<float>(1,1) };
      
       CvMat vec = cvMat(2, 1, CV_32F, v);
       CvMat mean_0 = cvMat(2, 1, CV_32F, m0);
       CvMat mean_1 = cvMat(2, 1, CV_32F, m1);
       CvMat cov_0 = cvMat(2, 2, CV_32F, a0);
       CvMat cov_1 = cvMat(2, 2, CV_32F, a1);
       
       prob0 = cvGaussPdf( &vec, &mean_0, &cov_0);
       prob1 = cvGaussPdf( &vec, &mean_1, &cov_1);
          
       karma_n0 = p0old*prob0 / (p0old*prob0+p1old*prob1);
       karma_n1 = p1old*prob1 / (p0old*prob0+p1old*prob1);

       n0 = n0 + karma_n0;
       n1 = n1 + karma_n1;

       sum0.x1 = sum0.x1 + karma_n0 * samples[n].x1;
       sum0.x2 = sum0.x2 + karma_n0 * samples[n].x2;
       sum1.x1 = sum1.x1 + karma_n1 * samples[n].x1;
       sum1.x2 = sum1.x2 + karma_n1 * samples[n].x2;

    }
  
    mean0.x1 = sum0.x1 / n0;
    mean0.x2 = sum0.x2 / n0;
    mean1.x1 = sum1.x1 / n1;
    mean1.x2 = sum1.x2 / n1;

    Mat var0new = Mat::zeros(2,2,CV_32F);
    Mat var1new = Mat::zeros(2,2,CV_32F);

    for(int n=0; n<sum_nb; n++){

       float v[] = { samples[n].x1, samples[n].x2 };
       float m0[] = { mean0old.x1, mean0old.x2 };
       float m1[] = { mean1old.x1, mean1old.x2 };
       float a0[] = { var0old.at<float>(0,0),var0old.at<float>(0,1), 
                        var0old.at<float>(1,0), var0old.at<float>(1,1) };
       float a1[] = { var1old.at<float>(0,0),var1old.at<float>(0,1), 
                        var1old.at<float>(1,0), var1old.at<float>(1,1) };

       CvMat vec  = cvMat(2, 1, CV_32F, v);
       CvMat mean_0 = cvMat(2, 1, CV_32F, m0);
       CvMat mean_1 = cvMat(2, 1, CV_32F, m1);
       CvMat cov_0  = cvMat(2, 2, CV_32F, a0);
       CvMat cov_1  = cvMat(2, 2, CV_32F, a1);

       prob0 = cvGaussPdf( &vec, &mean_0, &cov_0);
       prob1 = cvGaussPdf( &vec, &mean_1, &cov_1);
        
       karma_n0 = p0old*prob0 / (p0old*prob0+p1old*prob1);
       karma_n1 = p1old*prob1 / (p0old*prob0+p1old*prob1);

       temp0.at<float>(0,0) = samples[n].x1 - mean0.x1;
       temp0.at<float>(1,0) = samples[n].x2 - mean0.x2;
       temp0.at<float>(0,1) = 0;
       temp0.at<float>(1,1) = 0;

       temp1.at<float>(0,0) = samples[n].x1 - mean1.x1;
       temp1.at<float>(1,0) = samples[n].x2 - mean1.x2;
       temp1.at<float>(0,1) = 0;
       temp1.at<float>(1,1) = 0;
   
       var0new = var0new + karma_n0*temp0*temp0.t();
       var1new = var1new + karma_n1*temp1*temp1.t();
   
    }

    var0 = var0new / n0;
    var1 = var1new / n1;

    p0 = n0 / (n0+n1);
    p1 = n1 / (n0+n1);


    for(int n=0; n<sum_nb; n++){
 
      float v[] = { samples[n].x1, samples[n].x2 };
      float m0old[] = { mean0old.x1, mean0old.x2 };
      float m1old[] = { mean1old.x1, mean1old.x2 };
      float a0old[] = { var0old.at<float>(0,0),var0old.at<float>(0,1), 
                           var0old.at<float>(1,0), var0old.at<float>(1,1) };
      float a1old[] = { var1old.at<float>(0,0),var1old.at<float>(0,1), 
                           var1old.at<float>(1,0), var1old.at<float>(1,1) };

      CvMat vec = cvMat(2, 1, CV_32F, v);
      CvMat mean_0_old = cvMat(2, 1, CV_32F, m0old);
      CvMat mean_1_old = cvMat(2, 1, CV_32F, m1old);
      CvMat cov_0_old  = cvMat(2, 2, CV_32F, a0old);
      CvMat cov_1_old  = cvMat(2, 2, CV_32F, a1old);

      prob0 = cvGaussPdf( &vec, &mean_0_old, &cov_0_old);
      prob1 = cvGaussPdf( &vec, &mean_1_old, &cov_1_old);

      float m0[] = { mean0.x1, mean0.x2 };
      float m1[] = { mean1.x1, mean1.x2 };
      float a0[] = { var0.at<float>(0,0),var0.at<float>(0,1), 
                          var0.at<float>(1,0), var0.at<float>(1,1) };
      float a1[] = { var1.at<float>(0,0),var1.at<float>(0,1), 
                          var1.at<float>(1,0), var1.at<float>(1,1) };

      CvMat mean_0 = cvMat(2, 1, CV_32F, m0);
      CvMat mean_1 = cvMat(2, 1, CV_32F, m1);
      CvMat cov_0  = cvMat(2, 2, CV_32F, a0);
      CvMat cov_1  = cvMat(2, 2, CV_32F, a1);

      prob0new = cvGaussPdf( &vec, &mean_0, &cov_0);
      prob1new = cvGaussPdf( &vec, &mean_1, &cov_1);
       
      sum = sum + log(p0old*prob0 + p1old*prob1);
      sum_new = sum_new + log(p0*prob0new + p1*prob1new);
       
    }  
    
    
  }while(sum!=sum_new);
   
}


/*----------------------------------------------------------*/
/*         Obtain training data from input images           */
/*----------------------------------------------------------*/
void get_training(data &mean0, data &mean1, Mat var0, Mat var1, float &p0, float &p1)
{
   int selectedFrames[100];
   char fname[260];
   int nb_frame = 42;
   CvMat* pData = cvCreateMat(nb_frame, 4, CV_32F);
   
   for(int i=0;i<100;i++) selectedFrames[i]=i;    

   // printf("hello world");
   for(int k=1; k<nb_frame+1; k++){

      sprintf(fname,"train/%03d.png",selectedFrames[k]); 
      printf("%s\n",fname);
      IplImage *img =  cvLoadImage( fname, 0 );
      IplImage *img_blur = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
      /*-------------------------------------*/
 
      int length = img->height;
      int width = img->width;
      cvSmooth( img, img_blur, CV_BLUR, 2, 2);
      
      CvMat *dx = cvCreateMat( length, width, CV_32F);
      CvMat *dy = cvCreateMat( length, width, CV_32F);

      cvSobel(img_blur, dx, 0, 1, 3);
      cvSobel(img_blur, dy, 1, 0, 3);
      
      Mat tmp_dx(dx);
      Mat tmp_dy(dy);
      Mat grad(length, width, CV_32F);
      Mat angl(length, width, CV_32F);

      float ang_th = 22.5; /*Gradient angle tolerance in the region growing algorithm, 
                             Suggested value: 22.5*/
      float prec = PI * ang_th / 180.0;
      float rho = 5 / sin(prec); /*Bound to the quantization error on the gradient norm.
                                Example: if gray level is quantized to integer steps,
                                the gradient (computed by finite differences) error
                                due to quantization will be bounded by 2.0, as the
                                worst case is when the error are 1 and -1, that gives 
                                an error of 2.0. In addition,considering that small gradient 
                                usually represents flat zone or slow gray level transition
                                that could be ignored. Suggested value: 5.0*/

      for(int i=0; i<length; i++){
         for(int j=0; j<width; j++){
            grad.at<float>(i,j) = sqrt( pow(tmp_dx.at<float>(i,j),2) + pow(tmp_dy.at<float>(i,j),2));
            angl.at<float>(i,j) = atan2( tmp_dy.at<float>(i,j), tmp_dx.at<float>(i,j) );
            if(grad.at<float>(i,j)<=rho){
               grad.at<float>(i,j) = 0.0;
               angl.at<float>(i,j) = 0.0;
            }
         }
      } 
      
      recal(grad, length, width);


      /*---------------INITIALIZE LABEL-------------------*/
      Mat label(length, width, CV_32F);

      for (int i=0; i<length; i++) {
         for (int j=0; j<width; j++) {
            label.at<int>(i,j) = NOTUSED;
         }
      }

  
      /*----------------SORT NORM-----------------------*/
      norm_list list[length*width];
      sort(grad, list, length, width);
 


      /*---------------SUPPORT REGION-------------------*/
      int reg_size;
      float reg_angle;
      float log_nfa;
      float thickness;
      int min_reg_size;
      point *reg;
      rect rec;
      float density_th = 0.7; /*Minimal proportion of region points in a rectangle.
                             Suggested value: 0.7*/

      float eps = 0.0; /* Detection threshold, -log10(NFA).
                       The bigger, the more strict the detector is,
                       and will result in less detections.
                       The value -log10(NFA) is equivalent but more
                       intuitive than NFA:
                       - -1.0 corresponds to 10 mean false alarms
                       -  0.0 corresponds to 1 mean false alarm
                       -  1.0 corresponds to 0.1 mean false alarms
                       -  2.0 corresponds to 0.01 mean false alarms
                       .
                       Suggested value: 0.0*/

      float logNT = 5.0 * (log10(length) + log10(width)) / 2.0;
      int ls_count = 0;
      float p = ang_th / 180.0;
      float line_length[length*width];
      float line_contrast[length*width];
      float sum_length = 0.0;
      float sum_contrast = 0.0;
      float mean_length;
      float mean_contrast;
      int bins_length = 38;
      int bins_contrast = 31;
      float entropy_table_length[bins_length];
      float entropy_table_contrast[bins_contrast];
   
      min_reg_size = (int) (-logNT/log10(p)); 
      reg = (point *) calloc ((size_t)(length * width), sizeof(point));
      IplImage *Line = cvCreateImage(cvSize(length, width), IPL_DEPTH_8U, 3);

      for(int i=0; i<bins_length; i++) entropy_table_length[i] = 0.0;
      for(int i=0; i<bins_contrast; i++) entropy_table_contrast[i] = 0.0;
     
      for (int i=length*width-1; i>0; i--){
        if ( label.at<float>(int(list[i].x), int(list[i].y)) == NOTUSED && grad.at<float>(int(list[i].x),int(list[i].y))!=0){
          int x = list[i].x;
          int y = list[i].y;
          grow_region (grad, angl, label, x, y, reg, reg_size, reg_angle, prec, length, width);
          if( reg_size >= min_reg_size ){
            region2rect (reg, reg_size, grad, reg_angle, prec, rec, p);      
            if( refine (reg, reg_size, grad, reg_angle, prec, p, rec, label, angl, density_th, length, width )){ 
              log_nfa = rect_improve( rec, angl, logNT, eps, length, width);
              if( log_nfa <= eps && rec.width > 2){
                ++ls_count;
                cvLine (Line, cvPoint(rec.x1, rec.y1), cvPoint(rec.x2, rec.y2), cvScalar(255,255,255,255), 2, 8, 0);
                line_length[ls_count] = sqrt (pow((int)rec.x1 - (int)rec.x2, 2) + pow((int)rec.y1 - (int)rec.y2, 2)); 
                if (line_length[ls_count] > 150) line_length[ls_count] = 0.0;
                else sum_length += line_length[ls_count];
                entropy_table_length[(int)ceil(line_length[ls_count]/4)]++;     
                line_contrast[ls_count] = get_clsr(rec, tmp_dx, tmp_dy, length, width);   
                if (line_contrast[ls_count] > 3000) line_contrast[ls_count] = 0.0;
                else sum_contrast += line_contrast[ls_count];
                entropy_table_contrast[(int)ceil(line_contrast[ls_count]/95)]++;   
              }
            }
          }  
        }
      }
    
      mean_length = sum_length / ls_count;
      mean_contrast = sum_contrast / ls_count;
      
      float normalized_entropy_table_length[bins_length];
      float normalized_entropy_table_contrast[bins_contrast];
      float sum_entropy_length = 0;  
      float sum_entropy_contrast = 0;  
      float tmp_length = 0;
      float tmp_contrast = 0;
     
      for(int i=0; i<bins_length; i++) tmp_length += pow( entropy_table_length[i], 2 );
      for(int i=0; i<bins_contrast; i++) tmp_contrast += pow( entropy_table_contrast[i], 2 );
     
      for(int i=0; i<bins_length; i++) normalized_entropy_table_length[i] = (entropy_table_length[i] / sqrt (tmp_length))+1;
      for(int i=0; i<bins_contrast; i++) normalized_entropy_table_contrast[i] = (entropy_table_contrast[i] / sqrt (tmp_contrast))+1;

      for(int i=0; i<bins_length; i++) sum_entropy_length -= normalized_entropy_table_length[i]*log(normalized_entropy_table_length[i]);
      for(int i=0; i<bins_contrast; i++) sum_entropy_contrast -= normalized_entropy_table_contrast[i]*log(normalized_entropy_table_contrast[i]);
     
      pData->data.fl[(k-1)*pData->cols + 0] = mean_length;
      pData->data.fl[(k-1)*pData->cols + 1] = sum_entropy_length;
      pData->data.fl[(k-1)*pData->cols + 2] = mean_contrast;
      pData->data.fl[(k-1)*pData->cols + 3] = sum_entropy_contrast;
      
       
      printf("%d\t%f\t%f\t%f\t%f\n", k, pData->data.fl[(k-1)*pData->cols + 0], pData->data.fl[(k-1)*pData->cols + 1], 
                                                 pData->data.fl[(k-1)*pData->cols + 2], pData->data.fl[(k-1)*pData->cols + 3]);
    }

   //PCA 

   CvMat* pMean = cvCreateMat(1, 4, CV_32F);
   CvMat* pEigVals = cvCreateMat(1, min(nb_frame,4), CV_32F);
   CvMat* pEigVecs = cvCreateMat(min(nb_frame,4), 4, CV_32F);

   cvCalcPCA(pData, pMean, pEigVals, pEigVecs, CV_PCA_DATA_AS_ROW);
   
   CvMat* pResult = cvCreateMat(nb_frame, 2, CV_32F);
   cvProjectPCA(pData, pMean, pEigVecs, pResult);
  
   data* samples;
 
   samples = (data *) calloc ((size_t)(nb_frame), sizeof(data));

   for(int i=0; i<nb_frame; i++) {
      samples[i].x1 = pResult->data.fl[(i)*pResult->cols + 0];
      samples[i].x2 = pResult->data.fl[(i)*pResult->cols + 1];
   }

   float label[nb_frame]; 

   //EM parameters estimation
   paramEstim(samples, mean0, mean1, var0, var1, p0, p1, label, nb_frame); 

   printf("-----------------------Two Mixtured Gaussian----------------------\n");
   printf("mean0 = %f\t%f\n", mean0.x1, mean0.x2);
   printf("mean1 = %f\t%f\n", mean1.x1, mean1.x2);
   
   printf("var0 = %f\t%f\t%f\t%f\n",var0.at<float>(0,0), var0.at<float>(0,1), var0.at<float>(1,0), var0.at<float>(1,1));
   printf("var1 = %f\t%f\t%f\t%f\n",var1.at<float>(0,0), var1.at<float>(0,1), var1.at<float>(1,0), var1.at<float>(1,1));
  
   printf("p0 = %f\n", p0);
   printf("p1 = %f\n", p1);
 
}


/*---------------------------------------------------------*/
/*        Give a label to each tested image                */
/*---------------------------------------------------------*/
void predict(data* samples, data &mean0, data &mean1, Mat var0, Mat var1, float &p0, float &p1,  int nb_frame)
{
   
   
   float mean0param[] = { mean0.x1, mean0.x2 };
   float mean1param[] = { mean1.x1, mean1.x2 };
   CvMat vec_mean0 = cvMat(2, 1, CV_32F, mean0param);
   CvMat vec_mean1 = cvMat(2, 1, CV_32F, mean1param);

   float var0param[] = { var0.at<float>(0,0), var0.at<float>(0,1), var0.at<float>(1,0), var0.at<float>(1,1)};
   float var1param[] = { var1.at<float>(0,0), var1.at<float>(0,1), var1.at<float>(1,0), var1.at<float>(1,1)};
   CvMat mat_var0 = cvMat(2, 2, CV_32F, var0param);
   CvMat mat_var1 = cvMat(2, 2, CV_32F, var1param);

    
   for(int i=0; i<nb_frame; i++){

      float prob0, prob1;
      float test[] = {samples[i].x1, samples[i].x2 };   
      CvMat vec_test = cvMat(2, 1, CV_32F, test);

      prob0 = cvGaussPdf( &vec_test, &vec_mean0, &mat_var0, false);
      prob1 = cvGaussPdf( &vec_test, &vec_mean1, &mat_var1, false);
       
      
      if(p0*prob0 > p1*prob1) printf("%d\trural\n",i+1);
      else printf("%d\turban\n",i+1);
   }
}

/*---------------------------------------------------*/
/*        Classify tested images                     */
/*---------------------------------------------------*/
void get_classifier( data &mean0, data &mean1, Mat var0, Mat var1, float &p0, float &p1)
{
   int selectedFrames[100];
   char fname[260];
   int nb_frame = 20;
   CvMat* pData = cvCreateMat(nb_frame, 4, CV_32F);

   for(int i=0;i<100;i++) selectedFrames[i]=i; 

   for(int k=1; k<nb_frame+1; k++){

     sprintf(fname,"test/%03d.png",selectedFrames[k]); 
     // printf("%s\n", fname);
     IplImage *img =  cvLoadImage( fname, 0 );
     IplImage *img_blur = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
 
     int length = img->height;
     int width = img->width;

     cvSmooth( img, img_blur, CV_BLUR, 2, 2);
   
     CvMat *dx = cvCreateMat( length, width, CV_32F);
     CvMat *dy = cvCreateMat( length, width, CV_32F);

     cvSobel(img_blur, dx, 0, 1, 3);
     cvSobel(img_blur, dy, 1, 0, 3);
   
     Mat tmp_dx(dx);
     Mat tmp_dy(dy);
     Mat grad(length, width, CV_32F);
     Mat angl(length, width, CV_32F);

     float ang_th = 22.5;
     float prec = PI * ang_th / 180.0;
     float rho = 5 / sin(prec); 

     for(int i=0; i<length; i++){
       for(int j=0; j<width; j++){
         grad.at<float>(i,j) = sqrt( pow(tmp_dx.at<float>(i,j),2) + pow(tmp_dy.at<float>(i,j),2));
         angl.at<float>(i,j) = atan2( tmp_dy.at<float>(i,j), tmp_dx.at<float>(i,j) );
         if(grad.at<float>(i,j)<=rho){
           grad.at<float>(i,j) = 0.0;
           angl.at<float>(i,j) = 0.0;
         }
       }
     } 
   
     recal(grad, length, width);


     //INITIALIZE LABEL
     Mat label(length, width, CV_32F);

     for (int i=0; i<length; i++) {
       for (int j=0; j<width; j++) {
         label.at<int>(i,j) = NOTUSED;
       }
     }

  
     //SORT NORM
     norm_list list[length*width];
     sort(grad, list, length, width);
    
  
     //SUPPORT REGION
     int reg_size;
     float reg_angle;
     float log_nfa;
     float thickness;
     int min_reg_size;
     point *reg;
     rect rec;
     float density_th = 0.7, eps = 0.0;
     float logNT = 5.0 * (log10(length) + log10(width)) / 2.0;
     int ls_count = 0;
     float p = ang_th / 180.0;
     float line_length[length*width];
     float line_contrast[length*width];
     float sum_length = 0.0;
     float sum_contrast = 0.0;
     float mean_length;
     float mean_contrast;
     int bins_length = 38;
     int bins_contrast = 31;
     float entropy_table_length[bins_length];
     float entropy_table_contrast[bins_contrast];
   
     min_reg_size = (int) (-logNT/log10(p)); 
     reg = (point *) calloc ((size_t)(length * width), sizeof(point));
     IplImage *Line = cvCreateImage(cvSize(length, width), IPL_DEPTH_8U, 3);

     for(int i=0; i<bins_length; i++) entropy_table_length[i] = 0.0;
     for(int i=0; i<bins_contrast; i++) entropy_table_contrast[i] = 0.0;
   
     for (int i=length*width-1; i>0; i--){
       if ( label.at<float>(int(list[i].x), int(list[i].y)) == NOTUSED && grad.at<float>(int(list[i].x),int(list[i].y))!=0){
          int x = list[i].x;
          int y = list[i].y;
         grow_region (grad, angl, label, x, y, reg, reg_size, reg_angle, prec, length, width);
         if( reg_size >= min_reg_size ){
           region2rect (reg, reg_size, grad, reg_angle, prec, rec, p);      
           if( refine (reg, reg_size, grad, reg_angle, prec, p, rec, label, angl, density_th, length, width )){ 
             log_nfa = rect_improve( rec, angl, logNT, eps, length, width);
             if( log_nfa <= eps && rec.width > 2){
               ++ls_count;
               line_length[ls_count] = sqrt (pow((int)rec.x1 - (int)rec.x2, 2) + pow((int)rec.y1 - (int)rec.y2, 2)); 
               if (line_length[ls_count] > 150) line_length[ls_count] = 0.0;
               else sum_length += line_length[ls_count];
               entropy_table_length[(int)ceil(line_length[ls_count]/4)]++;      
               line_contrast[ls_count] = get_clsr(rec, tmp_dx, tmp_dy, length, width);   
               if (line_contrast[ls_count] > 3000) line_contrast[ls_count] = 0.0;
               else sum_contrast += line_contrast[ls_count];
               entropy_table_contrast[(int)ceil(line_contrast[ls_count]/95)]++;   
             }
           }
         }  
       }
     }
    
     mean_length = sum_length / ls_count;
     mean_contrast = sum_contrast / ls_count;
    
     float normalized_entropy_table_length[bins_length];
     float normalized_entropy_table_contrast[bins_contrast];
     float sum_entropy_length = 0;  
     float sum_entropy_contrast = 0;  
     float tmp_length = 0;
     float tmp_contrast = 0;
   
     for(int i=0; i<bins_length; i++) tmp_length += pow( entropy_table_length[i], 2 );
     for(int i=0; i<bins_contrast; i++) tmp_contrast += pow( entropy_table_contrast[i], 2 );
   
     for(int i=0; i<bins_length; i++) normalized_entropy_table_length[i] = (entropy_table_length[i] / sqrt (tmp_length))+1;
     for(int i=0; i<bins_contrast; i++) normalized_entropy_table_contrast[i] = (entropy_table_contrast[i] / sqrt (tmp_contrast))+1;

     for(int i=0; i<bins_length; i++) sum_entropy_length -= normalized_entropy_table_length[i]*log(normalized_entropy_table_length[i]);
     for(int i=0; i<bins_contrast; i++) sum_entropy_contrast -= normalized_entropy_table_contrast[i]*log(normalized_entropy_table_contrast[i]);
   
     pData->data.fl[(k-1)*pData->cols + 0] = mean_length;
     pData->data.fl[(k-1)*pData->cols + 1] = sum_entropy_length;
     pData->data.fl[(k-1)*pData->cols + 2] = mean_contrast;
     pData->data.fl[(k-1)*pData->cols + 3] = sum_entropy_contrast;
    
   }

   CvMat* pMean = cvCreateMat(1, 4, CV_32F);
   CvMat* pEigVals = cvCreateMat(1, min(nb_frame,4), CV_32F);
   CvMat* pEigVecs = cvCreateMat(min(nb_frame,4), 4, CV_32F);

   cvCalcPCA(pData, pMean, pEigVals, pEigVecs, CV_PCA_DATA_AS_ROW);
   
   CvMat* pResult = cvCreateMat(nb_frame, 2, CV_32F);
   cvProjectPCA(pData, pMean, pEigVecs, pResult);

   data* samples;
 
   samples = (data *) calloc ((size_t)(nb_frame), sizeof(data));

   for(int i=0; i<nb_frame; i++) {
      samples[i].x1 = pResult->data.fl[(i)*pResult->cols + 0];
      samples[i].x2 = pResult->data.fl[(i)*pResult->cols + 1];
   }

   predict(samples, mean0, mean1, var0, var1, p0, p1, nb_frame);



}





