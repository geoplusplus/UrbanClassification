#ifndef __func__
#define __func__

#include "const.h"
// #include "opencv2/core/core.hpp"
// #include "opencv2/ml/ml.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include "opencv2/opencv.hpp"
// #include "opencv/cvaux.h"

using namespace cv;

/*-------------------------------------------------------*/
/*                     Recal                             */
/*-------------------------------------------------------*/
void recal(Mat mat,int lgth,int wdth);


/*---------------------------------------------------*/
/*                                                   */
/*---------------------------------------------------*/
void sort(Mat grad, norm_list list[], int length, int width);



/*--------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.     */
/*--------------------------------------------------------------------------*/
float get_theta( point* reg, int &reg_size, int x, int y, 
							Mat grad, float &reg_angle, float prec);


/*----------------------------------------------------------*/
/**Computes a rectangle that covers a region of points.     */
/*--------------------------------------------------------- */
void region2rect( point* reg, int &reg_size, Mat grad, float &reg_angle, 
											float prec, rect &rec, float p);


/*--------------------------------------------------------------------------------*/
/* find regions where adjacent pixels share the common gradient orientation       */
/*--------------------------------------------------------------------------------*/
void grow_region(Mat grad, Mat angl, Mat label, int x, int y, point* reg, 
                          	int &reg_size, float &reg_angle, float prec, int length, int width);


/*----------------------------------------------------------------------------*/
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

float log_gamma_lanczos(float x);



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
float log_gamma_windschitl(float x);



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
float nfa(int n, int k, float p, float logNT);


/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
float rect_low(float x, float x1, float y1, float x2, float y2);


/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
float rect_hi(float x, float x1, float y1, float x2, float y2);


/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
int rect_end (rect_iter &i);


/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
void rect_inc (rect_iter &i);


/*-----------------------------------------------------------------*/
/*                                                                 */
/*-----------------------------------------------------------------*/
rect_iter rect_ini (rect &r);


/*--------------------------------------------------------------------------*/
/*            Copy one rectangle structure to another.                      */
/*--------------------------------------------------------------------------*/
void rect_copy(rect &in, rect &out);


/*--------------------------------------------------------------*/
/*               Compute a rectangle's NFA value.               */
/*--------------------------------------------------------------*/
float rect_nfa(rect &rec, Mat angl, float logNT, int length, int width);


/*--------------------------------------------------------------*/
/*                   Compute CLSR value                         */
/*--------------------------------------------------------------*/
float get_clsr(rect &rec, Mat tmp_dx, Mat tmp_dy, int length, int width);



/*---------------------------------------------------------------------------*/
/*    Try some rectangles variations to improve NFA value. Only if the       */
/*    rectangle is not meaningful (i.e., log_nfa <= eps).                    */
/*---------------------------------------------------------------------------*/
float rect_improve(rect &rec, Mat angl, float logNT, float eps, int length, int width );


/*----------------------------------------------------------------------------*/
/*  Reduce the region size, by elimination the points far from the            */
/*  starting point, until that leads to rectangle with the right              */
/*  density of region points or to discard the region if too small.           */
/*--------------------------------------------------------------------------- */
int reduce_region_radius(point* reg, int &reg_size, Mat grad, float &reg_angle, 
                      float prec, float p, rect &rec, Mat label, Mat angl, float density_th );


/*----------------------------------------------------*/
/*          Signed angle difference.                  */         
/*--------------------------------------------------- */
float angle_diff_signed(float a, float b);


/*----------------------------------------------------*/
/*          Refine a rectangle.                       */
/*----------------------------------------------------*/
int refine ( point* reg, int &reg_size, Mat grad, float &reg_angle, float prec, 
               float p, rect &rec, Mat label, Mat angl, float density_th, int length, int width);


/*-------------------------------------------------------------------------------------*/
/*        Compute PDF of Gaussian  provided by http://code.google.com/p/opencvx/       */
/*-------------------------------------------------------------------------------------*/
void cvMatGaussPdf( CvMat* samples, CvMat* mean, CvMat* cov, CvMat* probs, 
               bool normalize, bool logprob);

// void cvMatGaussPdf( CvMat* samples, CvMat* mean, CvMat* cov, CvMat* probs, 
//                bool normalize CV_DEFAULT(true), bool logprob CV_DEFAULT(false) );

float cvGaussPdf( CvMat* sample, CvMat* mean, CvMat* cov, 
            bool normalize, bool logprob);


// float cvGaussPdf( CvMat* sample, CvMat* mean, CvMat* cov, 
//             bool normalize CV_DEFAULT(true), bool logprob CV_DEFAULT(false) );



/*----------------------------------------------------------*/
/*        Estimate parametres of mixtured Gaussian          */
/*----------------------------------------------------------*/
void paramEstim ( data* samples, data &mean0, data &mean1, 
                    Mat var0, Mat var1, float &p0, float &p1, float* label, int sum_nb );



/*----------------------------------------------------------*/
/*         Obtain training data from input images           */
/*----------------------------------------------------------*/
void get_training(data &mean0, data &mean1, Mat var0, Mat var1, float &p0, float &p1);



/*---------------------------------------------------------*/
/*        Give a label to each tested image                */
/*---------------------------------------------------------*/
void predict(data* samples, data &mean0, data &mean1, 
						Mat var0, Mat var1, float &p0, float &p1,  int nb_frame);


/*---------------------------------------------------*/
/*        Classify tested images                     */
/*---------------------------------------------------*/
void get_classifier( data &mean0, data &mean1, Mat var0, Mat var1, float &p0, float &p1);



#endif