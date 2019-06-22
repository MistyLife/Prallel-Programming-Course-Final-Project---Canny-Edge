/** @file
  
  Final Project of Parallel Programming Course

  Date: 2019/06/17
  
  Title: Canny Edge speedup by CUDA parallel - serial version.

  Note: 
        (1) Boundary computing not verify, it might has some errors.

**/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>

//
// Debug Usage
//
#define CV_PRINT              0
#define OUTPUT_EACH_IMG       0

#if CV_PRINT
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//using namespace std;
using namespace cv;
#endif

//
// Image Setting
//
#define IMAGE_WIDTH           512
#define IMAGE_HEIGHT          512
#define IMAGE_SIZE            (IMAGE_WIDTH * IMAGE_HEIGHT)

// Gaussian blur parameters
#define SIGMA                 3
#define K_SIZE                11

// Double threshold parameters
#define STRONG_EDGE_PIXEL     255
#define WEAK_EDGE_PIXEL       127
#define HIGH_TH               30
#define LOW_TH                20

// Fixed parameter and formula
#define CONV_LEN              3
#define PI                    3.1415926
#define BILLION               1E9

#define GAUSSIAN_2D(x,y)      exp(-0.5 * (x * x + y * y) / (SIGMA * SIGMA)) / (2 * PI * SIGMA * SIGMA);
#define OUT_OF_BOUNDARY(x,y)  (x < 0 || x >= IMAGE_WIDTH || y < 0 || y >= IMAGE_HEIGHT)


//
// Emulation
//
enum _ANGLE {
  DEG_0   = 0,
  DEG_45  = 63,
  DEG_90  = 127,
  DEG_135 = 191
};

enum _TIME_COST {
  TIME_GAUSSIAN_BLUR,
  TIME_SOBEL_FILTER,
  TIME_NON_MAX_SUP,
  TIME_DOUBLE_TH,
  TIME_TRACK,
  TIME_MAX
};


//
// Structure
//
int sobel_dx[3][3]=
{
  -1,  0,  1,
  -2,  0,  2,
  -1,  0,  1,
};

int sobel_dy[3][3]=
{
  -1, -2, -1,
   0,  0,  0,
   1,  2,  1,
};


/**
  Use to get current time. Input it into end_time_ms() to compute the elasped time.

  @param[out] start      Current time.
  
**/
void start_time(struct timespec *start)
{
  clock_gettime(CLOCK_REALTIME, start);
}


/**
  Get the elasped time by compute the difference from start to current time.

  @param[in] start       Start time for compute.
  
  @retval elapsed time   Elapsed time.
  
**/
float end_time_ms(struct timespec start)
{
  struct timespec end, temp;  

  clock_gettime(CLOCK_REALTIME, &end);

  if ((end.tv_nsec - start.tv_nsec) < 0) { 
    temp.tv_sec = end.tv_sec - start.tv_sec - 1; 
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec; 
  } else { 
    temp.tv_sec = end.tv_sec - start.tv_sec; 
    temp.tv_nsec = end.tv_nsec - start.tv_nsec; 
  } 

  return ((float)(temp.tv_sec) + ((float)(temp.tv_nsec) / (float) BILLION * 1E3));
}


#if CV_PRINT
/**
  Print image in window. (by opencv)

  @param[in] image         Image to print
  @param[in] windows_name  Name of window.
  
**/
void cv_print_img(unsigned char *img, char const *window_name)
{
  Mat cv_img;

  cv_img.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
  memcpy(cv_img.data, img, IMAGE_SIZE);
  namedWindow(window_name, WINDOW_AUTOSIZE);
  imshow(window_name, cv_img);
}
#endif


/**
  Use Gaussian function to calculating each pixel with its around pixel.
  It used to reduce image noise and detail.

  Ref: https://en.wikipedia.org/wiki/Gaussian_blur

  @param[in]  image       Gray scale image.
  @param[out] result_img  Image output by Gaussian blur.

**/
void gassian_blur(unsigned char *image, unsigned char *result_img)
{
  int   center, idx, px, py;
  float *param, sum;
  float  x, y, fx;

  center = K_SIZE / 2;
  param = (float *) malloc(K_SIZE * K_SIZE * sizeof(float));

  // Compulate gaussian kernel matrix
  sum = 0;
  for (int col = 0; col < K_SIZE; col++){
    for (int row = 0; row < K_SIZE; row++){
      x = (float)(col - center);
      y = (float)(row - center);
      idx = col + row * K_SIZE;
      
      fx = GAUSSIAN_2D(x, y);
      
      param[idx] = fx;
      sum += fx;
    }
  }

  for (int col = 0; col < K_SIZE; col++){
    for (int row = 0; row < K_SIZE; row++){
      idx = col + row * K_SIZE;

      param[idx] /= sum;
    }
  }

  // Blur each pixel
  for (int x = 0; x < IMAGE_WIDTH; x++) {
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
      sum = 0;
      for (int gx = 0; gx < K_SIZE; gx++) {
        for (int gy = 0; gy < K_SIZE; gy++) {
          px = x - center + gx;
          py = y - center + gy;

          if (!(OUT_OF_BOUNDARY(px, py))) {
            sum += (float)((unsigned int)image[px + py * IMAGE_WIDTH] * param[gx + gy * K_SIZE]);
          }
        }
      }

      result_img[x + y * IMAGE_WIDTH] = (unsigned char)(sum + 0.5);  // Round up
    }
  }
}


/**
  Get image edge by Sobel operator.
  
  The operator uses two 3×3 kernels which are convolved with the original image 
  to calculate approximations of the derivatives – one for horizontal changes, 
  and one for vertical.

  Ref: https://en.wikipedia.org/wiki/Sobel_operator

  @param[in]  image           Gray scale image.
  @param[out] gradient        Gradient in each pixel. (Sobel edge result)
  @param[out] gradient_angle  Gradient angle in each pixel, separate with diff step.
                              Each step emulate at _ANGLE.

**/
void sobel_filter(unsigned char *image, unsigned char *gradient, unsigned char *gradient_angle)
{
  unsigned int   theta;
  unsigned int   img_idx, conv_idx;
  double         gx, gy;
    
  for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
    for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
      gx = 0;
      gy = 0;

      img_idx = x + y * IMAGE_WIDTH;
      for (int dx = 0; dx < CONV_LEN; dx++) {
        for (int dy = 0; dy < CONV_LEN; dy++) {
          conv_idx = (img_idx - CONV_LEN / 2 + dx) + (CONV_LEN / 2 + dy) * IMAGE_WIDTH;

          gx += image[conv_idx] * sobel_dx[dx][dy];
          gy += image[conv_idx] * sobel_dy[dx][dy];
        }
      }
      
      gradient[img_idx] = (unsigned int)(sqrt(gx * gx + gy * gy) + 0.5);  // Round up
      
      theta = atan(gy / gx) * 180 / PI + 0.5;  // Round up
      theta %= 180;
      
			if (theta <= 22.5 || theta > 157.5) {
        gradient_angle[img_idx] = (unsigned int)DEG_0;
      } else if (theta <= 67.5) {
        gradient_angle[img_idx] = (unsigned int)DEG_45;
      } else if (theta <= 112.5) {
        gradient_angle[img_idx] = (unsigned int)DEG_90;
      } else if (theta <= 157.5) {
        gradient_angle[img_idx] = (unsigned int)DEG_135;
      } else {
        printf("*********************************\n");
        printf("Exeception occured while assiging gradient_angle @ sobel_filter() theta %d\n", theta);
        printf("*********************************\n");
        assert(0);
      }
    }
  }
}


/**
  Get the largest edge by non-max suppression.
  
  1. Compare the edge strength of the current pixel with the edge strength of 
     the pixel in the positive and negative gradient directions.
  2. If the edge strength of the current pixel is the largest compared to the 
     other pixels in the mask with the same direction , the value will be 
     preserved. Otherwise, the value will be suppressed.

  Ref: https://en.wikipedia.org/wiki/Canny_edge_detector

  @param[in]  gradient        Gray scale gradient image.
  @param[in]  gradient_angle  Gradient angle in each pixel, separate with diff step.
                              Each step emulate at _ANGLE.
  @param[out] result_img      Result gradient.

**/
void non_max_suppress(unsigned char *gradient, unsigned char *gradient_angle, unsigned char *result_img)
{
  unsigned int img_idx, cmp_idx_1, cmp_idx_2;

  for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
    for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
      img_idx = x + y * IMAGE_WIDTH;

      switch (gradient_angle[img_idx]) {

      // gradient magnitude must greater than "east" and "west"
      case (DEG_0):
        cmp_idx_1 = img_idx + 1;
        cmp_idx_2 = img_idx - 1;
        break;

      // gradient magnitude must greater than "north east" and "south west"
      case (DEG_45):
        cmp_idx_1 = img_idx + IMAGE_WIDTH + 1;
        cmp_idx_2 = img_idx + IMAGE_WIDTH - 1;
        break;

      // gradient magnitude must greater than "north" and "south"
      case (DEG_90):
        cmp_idx_1 = img_idx + IMAGE_WIDTH;
        cmp_idx_2 = img_idx - IMAGE_WIDTH;
        break;

      // gradient magnitude must greater than "north west" and "south-east"
      case (DEG_135):
        cmp_idx_1 = img_idx + IMAGE_WIDTH - 1;
        cmp_idx_2 = img_idx - IMAGE_WIDTH + 1;
        break;
      }

      if (gradient[img_idx] > gradient[cmp_idx_1] && gradient[img_idx] > gradient[cmp_idx_2]) {
        result_img[img_idx] = 0;
      } else {
        result_img[img_idx] = gradient[img_idx];
      }
    }
  }
}


/**
  Remove edge that generate by noise throuugh double threshold.
  
  Divide edge into 3 types:  1. Noise  2. Weak Edge  3. Strong Edge
  Remove noise from image., mark weak/strong edge with diff colors.

  Ref: https://en.wikipedia.org/wiki/Canny_edge_detector

  @param[in]  image           Gray scale gradient image.
  @param[out] result_img      Result gradient.

**/
void double_threshold(unsigned char *image, unsigned char *result_img)
{
  unsigned int img_idx;
  
  for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
    for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
      img_idx = x + y * IMAGE_WIDTH;

      if (image[img_idx] > HIGH_TH) {
        result_img[img_idx] = STRONG_EDGE_PIXEL;
      } else if (image[img_idx] > LOW_TH) {  // low < magnitude < high 
        result_img[img_idx] = WEAK_EDGE_PIXEL;
      } else {  // low > magnitude
        result_img[img_idx] = 0;
      }
    }
  }
}


/**
  Edge tracking by hysteresis.
  
  Strong edge pixels should expexted as a true edge.
  Weak edge pixels might have some noise, treat the weak edge pixels that
  connect with strong edge pixel as true edges, otherwise treat as noise.
  Remove noise from image.

  Ref: https://en.wikipedia.org/wiki/Canny_edge_detector
       https://en.wikipedia.org/wiki/Connected-component_labeling

  @param[in]  image           Gray scale image filtered from double threshold.
  @param[out] result_img      Result gradient.

**/
void hysteresis_track(unsigned char *image, unsigned char *result_img)
{
  unsigned int img_idx;
  
  for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
    for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
      img_idx = x + y * IMAGE_WIDTH;
      
      // blob analysis
      if (image[img_idx] == WEAK_EDGE_PIXEL) {
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            if (image[img_idx + j * IMAGE_WIDTH + i] == STRONG_EDGE_PIXEL) {
              result_img[img_idx] = STRONG_EDGE_PIXEL;
              continue;
            }
          }
        }
        
        if (image[img_idx] == WEAK_EDGE_PIXEL) {
          result_img[img_idx] = 0;
        }
        
      } else if (image[img_idx] == STRONG_EDGE_PIXEL) {
        result_img[img_idx] = STRONG_EDGE_PIXEL;
      } else {  // Not edge
        result_img[img_idx] = 0;
      }
    }
  }
  
  // Skip boundary
  for (int y = 0; y < IMAGE_HEIGHT - 1; y++) {
    result_img[y * IMAGE_WIDTH] = 0;
    result_img[IMAGE_WIDTH -1 + y * IMAGE_WIDTH] = 0;
  }

  for (int x = 0; x < IMAGE_WIDTH; x++) {
    result_img[x] = 0;
    result_img[x + (IMAGE_WIDTH - 1) * IMAGE_WIDTH] = 0;
  }
}


/**
  Get image edge by canny edge.
  
  Canny edge can separate into below step:
  1. Gassian blur
  2. Sobel filter
  3. Non-maximum suppression
  4. Double threshold
  5. Hysteresis track

  Ref: https://en.wikipedia.org/wiki/Canny_edge_detector

  @param[in]  image           Gray scale image.
  @param[in]  temp            Image buffer.
  @param[in]  temp_2          Image buffer.

**/
void canny_edge(unsigned char *image, unsigned char *temp, unsigned char *temp_2)
{
  float            cost_time[5] = {0};
  struct timespec  time;
#if OUTPUT_EACH_IMG
  FILE  *fp_dbg = NULL;
#endif

  //
  // Gaussian Blur
  //
  start_time(&time);
  gassian_blur(image, temp);
  cost_time[TIME_GAUSSIAN_BLUR] = end_time_ms(time);
  
#if OUTPUT_EACH_IMG
  fp_dbg = fopen("lena512_gaussian.raw", "wb");
  fwrite(temp, 1, IMAGE_SIZE, fp_dbg);
#endif

#if CV_PRINT
  char const *name_1 = "Gaussian Blur";

  // Print image after gassian blur
  cv_print_img(temp, name_1);
#endif

  //
  // Sobel Filter
  //
  start_time(&time);
  sobel_filter(temp, image, temp_2);
  cost_time[TIME_SOBEL_FILTER] = end_time_ms(time);
  
#if OUTPUT_EACH_IMG
  fp_dbg = fopen("lena512_sobel_gradient.raw", "wb");
  fwrite(image, 1, IMAGE_SIZE, fp_dbg);

  fp_dbg = fopen("lena512_sobel_angle.raw", "wb");
  fwrite(temp_2, 1, IMAGE_SIZE, fp_dbg);
#endif

#if CV_PRINT
  char const *name_2a = "gradient - Gradient";
  char const *name_2b = "gradient - angle type";

  // Print image after sobel filter
  cv_print_img(image, name_2a);
  cv_print_img(temp_2, name_2b);
#endif

  //
  // Non-max suppression
  //
  start_time(&time);
  non_max_suppress(image, temp_2, temp);
  cost_time[TIME_NON_MAX_SUP] = end_time_ms(time);
  
#if OUTPUT_EACH_IMG
  fp_dbg = fopen("lena512_non_max.raw", "wb");
  fwrite(temp, 1, IMAGE_SIZE, fp_dbg);
#endif

#if CV_PRINT
  char const *name_3 = "Non-maximum suppression";

  // Print gradient after non-maximum suppression
  cv_print_img(temp, name_3);
#endif

  //
  // Double threshold
  //
  start_time(&time);
  double_threshold(temp, temp_2);
  cost_time[TIME_DOUBLE_TH] = end_time_ms(time);
  
#if OUTPUT_EACH_IMG
  fp_dbg = fopen("lena512_double_th.raw", "wb");
  fwrite(temp_2, 1, IMAGE_SIZE, fp_dbg);
#endif

#if CV_PRINT
  char const *name_4 = "Double threshold";

  // Print image after double threshold
  cv_print_img(temp_2, name_4);
#endif

  //
  // Hysteresis track
  //
  start_time(&time);
  hysteresis_track(temp_2, image);  // result return into image
  cost_time[TIME_TRACK] = end_time_ms(time);

#if OUTPUT_EACH_IMG
  fp_dbg = fopen("lena512_hys_track.raw", "wb");
  fwrite(image, 1, IMAGE_SIZE, fp_dbg);

  fclose(fp_dbg);
#endif

#if CV_PRINT
  char const *name_5 = "Hysteresis track";

  // Print image after non-maximum suppression
  cv_print_img(image, name_5);
#endif

  printf("cost time: (ms)\n");
  for (int idx = 0; idx < TIME_MAX; idx++) {
    printf("  #%d:  %f\n", idx, cost_time[idx]);
  }
}


/**
  Function entry, response for:
  1. Get required buffer for canny edge use.
  2. Get input image.

**/
int main()
{
  FILE            *fp = NULL;
  FILE            *fp_result = NULL;
  unsigned char   *image_data = NULL;
  unsigned char   *temp = NULL;
  unsigned char   *temp_2 = NULL;
  int             frame_size = IMAGE_SIZE;

  // Get input image and put it into array
  fp = fopen("lena512.raw", "rb");
  image_data = (unsigned char*) malloc (sizeof(unsigned char) * frame_size);
  fread(image_data, sizeof(unsigned char), frame_size, fp);

#if CV_PRINT
  char const *name_input = "Input";

  // Print input image
  cv_print_img(image_data, name_input);
#endif

  // Allocate temp space for image process
  temp = (unsigned char*) calloc (frame_size, sizeof(char));
  temp_2 = (unsigned char*) calloc (frame_size, sizeof(char));
  
  canny_edge(image_data, temp, temp_2);

  // Output as a raw picture
  fp_result = fopen("lena512_edge.raw", "wb");
  fwrite(image_data, 1, frame_size, fp_result);

#if CV_PRINT
  char const *name_output = "Canny edge";

  // Print result image
  cv_print_img(image_data, name_output);
  waitKey(0);
#endif

  free(image_data);
  free(temp);
  free(temp_2);
  fclose(fp);
}
