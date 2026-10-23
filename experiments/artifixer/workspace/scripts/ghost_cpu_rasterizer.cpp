// CPU reference for this installed gsplat's tiled, front-to-back compositor.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <omp.h>

extern "C" void composite(int n,int width,int height,const float* xy,const float* conic,
 const int* radii,const float* depth,const float* opacity,const float* color,
 const float* regions,float* out,double* attribution){
 const int tw=(width+15)/16,th=(height+15)/16;
 std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
 std::stable_sort(order.begin(),order.end(),[&](int a,int b){return depth[a]<depth[b];});
 std::vector<std::vector<int>> tiles(tw*th);
 for(int g:order){if(radii[g]<=0 || opacity[g]<=0)continue;
  int x0=std::max(0,(int)std::floor((xy[2*g]-radii[g])/16.f));
  int x1=std::min(tw,(int)std::ceil((xy[2*g]+radii[g])/16.f));
  int y0=std::max(0,(int)std::floor((xy[2*g+1]-radii[g])/16.f));
  int y1=std::min(th,(int)std::ceil((xy[2*g+1]+radii[g])/16.f));
  for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++)tiles[y*tw+x].push_back(g);
 }
 #pragma omp parallel for schedule(dynamic,1)
 for(int tile=0;tile<tw*th;tile++){
  float trans[256],rgb[768]={0};bool done[256]={false};
  std::fill(trans,trans+256,1.f);const int ox=(tile%tw)*16,oy=(tile/tw)*16;
  for(int g:tiles[tile]){
   double sums[3]={0,0,0};int active=0;
   for(int p=0;p<256;p++){
    int x=ox+p%16,y=oy+p/16;if(x>=width||y>=height||done[p])continue;
    active++;float dx=xy[2*g]-(x+.5f),dy=xy[2*g+1]-(y+.5f);
    float sigma=.5f*(conic[3*g]*dx*dx+conic[3*g+2]*dy*dy)+conic[3*g+1]*dx*dy;
    float a=std::min(.999f,opacity[g]*std::exp(-sigma));
    if(sigma<0.f||a<1.f/255.f)continue;
    float next=trans[p]*(1.f-a);if(next<=1e-4f){done[p]=true;continue;}
    float weight=a*trans[p];trans[p]=next;
    for(int c=0;c<3;c++){
     rgb[3*p+c]+=weight*color[3*g+c];
     if(regions)sums[c]+=(double)weight*regions[3*(y*width+x)+c];
    }
   }
   if(attribution)for(int c=0;c<3;c++){
    #pragma omp atomic
    attribution[3*g+c]+=sums[c];
   }
   if(active==0)break;
  }
  for(int p=0;p<256;p++){
   int x=ox+p%16,y=oy+p/16;if(x>=width||y>=height)continue;
   for(int c=0;c<3;c++)out[4*(y*width+x)+c]=rgb[3*p+c];
   out[4*(y*width+x)+3]=1.f-trans[p];
  }
 }
}
