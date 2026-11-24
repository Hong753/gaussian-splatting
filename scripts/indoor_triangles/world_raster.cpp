// Two-sided opaque rasterizer. Clip at the near plane, retain original face
// coordinates, interpolate in perspective and resolve equal depths by face ID.
#include <algorithm>
#include <cmath>
#include <cstdint>

struct Vertex { double x, y, z, b, c; };

static Vertex interpolate(const Vertex& a, const Vertex& b, double t) {
    return {a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.z+t*(b.z-a.z),
            a.b+t*(b.b-a.b), a.c+t*(b.c-a.c)};
}

extern "C" void render_faces(const float* xyz, const int32_t* faces, int count,
                             int w, int h, double fx, double fy, double cx, double cy,
                             int32_t* ids, float* depth, float* bary) {
    constexpr double near = 0.02;
    for (int face=0; face<count; ++face) {
        Vertex tri[3], poly[4];
        for (int j=0; j<3; ++j) {
            const float* p = xyz + 3*faces[3*face+j];
            tri[j] = {p[0],p[1],p[2],double(j==1),double(j==2)};
        }
        int n=0;
        for (int j=0; j<3; ++j) {
            const Vertex &a=tri[j], &b=tri[(j+1)%3];
            if (a.z>=near) poly[n++]=a;
            if ((a.z<near) != (b.z<near)) {
                Vertex cut = interpolate(a,b,(near-a.z)/(b.z-a.z));
                cut.z=near;
                poly[n++]=cut;
            }
        }
        for (int j=0; j<n; ++j) {
            auto& p=poly[j];
            p.x=fx*p.x/p.z+cx;
            p.y=fy*p.y/p.z+cy;
        }
        for (int j=1; j+1<n; ++j) {
            const Vertex &a=poly[0], &b=poly[j], &c=poly[j+1];
            const double area=(b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y);
            if (std::abs(area)<1e-12) continue;
            const double xmin=std::max(0., std::min({a.x,b.x,c.x}));
            const double xmax=std::min(double(w-1),std::max({a.x,b.x,c.x}));
            const double ymin=std::max(0., std::min({a.y,b.y,c.y}));
            const double ymax=std::min(double(h-1),std::max({a.y,b.y,c.y}));
            if (xmin>xmax || ymin>ymax) continue;
            for (int y=int(std::ceil(ymin)); y<=int(ymax); ++y) {
                for (int x=int(std::ceil(xmin)); x<=int(xmax); ++x) {
                    const double v=((x-a.x)*(c.y-a.y)-(c.x-a.x)*(y-a.y))/area;
                    const double t=((b.x-a.x)*(y-a.y)-(x-a.x)*(b.y-a.y))/area;
                    const double u=1.-v-t;
                    if (std::min({u,v,t}) < -1e-9) continue;
                    const double inv=u/a.z+v/b.z+t/c.z;
                    const float z=float(1./inv);
                    const int pix=y*w+x;
                    if (z>=depth[pix]) continue;
                    depth[pix]=z;
                    ids[pix]=face;
                    bary[2*pix]=float((u*a.b/a.z+v*b.b/b.z+t*c.b/c.z)/inv);
                    bary[2*pix+1]=float((u*a.c/a.z+v*b.c/b.z+t*c.c/c.z)/inv);
                }
            }
        }
    }
}
