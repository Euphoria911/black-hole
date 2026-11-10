// blackhole.hlsl - compute shaders for blackhole.exe / blackholewin.exe (Direct3D 11, cs_5_0)
// mode 0: Schwarzschild black hole. Every pixel ray is integrated through curved spacetime (u'' = -u + 1.5 u^2, units of the
//         Schwarzschild radius), the accretion disk is ray-marched along the bent ray with Doppler beaming and gravitational
//         redshift, and whatever the ray escapes to is looked up in a lensed sky.
// mode 1: the Milky Way, a fixed picture: spiral arms, bar, bulge, dust lanes, HII regions, resolved stars, clusters,
//         satellite galaxies and a deep field of background galaxies, all procedural.
// Passes: CS_Render -> 4x blur (bloom) -> CS_Compose (tone map) -> CS_Cells (terminal build only)

#define PI  3.14159265
#define TAU 6.28318531

cbuffer CB : register(b0)
{
    float4 camPos;    // xyz position, w = real time (s)
    float4 camFwd;    // xyz, w = simulation time (disk rotation, galaxy rotation)
    float4 camRight;  // xyz, w = tanHx
    float4 camUp;     // xyz, w = tanHy
    int4   dims;      // internal W, H, ssaa, cellsX
    int4   dims2;     // cellsY, sx, sy, mode (0 black hole, 1 Milky Way)
    int4   dims3;     // output W, H, windowed (overlay on), extra detail octaves
    float4 bh;        // x disk inner radius, y disk outer radius, z disk spin multiplier, w view distance
    float4 misc;      // x bloom narrow, y bloom wide, z exposure, w pixel angle (rad)
    float4 misc3;     // x galaxy rotation angle, y sun marker on
};

Texture2D<float4> tHdr : register(t0);
Texture2D<float4> tBlurIn : register(t1);
Texture2D<float4> tBlurB : register(t2);
Texture2D<float4> tBlurC : register(t3);
Texture2D<float4> tPix : register(t4);
Texture2D<float4> tOverlay : register(t7);

RWTexture2D<float4> uHdr : register(u0);
RWTexture2D<float4> uBlurOut : register(u2);
RWTexture2D<float4> uPix : register(u3);
RWStructuredBuffer<uint2> uCells : register(u4);

// ------------------------------------------------------------------ helpers
static int gExtra = 0;     // extra noise octaves when the window has pixels to spare

float Sm(float a, float b, float x) { float t = saturate((x - a) / (b - a)); return t * t * (3.0 - 2.0 * t); }

float Hash3(int3 p)
{
    uint3 u = asuint(p);
    uint h = u.x * 374761393u + u.y * 668265263u + u.z * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return float(h & 0xFFFFFFu) * (1.0 / 16777216.0);
}

float Noise3(float3 p)
{
    float3 f = floor(p);
    int3 i = (int3)f;
    float3 t = p - f;
    t = t * t * (3.0 - 2.0 * t);
    float x00 = lerp(Hash3(i), Hash3(i + int3(1, 0, 0)), t.x);
    float x10 = lerp(Hash3(i + int3(0, 1, 0)), Hash3(i + int3(1, 1, 0)), t.x);
    float x01 = lerp(Hash3(i + int3(0, 0, 1)), Hash3(i + int3(1, 0, 1)), t.x);
    float x11 = lerp(Hash3(i + int3(0, 1, 1)), Hash3(i + int3(1, 1, 1)), t.x);
    return lerp(lerp(x00, x10, t.y), lerp(x01, x11, t.y), t.z);
}

float Fbm(float3 p, int oct)
{
    oct = min(oct + gExtra, 9);
    float a = 0.5, s = 0.0, f = 1.0, tot = 0.0;
    [loop] for (int i = 0; i < oct; i++)
    {
        s += a * Noise3(p * f);
        tot += a;
        f *= 2.03;
        a *= 0.5;
    }
    return s / tot;
}

float Worley2(float u, float v)
{
    float fu = floor(u), fv = floor(v);
    int iu = (int)fu, iv = (int)fv;
    float best = 9.0;
    [loop] for (int dv = -1; dv <= 1; dv++)
        [loop] for (int du = -1; du <= 1; du++)
        {
            int cu = iu + du, cv = iv + dv;
            float px = float(cu) + Hash3(int3(cu, cv, 3));
            float py = float(cv) + Hash3(int3(cu, cv, 9));
            best = min(best, sqrt((u - px) * (u - px) + (v - py) * (v - py)));
        }
    return best;
}

// worley distance plus a hash of the cell that owns the nearest point
float Worley2h(float u, float v, out float hsh)
{
    float fu = floor(u), fv = floor(v);
    int iu = (int)fu, iv = (int)fv;
    float best = 9.0;
    hsh = 0.0;
    [loop] for (int dv = -1; dv <= 1; dv++)
        [loop] for (int du = -1; du <= 1; du++)
        {
            int cu = iu + du, cv = iv + dv;
            float px = float(cu) + Hash3(int3(cu, cv, 3));
            float py = float(cv) + Hash3(int3(cu, cv, 9));
            float d = sqrt((u - px) * (u - px) + (v - py) * (v - py));
            if (d < best) { best = d; hsh = Hash3(int3(cu, cv, 17)); }
        }
    return best;
}

float2 Rot2(float2 p, float a)
{
    float c = cos(a), s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Planckian color for a temperature in kelvin (linear RGB, brightest channel about 1)
float3 Blackbody(float T)
{
    float t = clamp(T, 1000.0, 40000.0) / 100.0;
    float r, g, b;
    if (t <= 66.0) { r = 1.0; g = saturate((99.4708025861 * log(t) - 161.1195681661) / 255.0); }
    else { r = saturate(1.292936186 * pow(t - 60.0, -0.1332047592)); g = saturate(1.1298908609 * pow(t - 60.0, -0.0755148492)); }
    if (t >= 66.0) b = 1.0; else if (t <= 19.0) b = 0.0; else b = saturate((138.5177312231 * log(t - 10.0) - 305.0447927307) / 255.0);
    return pow(float3(r, g, b), 2.2);
}

// ------------------------------------------------------------------ the sky behind everything
float3 Stars(float3 rd)
{
    float3 acc = 0.0;
    float sig = misc.w * 0.5;
    float rt = camPos.w;
    [unroll] for (int layer = 0; layer < 3; layer++)
    {
        float scale = layer == 0 ? 38.0 : (layer == 1 ? 90.0 : 190.0);
        float dens = layer == 0 ? 0.012 : (layer == 1 ? 0.010 : 0.006);
        float3 p = rd * scale;
        float3 fp = floor(p);
        int3 ib = (int3)fp;
        float3 fr = p - fp;
        int3 sg = int3(fr.x < 0.5 ? -1 : 1, fr.y < 0.5 ? -1 : 1, fr.z < 0.5 ? -1 : 1);
        [unroll] for (int k = 0; k < 8; k++)
        {
            int3 c = ib + int3((k & 1) ? sg.x : 0, (k & 2) ? sg.y : 0, (k & 4) ? sg.z : 0);
            float h = Hash3(c + int3(layer * 131, 7, 3));
            if (h > dens) continue;
            float3 sp = float3(c) + float3(Hash3(c + int3(17, 0, 0)), Hash3(c + int3(0, 31, 0)), Hash3(c + int3(0, 0, 47)));
            float3 dv = rd - normalize(sp);
            float hb = Hash3(c + int3(5, 9, 13));
            float b = pow(hb, 9.0) * 1.1 + 0.012;
            float tw = 0.85 + 0.15 * sin(rt * (1.0 + 4.0 * hb) + hb * 60.0);
            float3 tint = lerp(float3(0.75, 0.85, 1.0), float3(1.0, 0.82, 0.65), Hash3(c + int3(23, 29, 41)));
            acc += tint * (b * tw) * exp(-dot(dv, dv) / (sig * sig));
        }
    }
    return acc;
}

float3 Sky(float3 rd)
{
    float3 col = Stars(rd);
    float b = dot(rd, float3(0.2504, 0.8813, 0.4006));
    float band = exp(-b * b * 14.0);
    if (band > 0.03)
    {
        float cloud = Fbm(rd * 3.5, 4);
        float fine = Fbm(rd * 14.0, 3);
        float dust = Sm(0.35, 0.65, Fbm(rd * 6.0 + float3(5.0, 2.0, 9.0), 3));
        float v = band * (0.35 + 1.1 * cloud) * (0.55 + 0.9 * fine) * (1.0 - 0.55 * dust) * 0.085;
        col += lerp(float3(0.55, 0.62, 1.0), float3(1.0, 0.78, 0.55), saturate(cloud * 1.4 - 0.3)) * v;
    }
    col += float3(0.30, 0.10, 0.38) * Sm(0.55, 0.85, Fbm(rd * 2.2 + float3(9.0, 4.0, 1.0), 4)) * 0.030;
    col += float3(0.05, 0.14, 0.34) * Sm(0.50, 0.80, Fbm(rd * 1.7 + float3(2.0, 7.0, 3.0), 3)) * 0.020;
    return col;
}

// ------------------------------------------------------------------ black hole
// Light rays in the Schwarzschild metric obey u'' = -u + 1.5 u^2 with u = 1/r (r in Schwarzschild radii) and ' = d/dphi,
// in the plane that holds the camera, the black hole and the ray. We integrate that with leapfrog steps whose size follows
// the geometry (small near the disk and the photon sphere, large in the weak field). Rays that fall past r = 1 are
// captured, rays that climb back out are looked up in the sky along their final direction. Along the way the thick
// accretion disk emits and absorbs: temperature falls with radius, the matter orbits at the local Kepler speed so the
// approaching side is Doppler-boosted and blue, the receding side dim and red, and everything deep in the well is redshifted.
// disk color from relative observed temperature: dusty brown, orange, amber, white-hot, a hint of blue where it is beamed
float3 DiskColor(float x)
{
    float3 dust = float3(0.18, 0.06, 0.025);
    float3 orange = float3(0.95, 0.30, 0.07);
    float3 amber = float3(1.0, 0.66, 0.30);
    float3 white = float3(1.0, 0.94, 0.86);
    float3 hot = float3(0.86, 0.93, 1.0);
    float3 c = lerp(dust, orange, Sm(0.05, 0.28, x));
    c = lerp(c, amber, Sm(0.25, 0.6, x));
    c = lerp(c, white, Sm(0.55, 1.0, x));
    return lerp(c, hot, Sm(1.1, 1.8, x));
}

float3 TraceBH(float3 ro, float3 rd)
{
    float simT = camFwd.w;
    float rin = bh.x, rout = bh.y;
    float D = length(ro);
    float3 e1 = ro / D;
    float dn = dot(rd, e1);
    if (dn > 0.0 && D > rout * 1.2) return Sky(rd);          // looking away from the hole, nothing in the way
    float3 perp = rd - dn * e1;
    float pl = length(perp);
    float3 e2 = pl > 1e-5 ? perp / pl : normalize(cross(e1, float3(0.0, 1.0, 0.0)));
    pl = max(pl, 0.02);
    float b = D * pl;                                         // impact parameter
    bool farRay = b > rout * 1.3;                             // never gets near the disk: coarse steps
    float u = 1.0 / D;
    float v = -u * dn / pl;                                   // du/dphi
    float phi = 0.0;
    float3 acc = 0.0;
    float trans = 1.0;
    bool captured = false;
    float rCut = max(D * 1.02, 45.0);

    [loop] for (int i = 0; i < 320; i++)
    {
        float r = 1.0 / u;
        float h;
        if (farRay) h = 0.11;
        else if (r > rout * 1.5) h = 0.07;
        else
        {
            h = clamp(0.2 * u, 0.012, 0.085);
            float sn0, cs0;
            sincos(phi, sn0, cs0);
            float y0 = r * (cs0 * e1.y + sn0 * e2.y);
            if (abs(y0) > 2.4 * (0.14 * r + 0.06) && r > rin * 0.9) h *= 2.2;     // far above or below the plane nothing is emitted
        }
        float a0 = -u + 1.5 * u * u;
        float vh = v + 0.5 * h * a0;
        float un = u + h * vh;
        float a1 = -un + 1.5 * un * un;
        float vn = vh + 0.5 * h * a1;

        if (!farRay && r < 7.0)
        {
            float rnC = 1.0 / un;
            float dsC = sqrt((rnC - r) * (rnC - r) + (r * h) * (r * h));
            acc += trans * float3(1.0, 0.72, 0.45) * 0.010 * dsC * exp(-(r - 1.2) * 0.75);     // faint hot corona
        }
        if (!farRay)
        {
            float um = 0.5 * (u + un);
            float rm = 1.0 / um;
            if (rm > rin * 0.85 && rm < rout * 1.1)
            {
                float sn, cs;
                sincos(phi + 0.5 * h, sn, cs);
                float3 rad = cs * e1 + sn * e2;
                float3 p = rm * rad;
                float Hb = 0.016 * rm + 0.008;
                if (abs(p.y) < 0.14 * rm + 0.06)                         // cheap bound before any noise
                {
                    float rn = 1.0 / un;
                    float ds = sqrt((rn - r) * (rn - r) + (rm * h) * (rm * h));
                    float prof = Sm(rin * 0.85, rin * 1.35, rm) * (1.0 - Sm(rout * 0.55, rout * 1.05, rm));
                    float om = bh.z * 0.707 / (rm * sqrt(rm));          // Keplerian angular velocity
                    float sa, ca;
                    sincos(-om * simT, sa, ca);
                    float2 qq = float2(ca * p.x - sa * p.z, sa * p.x + ca * p.z);   // noise lives in the co-rotating frame
                    float thc = atan2(qq.y, qq.x);
                    float nz2 = Fbm(float3(rm * 1.3 + 7.0, thc * 0.9, 1.0), 4);      // big clumps: a warped, ragged plane
                    float H = Hb * (0.4 + 1.4 * nz2);
                    float yy = p.y - (nz2 - 0.5) * 0.10 * rm;
                    if (abs(yy) < 3.2 * H)
                    {
                        float vert = exp(-0.5 * (yy / H) * (yy / H));
                        float nz = Fbm(float3(rm * 3.2, thc * 1.6, p.y * 0.5), 5);   // streaks wound up by the differential rotation
                        float fil = Fbm(float3(rm * 13.0, thc * 5.0, 2.0), 3);       // fine bands
                        float dens = prof * vert * (0.06 + 2.4 * pow(nz, 2.6)) * (0.45 + 1.1 * fil) * (0.35 + 1.4 * nz2);
                        if (dens > 0.004)
                        {
                            float sq = sqrt(rin / rm);
                            float Te = pow(rin / rm, 0.75) * pow(max(1.0 - sq, 0.0), 0.25) / 0.49;   // thin-disk temperature profile, ~1 at its peak
                            // photon direction in the static frame and the disk's orbital motion
                            float3 tn = -sn * e1 + cs * e2;
                            float vm = 0.5 * (v + vn);
                            float3 tdir = normalize((-vm * rm * rm) * rad + rm * tn);     // dr/dphi = -v / u^2
                            float3 aph = normalize(float3(-p.z, 0.0, p.x));
                            float beta = min(sqrt(0.5 / max(rm - 1.0, 0.05)), 0.9);
                            float gam = rsqrt(1.0 - beta * beta);
                            float g = sqrt(max(1.0 - 1.0 / rm, 0.01)) / (gam * (1.0 - dot(aph * beta, -tdir)));
                            g = clamp(g, 0.15, 3.0);
                            float I = pow(g, 2.2) * pow(Te, 3.6) * 3.0;                   // beamed, redshifted intensity
                            float lane = Sm(0.55, 0.75, Fbm(float3(rm * 9.0, thc * 2.2, 6.0), 4));    // dust lanes: dark, and opaque to what lies behind
                            float fine = Fbm(float3(rm * 16.0, thc * 5.0, 4.0), 3);                     // thin filaments
                            float3 S = DiskColor(pow(Te, 1.8) * g) * I * (0.2 + 1.5 * nz) * (0.55 + 0.9 * fil) * (0.7 + 0.6 * fine) * (1.0 - 0.75 * lane);
                            float al = 1.0 - exp(-dens * ds * (4.5 + 9.0 * lane));
                            acc += trans * al * S;
                            trans *= 1.0 - al;
                        }
                    }
                }
            }
        }
        u = un; v = vn; phi += h;
        if (u >= 1.0) { captured = true; break; }                // fell behind the horizon
        if (v < 0.0 && 1.0 / u > rCut) break;                    // escaping
        if (trans < 0.01) { captured = true; break; }
    }
    if (captured || (u > 0.4 && trans > 0.0 && v >= 0.0 && 1.0 / u < rCut)) return acc;
    float r = 1.0 / u;
    float sn, cs;
    sincos(phi, sn, cs);
    float3 rad = cs * e1 + sn * e2;
    float3 tn = -sn * e1 + cs * e2;
    float3 dirOut = normalize((-v / (u * u)) * rad + r * tn);
    return acc + trans * Sky(dirOut);
}

// ------------------------------------------------------------------ Milky Way
float ArmField(float2 q, out float a)
{
    float R = length(q);
    float th = atan2(q.y, q.x);
    a = th - log(max(R, 0.03) / 0.12) / 0.30;                       // logarithmic spiral
    float wob = Fbm(float3(q * 6.0, 1.7), 4) - 0.5;
    a += wob * 0.9;
    float major = pow(0.5 + 0.5 * cos(2.0 * a), 4.5);               // two major arms
    float minorA = pow(0.5 - 0.5 * cos(2.0 * a), 3.5) * 0.45;       // two minor arms between them
    float spur = pow(0.5 + 0.5 * cos(9.0 * a + 3.0 * wob), 6.0) * 0.30 * Sm(0.25, 0.5, R);   // feathers and spurs
    float fade = Sm(0.07, 0.28, R) * (1.0 - Sm(0.95, 1.15, R));
    return saturate(major + minorA + spur) * fade;
}

float Bar2D(float2 q)
{
    const float ba = 0.47;
    float xb = q.x * cos(ba) + q.y * sin(ba);
    float yb = -q.x * sin(ba) + q.y * cos(ba);
    return exp(-pow(abs(xb) / 0.26, 3.2) - pow(abs(yb) / 0.078, 2.0));
}

float DustTau(float2 q, float a)
{
    const float ba = 0.47;
    float R = length(q);
    float lane = pow(0.5 + 0.5 * cos(2.0 * (a + 0.55)), 5.0);       // dust sits on the inner edge of the arms
    float w = Fbm(float3(q * 5.0, 7.0), 3);
    float ridge = pow(saturate(1.0 - abs(2.0 * Fbm(float3(q * 26.0 + w * 2.0, 4.0), 5) - 1.0)), 2.2);   // filaments
    float tau = (0.25 + 1.6 * lane) * ridge * exp(-R / 0.42) * Sm(0.05, 0.2, R) * (1.0 - Sm(0.9, 1.2, R));
    float xb = q.x * cos(ba) + q.y * sin(ba), yb = -q.x * sin(ba) + q.y * cos(ba);
    tau += 1.2 * exp(-pow((abs(yb) - 0.05) / 0.018, 2.0)) * (abs(xb) < 0.24 ? 1.0 : 0.0) * ridge;   // dust lanes along the bar
    tau += 0.22 * exp(-R / 0.7);
    return tau * 2.8;
}

// unresolved starlight, young blue stars in the arms, old yellow stars in the disk and bar, HII regions
float3 DiskEmission(float2 q, float seed)
{
    float a;
    float arm = ArmField(q, a);
    float R = length(q);
    float disk = exp(-R / 0.26) * (1.0 - 0.6 * exp(-R / 0.07)) * (1.0 - Sm(0.9, 1.15, R));
    float bar = Bar2D(q);
    float grain = Fbm(float3(q * 70.0, seed), 5);
    float grain2 = Noise3(float3(q * 210.0, seed + 3.0));
    float dens = disk * (0.12 + 2.3 * arm) * (0.55 + 0.9 * grain) * (0.8 + 0.4 * grain2);
    float3 c = lerp(float3(1.0, 0.82, 0.55), float3(0.62, 0.74, 1.0), saturate(arm * 1.4 - 0.15 + (grain - 0.5) * 0.5));
    float3 e = c * dens;
    e += float3(1.0, 0.70, 0.40) * bar * 0.55 * (0.8 + 0.4 * grain);
    float2 qw = q + 0.02 * float2(Noise3(float3(q * 25.0, 1.0)) - 0.5, Noise3(float3(q * 25.0, 2.0)) - 0.5);
    float hc;
    float w = Worley2h(qw.x * 30.0, qw.y * 30.0, hc);
    float pick = step(0.55, hc);                                    // only some cells hold a star-forming region
    float rad = 0.10 + 0.20 * frac(hc * 7.31);
    float hii = (1.0 - Sm(0.0, rad, w)) * pick * Sm(0.4, 0.8, arm) * Sm(0.4, 0.7, Noise3(float3(q * 9.0, 5.0)));
    float fil = 0.5 + 0.5 * Fbm(float3(q * 120.0, 2.0), 4);
    e += float3(1.0, 0.28, 0.42) * hii * 1.9 * fil;
    e += float3(0.7, 0.8, 1.0) * (1.0 - Sm(0.0, 0.07, w)) * pick * Sm(0.5, 0.8, arm) * Sm(0.5, 0.8, Noise3(float3(q * 9.0, 5.0))) * 1.2;
    return e;
}

// individual stars, in cells of several sizes; a size below one pixel melts into the grain above
float3 StarSprites(float2 q, float footprint, float R, float arm)
{
    float3 acc = 0.0;
    [unroll] for (int L = 0; L < 4; L++)
    {
        float cell = 0.012 * pow(0.46, float(L));
        if (cell < footprint * 0.9) continue;
        float2 p = q / cell;
        float2 fp = floor(p);
        int2 ib = (int2)fp;
        float2 fr = p - fp;
        int2 sg = int2(fr.x < 0.5 ? -1 : 1, fr.y < 0.5 ? -1 : 1);
        float dens = (0.10 + 0.9 * arm) * exp(-R / 0.32) * lerp(1.0, 0.35, float(L) / 3.0);
        [unroll] for (int k = 0; k < 4; k++)
        {
            int2 c = ib + int2((k & 1) ? sg.x : 0, (k >> 1) ? sg.y : 0);
            float h = Hash3(int3(c, 31 + L * 17));
            if (h > dens * 1.3) continue;
            float2 sp = float2(c) + float2(Hash3(int3(c, 5 + L)), Hash3(int3(c, 9 + L)));
            float2 d = (sp - p) * cell;
            float hb = Hash3(int3(c, 71 + L));
            float b = pow(hb, 5.0) * 2.2 + 0.08;
            float s2 = footprint * 0.65;
            float yf = saturate(arm * 1.2 - 0.2 + (Hash3(int3(c, 13 + L)) - 0.5) * 0.8);
            acc += lerp(float3(1.0, 0.78, 0.5), float3(0.65, 0.78, 1.0), yf) * b * exp(-dot(d, d) / (s2 * s2));
        }
    }
    return acc * 0.5;
}

// faint distant galaxies scattered over the sky
float3 DeepField(float3 rd)
{
    float3 acc = 0.0;
    float3 p = rd * 70.0;
    float3 fp = floor(p);
    int3 ib = (int3)fp;
    float3 fr = p - fp;
    int3 sg = int3(fr.x < 0.5 ? -1 : 1, fr.y < 0.5 ? -1 : 1, fr.z < 0.5 ? -1 : 1);
    [unroll] for (int k = 0; k < 8; k++)
    {
        int3 c = ib + int3((k & 1) ? sg.x : 0, (k & 2) ? sg.y : 0, (k & 4) ? sg.z : 0);
        if (Hash3(c + int3(401, 3, 9)) > 0.022) continue;
        float3 sd = normalize(float3(c) + float3(Hash3(c + int3(1, 2, 3)), Hash3(c + int3(4, 5, 6)), Hash3(c + int3(7, 8, 9))));
        float3 t1 = normalize(cross(sd, float3(0.0, 1.0, 0.0)));
        float3 t2 = cross(sd, t1);
        float3 dv = rd - sd;
        float2 d = float2(dot(dv, t1), dot(dv, t2));
        float ang = Hash3(c + int3(11, 12, 13)) * PI;
        float2 e = Rot2(d, ang);
        float size = 0.0015 + 0.0035 * Hash3(c + int3(14, 15, 16));
        float flat2 = 0.25 + 0.6 * Hash3(c + int3(17, 18, 19));
        float q = (e.x * e.x) / (size * size) + (e.y * e.y) / (size * size * flat2 * flat2);
        float spiral = 0.7 + 0.3 * cos(3.0 * atan2(e.y / flat2, e.x) + 14.0 * sqrt(q));
        float3 col = lerp(float3(1.0, 0.85, 0.65), float3(0.7, 0.8, 1.0), Hash3(c + int3(20, 21, 22)));
        acc += col * exp(-q * 1.2) * spiral * (0.06 + 0.30 * Hash3(c + int3(23, 24, 25)));
    }
    return acc;
}

// bulge and bar: a short ray march through the spheroid; the half behind the dust plane is dimmed
float3 Bulge(float3 ro, float3 rd, float tauPlane, float tPlane, float rotT)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - 0.45 * 0.45;
    float disc = b * b - c;
    if (disc <= 0.0) return 0.0;
    float sq = sqrt(disc);
    float t0 = max(-b - sq, 0.0), t1 = -b + sq;
    const int N = 14;
    float dt = (t1 - t0) / float(N);
    float3 acc = 0.0;
    [loop] for (int i = 0; i < N; i++)
    {
        float t = t0 + (float(i) + 0.5) * dt;
        float3 p = ro + rd * t;
        float s = length(float3(p.x, p.y / 0.62, p.z)) / 0.085;
        float rho = exp(-pow(s, 0.52) * 2.4);
        float2 q = Rot2(p.xz, rotT);
        float bar = Bar2D(q) * exp(-(p.y * p.y) / (0.05 * 0.05));
        float e = rho * 1.9 + bar * 0.8;
        float grain = 0.75 + 0.5 * Noise3(p * 60.0);
        float3 col = lerp(float3(1.0, 0.66, 0.34), float3(1.0, 0.86, 0.62), rho);
        float att = t > tPlane ? exp(-tauPlane) : 1.0;
        acc += col * e * grain * dt * att;
    }
    return acc * 0.7;
}

float3 Galaxy(float3 ro, float3 rd)
{
    float rotT = misc3.x;
    float pix = misc.w;
    float3 deep = Sky(rd) + DeepField(rd) * 1.3;
    float3 col = deep;
    if (rd.y > -0.02) return col;

    float tF = (0.012 - ro.y) / rd.y, t0 = -ro.y / rd.y, tB = (-0.012 - ro.y) / rd.y;
    float3 p0 = ro + rd * t0;
    float2 q0 = Rot2(p0.xz, rotT);
    float a0;
    ArmField(q0, a0);
    float tau = DustTau(q0, a0);
    float R0 = length(p0.xz);
    float footprint = t0 * pix / sqrt(max(abs(rd.y), 0.3));

    float3 pF = ro + rd * tF, pB = ro + rd * tB;
    float2 qF = Rot2(pF.xz, rotT), qB = Rot2(pB.xz, rotT);
    float aF, aB;
    float armF = ArmField(qF, aF), armB = ArmField(qB, aB);
    float3 eF = DiskEmission(qF, 1.0) + StarSprites(qF, footprint, length(qF), armF);
    float3 eB = DiskEmission(qB, 9.0) + StarSprites(qB, footprint, length(qB), armB);
    float3 gal = eF * 0.9 + exp(-tau) * eB * 0.9;
    gal += Bulge(ro, rd, tau, t0, rotT);

    // satellite galaxies (the Magellanic Clouds) and globular clusters
    float3 extra = 0.0;
    float3 cen[2] = { float3(1.45, -0.35, 0.55), float3(1.68, -0.5, 0.95) };
    float sizes[2] = { 0.09, 0.045 };
    [unroll] for (int s = 0; s < 2; s++)
    {
        float tc = dot(cen[s] - ro, rd);
        float3 pc = ro + rd * tc;
        float d = length(pc - cen[s]);
        float cl = 0.35 + 1.3 * Fbm((pc - cen[s]) * (s == 0 ? 22.0 : 40.0), 4);
        extra += lerp(float3(0.75, 0.82, 1.0), float3(1.0, 0.45, 0.55), Fbm(pc * 30.0, 3)) * exp(-d * d / (sizes[s] * sizes[s])) * cl * 0.55;
    }
    [loop] for (int g = 0; g < 36; g++)
    {
        float hr = Hash3(int3(g, 77, 5));
        float3 dir = normalize(float3(Hash3(int3(g, 1, 5)) - 0.5, (Hash3(int3(g, 2, 5)) - 0.5) * 0.9, Hash3(int3(g, 3, 5)) - 0.5));
        float3 pc = dir * (0.12 + 0.85 * hr * hr);
        float tc = dot(pc - ro, rd);
        float d = length(ro + rd * tc - pc);
        float s2 = max(0.0065, t0 * pix * 0.8);
        extra += float3(1.0, 0.88, 0.7) * (exp(-d * d / (s2 * s2)) * 0.45 + exp(-d * d / (s2 * s2 * 9.0)) * 0.08);
    }
    // the halo of old stars, brightest in the middle
    float tm = max(dot(-ro, rd), 0.0);
    float dm = length(ro + rd * tm);
    extra += float3(1.0, 0.85, 0.65) * 0.012 / pow(dm * dm + 0.06, 1.2);

    // "you are here": a small ring at the Sun's place between two arms
    if (misc3.y > 0.5)
    {
        float thp = 2.30;
        float2 sunq = 0.53 * float2(cos(thp), sin(thp));
        float2 sunw = Rot2(sunq, -rotT);
        float dsun = length(p0.xz - sunw);
        gal += float3(1.0, 0.9, 0.4) * exp(-pow((dsun - 0.022) / 0.0045, 2.0)) * 1.4;
    }
    col = col * exp(-tau * 0.6) + (gal + extra) * 1.5;
    return col;
}

// ------------------------------------------------------------------ passes
float3 PixelRay(float2 px)
{
    float u = (px.x / float(dims.x) * 2.0 - 1.0) * camRight.w;
    float v = (1.0 - px.y / float(dims.y) * 2.0) * camUp.w;
    return normalize(camFwd.xyz + camRight.xyz * u + camUp.xyz * v);
}

[numthreads(8, 8, 1)]
void CS_Render(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    gExtra = dims3.w;
    int S = dims.z;
    if (dims2.w == 0)
    {
        // rays that skim the photon sphere carry the fine detail (photon ring, the wrapped disk): give them extra samples
        float3 rd0 = PixelRay(float2(id.xy) + 0.5);
        float3 e0 = camPos.xyz / length(camPos.xyz);
        float dn0 = dot(rd0, e0);
        float b0 = length(camPos.xyz) * length(rd0 - dn0 * e0);
        if (dn0 < 0.0 && b0 > 2.25 && b0 < 3.2) S = max(S, 4);       // the photon ring itself: a razor-thin, very bright feature
        else if (dn0 < 0.0 && b0 > 1.8 && b0 < 4.6) S = max(S, 2);
    }
    float3 acc = 0.0;
    [loop] for (int sy = 0; sy < S; sy++)
        [loop] for (int sx = 0; sx < S; sx++)
        {
            float3 rd = PixelRay(float2(id.xy) + (float2(sx, sy) + 0.5) / float(S));
            float3 c = dims2.w == 1 ? Galaxy(camPos.xyz, rd) : TraceBH(camPos.xyz, rd);
            acc += min(c, dims2.w == 1 ? 30.0 : 5.0);       // clamp fireflies: the photon ring is a razor-thin, extremely bright feature
        }
    uHdr[id.xy] = float4(acc / float(S * S), 1.0);
}

float3 BrightOf(float3 c)
{
    float l = dot(c, float3(0.30, 0.59, 0.11));
    return c * saturate((l - 0.9) / 0.8);
}

float3 SceneAt(int2 p)
{
    p = clamp(p, int2(0, 0), int2(dims.x - 1, dims.y - 1));
    return tHdr[p].rgb;
}

[numthreads(8, 8, 1)]
void CS_BrightH1(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    float3 acc = 0.0; float wsum = 0.0;
    [loop] for (int k = -10; k <= 10; k++)
    {
        float w = exp(-float(k * k) / 32.0);
        acc += BrightOf(SceneAt(int2(id.xy) + int2(k, 0))) * w;
        wsum += w;
    }
    uBlurOut[id.xy] = float4(acc / wsum, 1.0);
}

float3 BlurTap(int2 p, int2 dir, int stepPx)
{
    float3 acc = 0.0; float wsum = 0.0;
    [loop] for (int k = -10; k <= 10; k++)
    {
        float w = exp(-float(k * k) / 32.0);
        int2 q = clamp(p + dir * (k * stepPx), int2(0, 0), int2(dims.x - 1, dims.y - 1));
        acc += tBlurIn[q].rgb * w;
        wsum += w;
    }
    return acc / wsum;
}

[numthreads(8, 8, 1)]
void CS_BlurV1(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    uBlurOut[id.xy] = float4(BlurTap(int2(id.xy), int2(0, 1), 1), 1.0);
}

[numthreads(8, 8, 1)]
void CS_BlurH2(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    uBlurOut[id.xy] = float4(BlurTap(int2(id.xy), int2(1, 0), 4), 1.0);
}

[numthreads(8, 8, 1)]
void CS_BlurV2(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    uBlurOut[id.xy] = float4(BlurTap(int2(id.xy), int2(0, 1), 4), 1.0);
}

float Tone(float c) { c = max(c, 0.0); return c < 0.8 ? c : 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) * 5.0)); }

float3 BilScene(float2 f)
{
    float2 q = f - 0.5;
    float2 p0 = floor(q);
    float2 t = q - p0;
    int2 i0 = int2(p0);
    return lerp(lerp(SceneAt(i0), SceneAt(i0 + int2(1, 0)), t.x), lerp(SceneAt(i0 + int2(0, 1)), SceneAt(i0 + int2(1, 1)), t.x), t.y);
}

float3 BilTex(Texture2D<float4> tex, float2 f)
{
    float2 q = f - 0.5;
    float2 p0 = floor(q);
    float2 t = q - p0;
    int2 i0 = int2(p0);
    int2 mx = int2(dims.x - 1, dims.y - 1);
    float3 a = tex[clamp(i0, int2(0, 0), mx)].rgb;
    float3 b = tex[clamp(i0 + int2(1, 0), int2(0, 0), mx)].rgb;
    float3 c = tex[clamp(i0 + int2(0, 1), int2(0, 0), mx)].rgb;
    float3 d = tex[clamp(i0 + int2(1, 1), int2(0, 0), mx)].rgb;
    return lerp(lerp(a, b, t.x), lerp(c, d, t.x), t.y);
}

[numthreads(8, 8, 1)]
void CS_Compose(uint3 id : SV_DispatchThreadID)
{
    int ow = dims3.x, oh = dims3.y;
    if (id.x >= (uint)ow || id.y >= (uint)oh) return;
    int2 p = int2(id.xy);
    float3 c;
    if (ow == dims.x && oh == dims.y)
        c = tHdr[p].rgb + tBlurB[p].rgb * misc.x + tBlurC[p].rgb * misc.y;
    else
    {
        float2 f = (float2(p) + 0.5) * float2(float(dims.x), float(dims.y)) / float2(float(ow), float(oh));
        c = BilScene(f) + BilTex(tBlurB, f) * misc.x + BilTex(tBlurC, f) * misc.y;
    }
    float2 ndc = float2((float(p.x) + 0.5) / float(ow) * 2.0 - 1.0, 1.0 - (float(p.y) + 0.5) / float(oh) * 2.0);
    c *= misc.z;
    c *= 1.0 - 0.20 * dot(ndc, ndc);                     // vignette
    float3 o;
    if (dims2.w == 0)
    {
        float3 x = c * 0.62;                             // filmic curve: highlights roll off toward white the way film does
        o = saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
        o = pow(o, 1.0 / 2.2);
    }
    else
    {
        o = float3(Tone(c.x), Tone(c.y), Tone(c.z));
        o = sqrt(o);                                     // display gamma
    }
    if (dims3.z != 0)
    {
        float4 ov = tOverlay[p];
        o = lerp(o, 0.0, ov.g * 0.75);
        o = lerp(o, float3(0.84, 0.87, 0.94), ov.r);
    }
    o += (Hash3(int3(p, 7)) - 0.5) * (1.0 / 255.0);      // dither against banding
    uPix[p] = float4(saturate(o), 1.0);
}

// ------------------------------------------------------------------ best two-color fit per terminal cell
[numthreads(8, 8, 1)]
void CS_Cells(uint3 id : SV_DispatchThreadID)
{
    int cx = (int)id.x, cy = (int)id.y;
    int cellsX = dims.w, cellsY = dims2.x;
    if (cx >= cellsX || cy >= cellsY) return;
    int sx = dims2.y, sy = dims2.z;
    int n = sx * sy;
    int3 px[6];
    [unroll] for (int z = 0; z < 6; z++) px[z] = int3(0, 0, 0);
    [loop] for (int j = 0; j < sy; j++)
        [loop] for (int i = 0; i < sx; i++)
            px[j * sx + i] = (int3)(saturate(tPix[int2(cx * sx + i, cy * sy + j)].rgb) * 255.0 + 0.5);

    int3 mn = int3(255, 255, 255), mx = int3(0, 0, 0), sumAll = int3(0, 0, 0);
    [loop] for (int k = 0; k < n; k++) { mn = min(mn, px[k]); mx = max(mx, px[k]); sumAll += px[k]; }
    int3 fgm, bgm;
    int best = 0;
    int3 spread = mx - mn;
    if (spread.x < 6 && spread.y < 6 && spread.z < 6)
    {
        fgm = sumAll / n; bgm = fgm;
    }
    else
    {
        int bestErr = 0x7fffffff;
        int nm = 1 << (n - 1);
        fgm = sumAll / n; bgm = fgm;
        [loop] for (int m = 1; m < nm; m++)
        {
            int3 sa = int3(0, 0, 0), sb = int3(0, 0, 0);
            int ca = 0, cb = 0;
            [loop] for (int q = 0; q < n; q++)
            {
                if ((m >> q) & 1) { sa += px[q]; ca++; } else { sb += px[q]; cb++; }
            }
            int3 ma = sa / max(ca, 1), mb = sb / max(cb, 1);
            int err = 0;
            [loop] for (int r = 0; r < n; r++)
            {
                int3 d = px[r] - (((m >> r) & 1) ? ma : mb);
                err += d.x * d.x + d.y * d.y + d.z * d.z;
            }
            if (err < bestErr) { bestErr = err; best = m; fgm = ma; bgm = mb; }
        }
        int3 mean = sumAll / n;
        int flatErr = 0;
        [loop] for (int u = 0; u < n; u++) { int3 d = px[u] - mean; flatErr += d.x * d.x + d.y * d.y + d.z * d.z; }
        if (flatErr <= bestErr + 24) { best = 0; fgm = mean; bgm = mean; }
    }
    fgm &= 0xFC; bgm &= 0xFC;
    uint fg = ((uint)fgm.x << 16) | ((uint)fgm.y << 8) | (uint)fgm.z;
    uint bg = ((uint)bgm.x << 16) | ((uint)bgm.y << 8) | (uint)bgm.z;
    uCells[cy * cellsX + cx] = uint2(fg | ((uint)best << 24), bg);
}
