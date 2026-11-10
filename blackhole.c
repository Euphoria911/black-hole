// blackhole.c - a Schwarzschild black hole ray-traced on the GPU (Direct3D 11 compute shaders), with a locked Milky Way view.
// Two front ends, one ray tracer:
//
//   terminal build   cl /O2 /W4 /nologo blackhole.c
//       draws into a console with Unicode sextants (2x3 sub-pixels per cell, truecolor). Needs Windows Terminal or another
//       console with VT, truecolor, UTF-8 and the Symbols for Legacy Computing glyphs.
//
//   window build     cl /O2 /W4 /nologo /DWINDOWED /Foblackholewin.obj /Fe:blackholewin.exe blackhole.c /link /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup
//       draws into a native window at full resolution, with dynamic resolution scaling and extra detail.
//
// Needs blackhole.hlsl next to the exe (compiled shaders are cached as .cso) and a GPU with Direct3D 11 compute (FL 11_0).
//
// Black hole view: every pixel's light ray is integrated through the Schwarzschild metric, so you get the real shadow,
// photon ring, and the disk wrapped over the top and under the bottom of the hole. The accretion disk is ray-marched
// along the bent rays; its Keplerian rotation Doppler-beams the approaching side and redshifts everything near the horizon.
//
// keys:  drag mouse / arrows / a d w s   orbit              wheel / + -   zoom
//        1 edge-on  2 oblique  3 steep  4 top-down  5 close-in  0 reset
//        t auto-orbit on/off   [ ] disk spin speed   space pause   l labels   h hide help (window)   F11 fullscreen (window)
//        m  Milky Way. A fixed, locked picture: no navigation from there, press m again to come back.
//        n  cycle the terminal sub-pixel mode (terminal build)         q or esc quit
// dev:   blackhole.exe --frames N [--size CxR] [--view K] [--galaxy]    render N frames to stdout and exit
//        blackhole.exe --view K [--galaxy] [--fps N]        start live on preset K
//        blackholewin.exe --view K [--galaxy]
#define _CRT_SECURE_NO_WARNINGS
#define COBJMACROS
#include <conio.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <windowsx.h>
#include <mmsystem.h>
#pragma warning(push)
#pragma warning(disable : 4201)   // nameless unions in the Windows SDK headers
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#pragma warning(pop)
#pragma warning(disable : 4505)   // each front end leaves the other one's helpers unused
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

#define kPi  3.14159265f
#define kTau 6.28318531f

// ---------------------------------------------------------------- config
static const float kTanHalfFovBH = 0.42f;
static const float kTanHalfFovGal = 0.45f;
static const float kCellAspect = 2.0f;           // terminal cell height : width (Windows Terminal default font is 1:2)
static const float kBHRoll = -0.10f;               // the picture is tilted a little, like a photograph
static const float kDiskIn = 3.0f;               // inner edge = innermost stable orbit, in Schwarzschild radii
static const float kDiskOut = 20.0f;
static const float kGalaxyTilt = 0.70f;          // inclination of the Milky Way view from face-on (rad)
static const float kGalaxyDist = 2.2f;
static const int   kMaxPixels = 700000;          // terminal: above this the mode steps down (sextant -> quadrant -> half)
static const float kTargetGpuMs = 14.0f;         // window: dynamic resolution aims for this GPU time per frame
static int gTickMs = 16;                          // terminal frame cap, --fps N

// ---------------------------------------------------------------- vec3
typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
static V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 mulf(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(dot(a, a)); }
static V3 norm(V3 a) { float l = vlen(a); return l > 1e-8f ? mulf(a, 1.0f / l) : a; }
static float Clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---------------------------------------------------------------- output modes and camera
enum { M_HALF, M_QUAD, M_SEXT, M_WIN };
static const char *kModeName[4] = { "half", "quad", "sext", "win" };
static int   gModePref = M_SEXT, gMode = M_SEXT, gSx = 2, gSy = 3, gSsaa = 2, gModeChanged;
static float gPixAspect = 0.75f;                 // pixel width / pixel height
static int   gW, gH;                             // output pixels
static int   gIW, gIH;                           // internal render resolution (window: scaled by dynamic resolution)
static int   gExtraOct;                          // extra noise octaves, window only

typedef struct { V3 pos, tgt, fwd, right, up; float tanHx, tanHy; } Cam;
static Cam gCam;

static void SetMode(int mode)
{
    gMode = mode;
    gSx = mode == M_HALF ? 1 : 2;
    gSy = mode == M_SEXT ? 3 : 2;
    if (mode == M_WIN) { gSx = gSy = 1; }
    gPixAspect = mode == M_WIN ? 1.0f : ((float)gSy / (float)gSx) / kCellAspect;
}

static void BuildCam(V3 pos, V3 tgt, V3 upHint, float tanHy, float roll)
{
    gCam.pos = pos;
    gCam.tgt = tgt;
    gCam.fwd = norm(sub(tgt, pos));
    gCam.right = norm(cross(gCam.fwd, upHint));
    gCam.up = cross(gCam.right, gCam.fwd);
    {                                                    // roll the picture about the view axis
        float cr = cosf(roll), sr = sinf(roll);
        V3 r2 = add(mulf(gCam.right, cr), mulf(gCam.up, sr));
        V3 u2 = sub(mulf(gCam.up, cr), mulf(gCam.right, sr));
        gCam.right = r2;
        gCam.up = u2;
    }
    gCam.tanHy = tanHy;
    gCam.tanHx = tanHy * ((float)gW * gPixAspect) / (float)gH;
}

static int Project(V3 p, float *sx, float *sy)
{
    V3 v = sub(p, gCam.pos);
    float z = dot(v, gCam.fwd);
    if (z < 0.05f) return 0;
    float x = dot(v, gCam.right) / z / gCam.tanHx;
    float y = dot(v, gCam.up) / z / gCam.tanHy;
    *sx = (x * 0.5f + 0.5f) * (float)gW;
    *sy = (0.5f - y * 0.5f) * (float)gH;
    return 1;
}

// ---------------------------------------------------------------- GPU
typedef struct { float x, y, z, w; } F4;
typedef struct {
    F4 camPos, camFwd, camRight, camUp;
    int32_t dims[4], dims2[4], dims3[4];
    F4 bh, misc, misc3;
} CB;

enum { S_RENDER, S_BRIGHT, S_BV1, S_BH2, S_BV2, S_COMPOSE, S_CELLS, S_COUNT };
static const char *kEntry[S_COUNT] = { "CS_Render", "CS_BrightH1", "CS_BlurV1", "CS_BlurH2", "CS_BlurV2", "CS_Compose", "CS_Cells" };

#define REL(p) do { if (p) { IUnknown_Release((IUnknown *)(p)); (p) = NULL; } } while (0)

typedef ID3D11ShaderResourceView *SRVp;
typedef ID3D11UnorderedAccessView *UAVp;

static ID3D11Device *gDev;
static ID3D11DeviceContext *gCtx;
static ID3D11ComputeShader *gCS[S_COUNT];
static ID3D11Buffer *gCBuf, *gCellsBuf, *gStage[2];
static ID3D11Texture2D *gTHdr, *gTBA, *gTBB, *gTBC, *gTPix, *gTOverlay;
static SRVp gSHdr, gSBA, gSBB, gSBC, gSPix, gSOverlay;
static UAVp gUHdr, gUBA, gUBB, gUBC, gUPix, gUCells;
static char gGpuName[128] = "?";
static int  gStageValid[2], gStageIdx;
static uint32_t *gCellRaw;                    // cells * 2 uints read back from the GPU

// GPU timing (timestamp queries, read two frames later so nothing stalls)
enum { kQRing = 8 };
static ID3D11Query *gQD[kQRing], *gQ0[kQRing], *gQ1[kQRing];
static int gQHave[kQRing], gQFrame;
static float gGpuMs;

static void ReleaseTargets(void)
{
    REL(gSHdr); REL(gSBA); REL(gSBB); REL(gSBC); REL(gSPix); REL(gSOverlay);
    REL(gUHdr); REL(gUBA); REL(gUBB); REL(gUBC); REL(gUPix); REL(gUCells);
    REL(gTHdr); REL(gTBA); REL(gTBB); REL(gTBC); REL(gTPix); REL(gTOverlay);
    REL(gCellsBuf); REL(gStage[0]); REL(gStage[1]);
    gStageValid[0] = gStageValid[1] = 0;
    free(gCellRaw);
    gCellRaw = NULL;
}

static ID3DBlob *LoadShader(const char *entry, const wchar_t *hlsl, const wchar_t *dir)
{
    wchar_t cso[MAX_PATH];
    swprintf(cso, MAX_PATH, L"%ls\\blackhole_%hs.cso", dir, entry);
    WIN32_FILE_ATTRIBUTE_DATA a, b;
    if (GetFileAttributesExW(cso, GetFileExInfoStandard, &a) && GetFileAttributesExW(hlsl, GetFileExInfoStandard, &b) &&
        CompareFileTime(&a.ftLastWriteTime, &b.ftLastWriteTime) > 0) {
        ID3DBlob *blob = NULL;
        if (SUCCEEDED(D3DReadFileToBlob(cso, &blob))) return blob;
    }
    ID3DBlob *code = NULL, *err = NULL;
    HRESULT hr = D3DCompileFromFile(hlsl, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (FAILED(hr)) {
        const char *msg = err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "(no message)";
        fprintf(stderr, "shader %s failed:\n%s\n", entry, msg);
        MessageBoxA(NULL, msg, "blackhole shader error", MB_OK | MB_ICONERROR);
        REL(err);
        return NULL;
    }
    REL(err);
    D3DWriteBlobToFile(code, cso, TRUE);
    return code;
}

static int InitGpu(void)
{
    IDXGIFactory1 *fac = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&fac))) { fputs("no DXGI\n", stderr); return 0; }
    IDXGIAdapter1 *best = NULL;
    SIZE_T bestMem = 0;
    DXGI_ADAPTER_DESC1 bd;
    memset(&bd, 0, sizeof bd);
    for (UINT i = 0;; i++) {
        IDXGIAdapter1 *ad = NULL;
        if (IDXGIFactory1_EnumAdapters1(fac, i, &ad) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d;
        IDXGIAdapter1_GetDesc1(ad, &d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && (d.DedicatedVideoMemory >= bestMem || !best)) {
            REL(best);
            best = ad;
            bestMem = d.DedicatedVideoMemory;
            bd = d;
        } else {
            REL(ad);
        }
    }
    REL(fac);
    if (!best) { fputs("no hardware GPU found\n", stderr); return 0; }
    WideCharToMultiByte(CP_UTF8, 0, bd.Description, -1, gGpuName, (int)sizeof gGpuName, NULL, NULL);

    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice((IDXGIAdapter *)best, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &gDev, &fl, &gCtx);
    REL(best);
    if (FAILED(hr)) { fprintf(stderr, "D3D11CreateDevice failed 0x%08lx\n", (unsigned long)hr); return 0; }

    wchar_t exe[MAX_PATH], dir[MAX_PATH], hlsl[MAX_PATH];
    GetModuleFileNameW(NULL, exe, MAX_PATH);
    wcscpy(dir, exe);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;
    swprintf(hlsl, MAX_PATH, L"%ls\\blackhole.hlsl", dir);
    for (int i = 0; i < S_COUNT; i++) {
        ID3DBlob *blob = LoadShader(kEntry[i], hlsl, dir);
        if (!blob) return 0;
        hr = ID3D11Device_CreateComputeShader(gDev, ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), NULL, &gCS[i]);
        REL(blob);
        if (FAILED(hr)) { fprintf(stderr, "CreateComputeShader %s failed 0x%08lx\n", kEntry[i], (unsigned long)hr); return 0; }
    }
    D3D11_BUFFER_DESC cb;
    memset(&cb, 0, sizeof cb);
    cb.ByteWidth = (UINT)sizeof(CB);
    cb.Usage = D3D11_USAGE_DEFAULT;
    cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(ID3D11Device_CreateBuffer(gDev, &cb, NULL, &gCBuf))) return 0;

    for (int i = 0; i < kQRing; i++) {
        D3D11_QUERY_DESC q;
        memset(&q, 0, sizeof q);
        q.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        ID3D11Device_CreateQuery(gDev, &q, &gQD[i]);
        q.Query = D3D11_QUERY_TIMESTAMP;
        ID3D11Device_CreateQuery(gDev, &q, &gQ0[i]);
        ID3D11Device_CreateQuery(gDev, &q, &gQ1[i]);
    }
    return 1;
}

static void GpuTimingBegin(void)
{
    int s = gQFrame % kQRing;
    ID3D11DeviceContext_Begin(gCtx, (ID3D11Asynchronous *)gQD[s]);
    ID3D11DeviceContext_End(gCtx, (ID3D11Asynchronous *)gQ0[s]);
}

static void GpuTimingEnd(void)
{
    int s = gQFrame % kQRing;
    ID3D11DeviceContext_End(gCtx, (ID3D11Asynchronous *)gQ1[s]);
    ID3D11DeviceContext_End(gCtx, (ID3D11Asynchronous *)gQD[s]);
    gQHave[s] = 1;
    int r = (gQFrame + 1) % kQRing;                      // the oldest slot: a few frames old, so the GPU is done with it
    if (gQHave[r]) {
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
        if (ID3D11DeviceContext_GetData(gCtx, (ID3D11Asynchronous *)gQD[r], &dj, sizeof dj, D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK) {
            UINT64 a, b;
            if (ID3D11DeviceContext_GetData(gCtx, (ID3D11Asynchronous *)gQ0[r], &a, sizeof a, 0) == S_OK &&
                ID3D11DeviceContext_GetData(gCtx, (ID3D11Asynchronous *)gQ1[r], &b, sizeof b, 0) == S_OK && dj.Frequency) {
                float ms = (float)((double)(b - a) * 1000.0 / (double)dj.Frequency);   // a disjoint flag only means the GPU clock moved
                if (ms >= 0.0f && ms < 200.0f) gGpuMs = gGpuMs <= 0.0f ? ms : gGpuMs + (ms - gGpuMs) * 0.15f;
            }
            gQHave[r] = 0;
        }
    }
    gQFrame++;
}

static int MakeTex(DXGI_FORMAT fmt, ID3D11Texture2D **t, SRVp *s, UAVp *u)
{
    D3D11_TEXTURE2D_DESC d;
    memset(&d, 0, sizeof d);
    d.Width = (UINT)gW; d.Height = (UINT)gH; d.MipLevels = 1; d.ArraySize = 1; d.Format = fmt;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(ID3D11Device_CreateTexture2D(gDev, &d, NULL, t))) return 0;
    if (FAILED(ID3D11Device_CreateShaderResourceView(gDev, (ID3D11Resource *)*t, NULL, s))) return 0;
    if (FAILED(ID3D11Device_CreateUnorderedAccessView(gDev, (ID3D11Resource *)*t, NULL, u))) return 0;
    return 1;
}

// all textures are allocated at the output size; the internal (dynamic-resolution) render uses a corner of them
static int CreateTargets(int cellsX, int cellsY)
{
    ReleaseTargets();
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTHdr, &gSHdr, &gUHdr)) return 0;
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTBA, &gSBA, &gUBA)) return 0;
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTBB, &gSBB, &gUBB)) return 0;
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTBC, &gSBC, &gUBC)) return 0;
    if (cellsX > 0) {
        if (!MakeTex(DXGI_FORMAT_R32G32B32A32_FLOAT, &gTPix, &gSPix, &gUPix)) return 0;
        D3D11_BUFFER_DESC d;
        memset(&d, 0, sizeof d);
        d.ByteWidth = (UINT)(8 * cellsX * cellsY);
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        d.StructureByteStride = 8;
        if (FAILED(ID3D11Device_CreateBuffer(gDev, &d, NULL, &gCellsBuf))) return 0;
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud;
        memset(&ud, 0, sizeof ud);
        ud.Format = DXGI_FORMAT_UNKNOWN;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.NumElements = (UINT)(cellsX * cellsY);
        if (FAILED(ID3D11Device_CreateUnorderedAccessView(gDev, (ID3D11Resource *)gCellsBuf, &ud, &gUCells))) return 0;
        for (int i = 0; i < 2; i++) {
            D3D11_BUFFER_DESC sd;
            memset(&sd, 0, sizeof sd);
            sd.ByteWidth = (UINT)(8 * cellsX * cellsY);
            sd.Usage = D3D11_USAGE_STAGING;
            sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            if (FAILED(ID3D11Device_CreateBuffer(gDev, &sd, NULL, &gStage[i]))) return 0;
        }
        gCellRaw = malloc((size_t)8 * (size_t)cellsX * (size_t)cellsY);
        return gCellRaw != NULL;
    }
    return 1;
}

static void RunPass(int shader, SRVp s0, SRVp s1, SRVp s2, SRVp s3, SRVp s4, SRVp s7,
                    UAVp u0, UAVp u2, UAVp u3, UAVp u4, UINT gx, UINT gy)
{
    SRVp S[8] = { s0, s1, s2, s3, s4, 0, 0, s7 };
    UAVp U[6] = { u0, 0, u2, u3, u4, 0 };
    SRVp noS[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    UAVp noU[6] = { 0, 0, 0, 0, 0, 0 };
    ID3D11DeviceContext_CSSetShader(gCtx, gCS[shader], NULL, 0);
    ID3D11DeviceContext_CSSetShaderResources(gCtx, 0, 8, S);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(gCtx, 0, 6, U, NULL);
    ID3D11DeviceContext_Dispatch(gCtx, gx, gy, 1);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(gCtx, 0, 6, noU, NULL);
    ID3D11DeviceContext_CSSetShaderResources(gCtx, 0, 8, noS);
}

// scene and bloom at the internal resolution
static void GpuScene(const CB *cb)
{
    ID3D11DeviceContext_UpdateSubresource(gCtx, (ID3D11Resource *)gCBuf, 0, NULL, cb, 0, 0);
    ID3D11DeviceContext_CSSetConstantBuffers(gCtx, 0, 1, &gCBuf);
    UINT gx = (UINT)((gIW + 7) / 8), gy = (UINT)((gIH + 7) / 8);
    RunPass(S_RENDER, 0, 0, 0, 0, 0, 0, gUHdr, 0, 0, 0, gx, gy);
    RunPass(S_BRIGHT, gSHdr, 0, 0, 0, 0, 0, 0, gUBA, 0, 0, gx, gy);
    RunPass(S_BV1, 0, gSBA, 0, 0, 0, 0, 0, gUBB, 0, 0, gx, gy);
    RunPass(S_BH2, 0, gSBB, 0, 0, 0, 0, 0, gUBA, 0, 0, gx, gy);
    RunPass(S_BV2, 0, gSBA, 0, 0, 0, 0, 0, gUBC, 0, 0, gx, gy);
}

// tone map into `target` (the float pixel texture for the terminal, the swap chain back buffer for the window)
static void GpuCompose(UAVp target)
{
    RunPass(S_COMPOSE, gSHdr, 0, gSBB, gSBC, 0, gSOverlay, 0, 0, target, 0, (UINT)((gW + 7) / 8), (UINT)((gH + 7) / 8));
}

// terminal: fit cells on the GPU and read them back (this frame when sync, else the previous frame)
static const uint32_t *GpuCells(int sync)
{
    RunPass(S_CELLS, 0, 0, 0, 0, gSPix, 0, 0, 0, 0, gUCells, (UINT)((gW / gSx + 7) / 8), (UINT)(((gH / gSy) + 7) / 8));
    int cur = gStageIdx;
    ID3D11DeviceContext_CopyResource(gCtx, (ID3D11Resource *)gStage[cur], (ID3D11Resource *)gCellsBuf);
    gStageValid[cur] = 1;
    int rd = sync ? cur : 1 - cur;
    gStageIdx = 1 - cur;
    if (!gStageValid[rd]) return NULL;
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ID3D11DeviceContext_Map(gCtx, (ID3D11Resource *)gStage[rd], 0, D3D11_MAP_READ, 0, &m))) return NULL;
    memcpy(gCellRaw, m.pData, (size_t)8 * (size_t)(gW / gSx) * (size_t)(gH / gSy));
    ID3D11DeviceContext_Unmap(gCtx, (ID3D11Resource *)gStage[rd], 0);
    return gCellRaw;
}

static F4 f4(V3 v, float w) { F4 r; r.x = v.x; r.y = v.y; r.z = v.z; r.w = w; return r; }

// ---------------------------------------------------------------- scene state, camera, input (shared by both front ends)
static volatile LONG gRun = 1;
static BOOL WINAPI OnCtrl(DWORD type)
{
    (void)type;
    InterlockedExchange(&gRun, 0);
    return TRUE;
}

enum { K_LEFT = 1001, K_RIGHT, K_UP, K_DOWN, K_FULL };

static int   gGalaxy;                                     // Milky Way view: fixed, no navigation
static int   gAuto = 1, gLabels = 1, gPaused;
static float gAz = 0.7f, gEl = 0.10f, gDist = 34.0f;      // black hole camera: azimuth, elevation, distance (Schwarzschild radii)
static float gElT = 0.10f, gDistT = 34.0f;
static float gSpin = 1.0f;                                // disk rotation speed multiplier
static float gRealT, gSimT;

static void FillCamera(void)
{
    if (gGalaxy) {
        V3 pos = v3(0.0f, kGalaxyDist * cosf(kGalaxyTilt), kGalaxyDist * sinf(kGalaxyTilt));
        BuildCam(pos, v3(0.0f, 0.0f, 0.0f), v3(0.0f, 0.0f, -1.0f), kTanHalfFovGal, 0.0f);
    } else {
        V3 pos = v3(gDist * cosf(gEl) * sinf(gAz), gDist * sinf(gEl), gDist * cosf(gEl) * cosf(gAz));
        BuildCam(pos, v3(0.0f, 0.0f, 0.0f), v3(0.0f, 1.0f, 0.0f), kTanHalfFovBH, kBHRoll);
    }
}

static void StepWorld(float dt, int testMode)
{
    gRealT += dt;
    if (!gPaused) gSimT += dt;
    if (!gGalaxy) {
        if (gAuto && !testMode) gAz += dt * 0.045f;
        float k = 1.0f - expf(-dt * 5.0f);
        gEl += (gElT - gEl) * k;
        gDist += (gDistT - gDist) * k;
    }
    FillCamera();
}

static void Preset(int k)
{
    static const float el[6] = { 0.10f, 0.03f, 0.30f, 0.90f, 1.45f, 0.12f };
    static const float di[6] = { 34.0f, 36.0f, 32.0f, 30.0f, 28.0f, 10.0f };
    gElT = el[k];
    gDistT = di[k];
}

// returns 1 to quit
static int HandleKey(int c)
{
    if (c == 'q' || c == 'Q' || c == 27) return 1;
    if (c == 'm' || c == 'M') { gGalaxy = !gGalaxy; return 0; }
    if (c == 'l' || c == 'L') gLabels = !gLabels;
    else if (c == ' ') gPaused = !gPaused;
    if (gGalaxy) return 0;                               // the Milky Way is a fixed picture: nothing below applies
    if (c == K_LEFT) { gAz -= 0.1f; gAuto = 0; }
    else if (c == K_RIGHT) { gAz += 0.1f; gAuto = 0; }
    else if (c == K_UP) gElT = Clampf(gElT + 0.07f, -1.45f, 1.45f);
    else if (c == K_DOWN) gElT = Clampf(gElT - 0.07f, -1.45f, 1.45f);
    else if (c >= '0' && c <= '5') Preset(c - '0');
    else if (c == 't' || c == 'T') gAuto = !gAuto;
    else if (c == 'a' || c == 'A') { gAz -= 0.1f; gAuto = 0; }
    else if (c == 'd' || c == 'D') { gAz += 0.1f; gAuto = 0; }
    else if (c == 'w' || c == 'W') gElT = Clampf(gElT + 0.07f, -1.45f, 1.45f);
    else if (c == 's' || c == 'S') gElT = Clampf(gElT - 0.07f, -1.45f, 1.45f);
    else if (c == '+' || c == '=') gDistT = Clampf(gDistT * 0.88f, 4.5f, 150.0f);
    else if (c == '-' || c == '_') gDistT = Clampf(gDistT * 1.14f, 4.5f, 150.0f);
    else if (c == ']') gSpin = Clampf(gSpin * 1.4f, 0.05f, 30.0f);
    else if (c == '[') gSpin = Clampf(gSpin / 1.4f, 0.05f, 30.0f);
    return 0;
}

static void FillCB(CB *cb, int ssaa, int windowed)
{
    memset(cb, 0, sizeof *cb);
    cb->camPos = f4(gCam.pos, gRealT);
    cb->camFwd = f4(gCam.fwd, gSimT * gSpin);
    cb->camRight = f4(gCam.right, gCam.tanHx);
    cb->camUp = f4(gCam.up, gCam.tanHy);
    cb->dims[0] = gIW; cb->dims[1] = gIH; cb->dims[2] = ssaa; cb->dims[3] = gW / gSx;
    cb->dims2[0] = gH / gSy; cb->dims2[1] = gSx; cb->dims2[2] = gSy; cb->dims2[3] = gGalaxy;
    cb->dims3[0] = gW; cb->dims3[1] = gH; cb->dims3[2] = windowed; cb->dims3[3] = gExtraOct;
    cb->bh.x = kDiskIn; cb->bh.y = kDiskOut; cb->bh.z = 1.0f; cb->bh.w = gDist;
    cb->misc.x = gGalaxy ? 0.55f : 0.45f;
    cb->misc.y = gGalaxy ? 0.50f : 0.3f;
    cb->misc.z = gGalaxy ? 0.85f : 1.25f;
    cb->misc.w = 2.0f * gCam.tanHy / (float)gIH;
    cb->misc3.x = gSimT * 0.03f;                         // galaxy pattern rotation
    cb->misc3.y = 1.0f;
}

// ---------------------------------------------------------------- labels (positions in output pixels)
typedef struct { float x, y; const char *text; int centered; } Lab;

static int BuildLabels(Lab *out)
{
    int n = 0;
    if (!gGalaxy) {
        float cx = (float)gW * 0.5f, cy = (float)gH * 0.5f;
        float ang = asinf(Clampf(2.598f / gDist, 0.0f, 0.99f));
        float shadowPx = tanf(ang) * (float)gH / (2.0f * gCam.tanHy);
        out[n++] = (Lab){ cx, cy, "Event horizon", 1 };
        out[n++] = (Lab){ cx + shadowPx * 1.15f, cy - shadowPx * 0.85f, "Photon ring", 0 };
        float sx, sy;
        if (Project(v3(kDiskOut * 0.62f, 0.0f, 0.0f), &sx, &sy)) out[n++] = (Lab){ sx, sy, "Accretion disk", 0 };
        // the side of the disk moving toward the camera is the bright one
        V3 camDir = norm(v3(gCam.pos.x, 0.0f, gCam.pos.z));
        V3 side = norm(cross(v3(0.0f, 1.0f, 0.0f), camDir));
        for (int s = -1; s <= 1; s += 2) {
            V3 p = mulf(side, 6.0f * (float)s);
            V3 vel = norm(v3(-p.z, 0.0f, p.x));
            if (dot(vel, sub(gCam.pos, p)) > 0.0f && Project(p, &sx, &sy)) out[n++] = (Lab){ sx, sy, "Approaching side (Doppler-brightened)", 0 };
        }
    } else {
        float rotT = gSimT * 0.03f;
        struct { float R, th; const char *name; } pts[6] = {
            { 0.53f, 2.30f, "Sun (you are here)" }, { 0.70f, 8.82f, "Perseus Arm" }, { 0.42f, 7.83f, "Sagittarius Arm" },
            { 0.62f, 11.35f, "Scutum-Centaurus Arm" }, { 0.30f, 9.29f, "Norma Arm" }, { 0.0f, 0.0f, "Galactic center (Sgr A*)" } };
        for (int i = 0; i < 6; i++) {
            float th = pts[i].th - rotT;
            V3 p = v3(pts[i].R * cosf(th), 0.0f, pts[i].R * sinf(th));
            float sx, sy;
            if (Project(p, &sx, &sy)) out[n++] = (Lab){ sx, sy, pts[i].name, 0 };
        }
        float sx, sy;
        if (Project(v3(1.45f, -0.35f, 0.55f), &sx, &sy)) out[n++] = (Lab){ sx, sy, "Large Magellanic Cloud", 0 };
        if (Project(v3(1.68f, -0.5f, 0.95f), &sx, &sy)) out[n++] = (Lab){ sx, sy, "Small Magellanic Cloud", 0 };
    }
    return n;
}

static void InitView(int preset, int galaxy)
{
    gGalaxy = galaxy;
    if (preset >= 0 && preset <= 5) {
        Preset(preset);
        gEl = gElT;
        gDist = gDistT;
        gAuto = 0;
    }
    FillCamera();
}

// ================================================================= terminal front end
typedef struct { uint8_t fr, fg, fb, br, bg, bb; uint32_t cp; } Cell;
static Cell *gCells, *gPrev;
static int gCols, gRows;
static char *gOut;

static uint32_t SextantCp(int p)    // 6-bit mask, bit0 = top-left, row-major
{
    if (p == 0 || p == 63) return ' ';
    if (p == 21) return 0x258C;
    if (p == 42) return 0x2590;
    return 0x1FB00u + (unsigned)(p - 1) - (p > 21 ? 1u : 0u) - (p > 42 ? 1u : 0u);
}

static uint32_t GlyphFor(int mode, int m)
{
    static const uint32_t quad[16] = { 0x20, 0x2598, 0x259D, 0x2580, 0x2596, 0x258C, 0x259E, 0x259B,
                                       0x2597, 0x259A, 0x2590, 0x259C, 0x2584, 0x2599, 0x259F, 0x2588 };
    if (mode == M_HALF) return m ? 0x2580 : ' ';
    if (mode == M_QUAD) return quad[m & 15];
    return SextantCp(m);
}

static void CellsFromGpu(const uint32_t *raw)
{
    int cellsX = gCols, cellsY = gRows - 1;
    for (int i = 0; i < cellsX * cellsY; i++) {
        uint32_t w0 = raw[2 * i], w1 = raw[2 * i + 1];
        Cell *c = &gCells[i];
        memset(c, 0, sizeof *c);
        c->fr = (uint8_t)(w0 >> 16); c->fg = (uint8_t)(w0 >> 8); c->fb = (uint8_t)w0;
        c->br = (uint8_t)(w1 >> 16); c->bg = (uint8_t)(w1 >> 8); c->bb = (uint8_t)w1;
        int mask = (int)((w0 >> 24) & 63u);
        c->cp = mask ? GlyphFor(gMode, mask) : ' ';
    }
    for (int cx = 0; cx < gCols; cx++) memset(&gCells[(gRows - 1) * gCols + cx], 0, sizeof(Cell));
}

static void PutText(int cx, int cy, const char *s, uint8_t r, uint8_t g, uint8_t b)
{
    for (; *s; s++, cx++) {
        if ((unsigned)cx >= (unsigned)gCols || (unsigned)cy >= (unsigned)gRows) continue;
        Cell *c = &gCells[cy * gCols + cx];
        uint8_t nr = (uint8_t)((c->fr + c->br) / 6), ng = (uint8_t)((c->fg + c->bg) / 6), nb = (uint8_t)((c->fb + c->bb) / 6);
        c->br = nr; c->bg = ng; c->bb = nb;
        c->fr = r; c->fg = g; c->fb = b;
        c->cp = (uint32_t)(unsigned char)*s;
    }
}

static int TextFits(int cx, int cy, int len)
{
    if (cx < 0 || cx + len > gCols || cy < 0 || cy >= gRows - 1) return 0;
    for (int i = 0; i < len; i++) {
        uint32_t cp = gCells[cy * gCols + cx + i].cp;
        if (cp > ' ' && cp < 0x80) return 0;
    }
    return 1;
}

static void WriteUtf8(char **pp, uint32_t cp)
{
    char *p = *pp;
    if (cp < 0x80) *p++ = (char)cp;
    else if (cp < 0x800) { *p++ = (char)(0xC0 | (cp >> 6)); *p++ = (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { *p++ = (char)(0xE0 | (cp >> 12)); *p++ = (char)(0x80 | ((cp >> 6) & 63)); *p++ = (char)(0x80 | (cp & 63)); }
    else { *p++ = (char)(0xF0 | (cp >> 18)); *p++ = (char)(0x80 | ((cp >> 12) & 63)); *p++ = (char)(0x80 | ((cp >> 6) & 63)); *p++ = (char)(0x80 | (cp & 63)); }
    *pp = p;
}

static int CellEq(const Cell *a, const Cell *b)
{
    return a->fr == b->fr && a->fg == b->fg && a->fb == b->fb && a->br == b->br && a->bg == b->bg && a->bb == b->bb && a->cp == b->cp;
}

static size_t EmitFrame(int full)
{
    char *p = gOut;
    int curFg = -1, curBg = -1;
    for (int r = 0; r < gRows; r++) {
        int expect = -1;
        for (int c = 0; c < gCols; c++) {
            Cell *cell = &gCells[r * gCols + c];
            Cell *prev = &gPrev[r * gCols + c];
            if (!full && CellEq(cell, prev)) continue;
            if (expect != c) p += sprintf(p, "\x1b[%d;%dH", r + 1, c + 1);
            int bg = (cell->br << 16) | (cell->bg << 8) | cell->bb;
            int fg = (cell->fr << 16) | (cell->fg << 8) | cell->fb;
            if (bg != curBg) {
                p += sprintf(p, "\x1b[48;2;%d;%d;%dm", (bg >> 16) & 255, (bg >> 8) & 255, bg & 255);
                curBg = bg;
            }
            if (cell->cp != ' ' && fg != curFg) {
                p += sprintf(p, "\x1b[38;2;%d;%d;%dm", (fg >> 16) & 255, (fg >> 8) & 255, fg & 255);
                curFg = fg;
            }
            WriteUtf8(&p, cell->cp ? cell->cp : ' ');
            expect = c + 1;
            *prev = *cell;
        }
    }
    return (size_t)(p - gOut);
}

static void ConsoleSize(HANDLE hOut, int *cols, int *rows)
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(hOut, &csbi)) {
        *cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        *rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    } else {
        *cols = 120;
        *rows = 30;
    }
}

static void FreeCpuBuffers(void)
{
    free(gCells); free(gPrev); free(gOut);
    gCells = gPrev = NULL; gOut = NULL;
}

static int ResizeTerminal(int cols, int rows)
{
    FreeCpuBuffers();
    int mode = gModePref;
    for (;;) {
        SetMode(mode);
        if (mode == M_HALF || cols * gSx * (rows - 1) * gSy <= kMaxPixels) break;
        mode--;
    }
    gCols = cols;
    gRows = rows;
    gW = gIW = cols * gSx;
    gH = gIH = (rows - 1) * gSy;
    gSsaa = gW * gH <= 120000 ? 2 : 1;
    gExtraOct = 0;
    gCells = calloc((size_t)cols * (size_t)rows, sizeof(Cell));
    gPrev = calloc((size_t)cols * (size_t)rows, sizeof(Cell));
    gOut = malloc((size_t)cols * (size_t)rows * 64 + 4096);
    return gCells && gPrev && gOut && CreateTargets(cols, rows - 1);
}

static int RunTerminal(int preset, int galaxy, int testFrames, int forceC, int forceR)
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  oldMode = 0;
    UINT   oldCp = GetConsoleOutputCP();
    BOOL   haveMode = GetConsoleMode(hOut, &oldMode);
    if (haveMode) {
        SetConsoleMode(hOut, oldMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        SetConsoleOutputCP(CP_UTF8);
    }
    SetConsoleCtrlHandler(OnCtrl, TRUE);
    timeBeginPeriod(1);

    static const char enter[] = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J";
    DWORD wr;
    WriteFile(hOut, enter, (DWORD)(sizeof enter - 1), &wr, NULL);

    LARGE_INTEGER freq, prev, now, frameStart;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);

    int frames = 0, lastC = 0, lastR = 0, forceFull = 1, failed = 0;
    float fps = 30.0f, cpuMs = 0.0f;
    InitView(preset, galaxy);

    while (gRun) {
        QueryPerformanceCounter(&frameStart);
        if (_kbhit()) {
            int c = _getch();
            if (c == 0 || c == 224) {
                int c2 = _getch();
                c = c2 == 75 ? K_LEFT : c2 == 77 ? K_RIGHT : c2 == 72 ? K_UP : c2 == 80 ? K_DOWN : 0;
            } else if (c == 'n' || c == 'N') {
                gModePref = (gModePref + 2) % 3;
                gModeChanged = 1;
                c = 0;
            }
            if (c && HandleKey(c)) break;
        }

        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - prev.QuadPart) / (double)freq.QuadPart);
        prev = now;
        if (testFrames) dt = 1.0f / 30.0f;
        if (dt > 0.2f) dt = 0.2f;
        if (dt > 1e-4f) fps += (1.0f / dt - fps) * 0.1f;

        int tc, tr;
        if (forceC > 0 && forceR > 0) { tc = forceC; tr = forceR; }
        else ConsoleSize(hOut, &tc, &tr);
        if (tc < 30) tc = 30;
        if (tr < 10) tr = 10;
        if (tc != lastC || tr != lastR || gModeChanged) {
            gModeChanged = 0;
            if (!ResizeTerminal(tc, tr)) { failed = 1; break; }
            lastC = tc;
            lastR = tr;
            forceFull = 1;
            WriteFile(hOut, "\x1b[2J", 4, &wr, NULL);
        }

        StepWorld(dt, testFrames != 0);

        CB cb;
        FillCB(&cb, gSsaa, 0);
        GpuTimingBegin();
        GpuScene(&cb);
        GpuCompose(gUPix);
        const uint32_t *raw = GpuCells(testFrames != 0);
        GpuTimingEnd();
        if (raw) {
            CellsFromGpu(raw);
            if (gLabels) {
                Lab labs[16];
                int n = BuildLabels(labs);
                for (int i = 0; i < n; i++) {
                    int len = (int)strlen(labs[i].text);
                    int cx = (int)(labs[i].x / (float)gSx), cy = (int)(labs[i].y / (float)gSy);
                    if (labs[i].centered) cx -= len / 2; else cx += 2;
                    if (TextFits(cx - 1, cy, len + 2)) PutText(cx, cy, labs[i].text, 205, 210, 225);          // keep a cell of air between labels
                    else if (!labs[i].centered && TextFits(cx - 5 - len, cy, len + 2)) PutText(cx - 4 - len, cy, labs[i].text, 205, 210, 225);
                }
            }
            char hud[320];
            if (gGalaxy)
                snprintf(hud, sizeof hud, " MILKY WAY  fixed view, nothing to navigate  |  [m] back to the black hole  [l] labels  [q] quit  |  GPU %s %.1fms  %s %dx%d  %.0ffps",
                         gGpuName, gGpuMs, kModeName[gMode], gW, gH, fps);
            else
                snprintf(hud, sizeof hud, " BLACK HOLE  d=%.1f rs  incl %.0f deg  spin x%.2f%s%s  |  [m] Milky Way [1-5] views [t] auto-orbit [arrows] orbit [+/-] zoom [ [ ] ] spin [l] [q]  |  GPU %.1fms %s %dx%d %.0ffps cpu %.1fms",
                         gDist, gEl * 57.2958f, gSpin, gPaused ? " (paused)" : "", gAuto ? " auto-orbit" : "", gGpuMs, kModeName[gMode], gW, gH, fps, cpuMs);
            PutText(0, gRows - 1, hud, 140, 145, 160);
            size_t n = EmitFrame(forceFull);
            forceFull = 0;
            WriteFile(hOut, gOut, (DWORD)n, &wr, NULL);
        }

        frames++;
        if (testFrames && frames >= testFrames) break;
        QueryPerformanceCounter(&now);
        double ms = (double)(now.QuadPart - frameStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
        cpuMs += ((float)ms - cpuMs) * 0.1f;
        if (!testFrames && ms < gTickMs) Sleep((DWORD)(gTickMs - ms));
    }

    static const char leave[] = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l";
    WriteFile(hOut, leave, (DWORD)(sizeof leave - 1), &wr, NULL);
    if (haveMode) {
        SetConsoleMode(hOut, oldMode);
        SetConsoleOutputCP(oldCp);
    }
    timeEndPeriod(1);
    ReleaseTargets();
    FreeCpuBuffers();
    if (failed) fputs("failed to allocate GPU resources\n", stderr);
    return failed;
}

// ================================================================= window front end
static HWND gHwnd;
static int  gWinW = 1600, gWinH = 900, gWinDirty, gFullscreen, gDragging, gLastMx, gLastMy;
static int  gKeyQ[128], gKeyN;
static WINDOWPLACEMENT gWp;
static IDXGISwapChain1 *gSwap;
static ID3D11Texture2D *gBackTex;
static UAVp gBackUav;

// text overlay: a GDI DIB drawn on the CPU, uploaded only where text changed, blended in the compose shader
typedef struct { int x0, y0, x1, y1; } Rect;
static HDC gDc;
static HBITMAP gDib;
static uint8_t *gDibBits;
static HFONT gFont;
static int gDibW, gDibH;
static Rect gRectPrev[48], gRectCur[48];
static int gNPrev, gNCur;

static void PushKey(int k) { if (gKeyN < 128) gKeyQ[gKeyN++] = k; }

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m) {
    case WM_SIZE:
        if (w != SIZE_MINIMIZED && LOWORD(l) > 0 && HIWORD(l) > 0) { gWinW = LOWORD(l); gWinH = HIWORD(l); gWinDirty = 1; }
        return 0;
    case WM_CLOSE:
        InterlockedExchange(&gRun, 0);
        return 0;
    case WM_CHAR:
        PushKey((int)w);
        return 0;
    case WM_KEYDOWN:
        if (w == VK_LEFT) PushKey(K_LEFT);
        else if (w == VK_RIGHT) PushKey(K_RIGHT);
        else if (w == VK_UP) PushKey(K_UP);
        else if (w == VK_DOWN) PushKey(K_DOWN);
        else if (w == VK_F11) PushKey(K_FULL);
        return 0;
    case WM_LBUTTONDOWN:
        SetCapture(h);
        gDragging = 1;
        gLastMx = GET_X_LPARAM(l);
        gLastMy = GET_Y_LPARAM(l);
        return 0;
    case WM_LBUTTONUP:
        gDragging = 0;
        ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if (gDragging && !gGalaxy) {                       // the Milky Way view ignores the mouse
            int x = GET_X_LPARAM(l), y = GET_Y_LPARAM(l);
            gAz += (float)(x - gLastMx) * 0.006f;
            gEl = Clampf(gEl + (float)(y - gLastMy) * 0.005f, -1.45f, 1.45f);
            gElT = gEl;
            gAuto = 0;
            gLastMx = x;
            gLastMy = y;
        } else if (gDragging) {
            gLastMx = GET_X_LPARAM(l);
            gLastMy = GET_Y_LPARAM(l);
        }
        return 0;
    case WM_MOUSEWHEEL:
        if (!gGalaxy) gDistT = Clampf(gDistT * (GET_WHEEL_DELTA_WPARAM(w) > 0 ? 0.9f : 1.11f), 4.5f, 150.0f);
        return 0;
    case WM_LBUTTONDBLCLK:
        PushKey(K_FULL);
        return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

static void ToggleFullscreen(void)
{
    DWORD style = (DWORD)GetWindowLongA(gHwnd, GWL_STYLE);
    if (!gFullscreen) {
        MONITORINFO mi;
        memset(&mi, 0, sizeof mi);
        mi.cbSize = sizeof mi;
        gWp.length = sizeof gWp;
        GetWindowPlacement(gHwnd, &gWp);
        GetMonitorInfoA(MonitorFromWindow(gHwnd, MONITOR_DEFAULTTONEAREST), &mi);
        SetWindowLongA(gHwnd, GWL_STYLE, (LONG)(style & ~WS_OVERLAPPEDWINDOW));
        SetWindowPos(gHwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
    } else {
        SetWindowLongA(gHwnd, GWL_STYLE, (LONG)(style | WS_OVERLAPPEDWINDOW));
        SetWindowPlacement(gHwnd, &gWp);
        SetWindowPos(gHwnd, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    gFullscreen = !gFullscreen;
}

static int MakeBackUav(void)
{
    REL(gBackUav);
    REL(gBackTex);
    if (FAILED(IDXGISwapChain1_GetBuffer(gSwap, 0, &IID_ID3D11Texture2D, (void **)&gBackTex))) return 0;
    if (FAILED(ID3D11Device_CreateUnorderedAccessView(gDev, (ID3D11Resource *)gBackTex, NULL, &gBackUav))) {
        MessageBoxA(NULL, "This GPU cannot write the swap chain from a compute shader.", "blackhole", MB_OK | MB_ICONERROR);
        return 0;
    }
    return 1;
}

static int CreateSwap(void)
{
    IDXGIFactory2 *f2 = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory2, (void **)&f2))) return 0;
    DXGI_SWAP_CHAIN_DESC1 sd;
    memset(&sd, 0, sizeof sd);
    sd.Width = (UINT)gWinW;
    sd.Height = (UINT)gWinH;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_UNORDERED_ACCESS | DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    HRESULT hr = IDXGIFactory2_CreateSwapChainForHwnd(f2, (IUnknown *)gDev, gHwnd, &sd, NULL, NULL, &gSwap);
    if (SUCCEEDED(hr)) {                                // one queued frame: less input lag, and GPU timings stay current
        IDXGIDevice1 *dxgiDev = NULL;
        if (SUCCEEDED(ID3D11Device_QueryInterface(gDev, &IID_IDXGIDevice1, (void **)&dxgiDev))) {
            IDXGIDevice1_SetMaximumFrameLatency(dxgiDev, 1);
            REL(dxgiDev);
        }
    }
    IDXGIFactory2_MakeWindowAssociation(f2, gHwnd, DXGI_MWA_NO_ALT_ENTER);
    REL(f2);
    return SUCCEEDED(hr) && MakeBackUav();
}

static void FreeOverlay(void)
{
    REL(gSOverlay);
    REL(gTOverlay);
    if (gDc) { DeleteDC(gDc); gDc = NULL; }
    if (gDib) { DeleteObject(gDib); gDib = NULL; }
    if (gFont) { DeleteObject(gFont); gFont = NULL; }
    gDibBits = NULL;
    gNPrev = gNCur = 0;
}

static int CreateOverlay(int w, int h)
{
    FreeOverlay();
    BITMAPINFO bi;
    memset(&bi, 0, sizeof bi);
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    gDc = CreateCompatibleDC(NULL);
    gDib = CreateDIBSection(gDc, &bi, DIB_RGB_COLORS, (void **)&gDibBits, NULL, 0);
    if (!gDib || !gDibBits) return 0;
    memset(gDibBits, 0, (size_t)w * (size_t)h * 4);
    SelectObject(gDc, gDib);
    int fh = gWinH / 52;
    if (fh < 14) fh = 14;
    if (fh > 30) fh = 30;
    gFont = CreateFontA(-fh, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    SelectObject(gDc, gFont);
    SetBkMode(gDc, TRANSPARENT);
    gDibW = w;
    gDibH = h;
    D3D11_TEXTURE2D_DESC d;
    memset(&d, 0, sizeof d);
    d.Width = (UINT)w; d.Height = (UINT)h; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sr;
    sr.pSysMem = gDibBits;
    sr.SysMemPitch = (UINT)(w * 4);
    sr.SysMemSlicePitch = 0;
    if (FAILED(ID3D11Device_CreateTexture2D(gDev, &d, &sr, &gTOverlay))) return 0;
    if (FAILED(ID3D11Device_CreateShaderResourceView(gDev, (ID3D11Resource *)gTOverlay, NULL, &gSOverlay))) return 0;
    return 1;
}

static void OverlayBegin(void)
{
    for (int i = 0; i < gNPrev; i++) {
        const Rect *r = &gRectPrev[i];
        for (int y = r->y0; y < r->y1; y++) memset(gDibBits + ((size_t)y * (size_t)gDibW + (size_t)r->x0) * 4, 0, (size_t)(r->x1 - r->x0) * 4);
    }
    gNCur = 0;
}

static int OverlayText(int x, int y, const char *s)
{
    int len = (int)strlen(s);
    SIZE sz;
    GetTextExtentPoint32A(gDc, s, len, &sz);
    Rect r = { x - 3, y - 3, x + sz.cx + 4, y + sz.cy + 4 };
    if (r.x0 < 0 || r.y0 < 0 || r.x1 > gDibW || r.y1 > gDibH || gNCur >= 48) return 0;
    for (int i = 0; i < gNCur; i++) {
        const Rect *o = &gRectCur[i];
        if (r.x0 < o->x1 && r.x1 > o->x0 && r.y0 < o->y1 && r.y1 > o->y0) return 0;
    }
    SetTextColor(gDc, RGB(0, 255, 0));
    TextOutA(gDc, x + 1, y + 1, s, len);
    SetTextColor(gDc, RGB(255, 0, 0));
    TextOutA(gDc, x, y, s, len);
    gRectCur[gNCur++] = r;
    return 1;
}

static void OverlayEnd(void)
{
    GdiFlush();
    for (int pass = 0; pass < 2; pass++) {
        const Rect *rs = pass ? gRectCur : gRectPrev;
        int n = pass ? gNCur : gNPrev;
        for (int i = 0; i < n; i++) {
            D3D11_BOX box;
            box.left = (UINT)rs[i].x0; box.right = (UINT)rs[i].x1; box.top = (UINT)rs[i].y0; box.bottom = (UINT)rs[i].y1;
            box.front = 0; box.back = 1;
            const uint8_t *src = gDibBits + ((size_t)rs[i].y0 * (size_t)gDibW + (size_t)rs[i].x0) * 4;
            ID3D11DeviceContext_UpdateSubresource(gCtx, (ID3D11Resource *)gTOverlay, 0, &box, src, (UINT)(gDibW * 4), 0);
        }
    }
    memcpy(gRectPrev, gRectCur, sizeof(Rect) * (size_t)gNCur);
    gNPrev = gNCur;
}

static int RunWindow(int preset, int galaxy)
{
    SetProcessDPIAware();
    HINSTANCE hi = GetModuleHandleA(NULL);
    WNDCLASSEXA wc;
    memset(&wc, 0, sizeof wc);
    wc.cbSize = sizeof wc;
    wc.style = CS_DBLCLKS;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hi;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "blackholeWnd";
    RegisterClassExA(&wc);
    int scr = GetSystemMetrics(SM_CXSCREEN), scrH = GetSystemMetrics(SM_CYSCREEN);
    int cw = scr * 3 / 4, ch = cw * 9 / 16;
    if (ch > scrH * 17 / 20) { ch = scrH * 17 / 20; cw = ch * 16 / 9; }
    RECT r = { 0, 0, cw, ch };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    gHwnd = CreateWindowExA(0, "blackholeWnd", "Black hole - GPU ray tracer", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                            CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, NULL, NULL, hi, NULL);
    if (!gHwnd) return 1;
    RECT cr;
    GetClientRect(gHwnd, &cr);
    gWinW = cr.right;
    gWinH = cr.bottom;
    if (!CreateSwap()) return 1;
    SetMode(M_WIN);
    timeBeginPeriod(1);

    static const float kScale[6] = { 0.50f, 0.65f, 0.80f, 1.0f, 1.0f, 1.0f };
    static const int   kSsaa[6]  = { 1, 1, 1, 1, 2, 2 };
    int level = 3, slow = 0, fast = 0, hudOn = 1, warmFrames = 0;
    gW = gWinW; gH = gWinH;
    if (!CreateTargets(0, 0) || !CreateOverlay(gW, gH)) return 1;

    LARGE_INTEGER freq, prev, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);
    InitView(preset, galaxy);
    float fps = 60.0f, helpT = 0.0f, titleT = 0.0f;

    while (gRun) {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        for (int i = 0; i < gKeyN; i++) {
            int k = gKeyQ[i];
            if (k == K_FULL) ToggleFullscreen();
            else if (k == 'h' || k == 'H') hudOn = !hudOn;
            else if (HandleKey(k)) InterlockedExchange(&gRun, 0);
        }
        gKeyN = 0;
        if (!gRun) break;

        if (gWinDirty && (gWinW != gW || gWinH != gH)) {
            gWinDirty = 0;
            ID3D11DeviceContext_ClearState(gCtx);
            REL(gBackUav);
            REL(gBackTex);
            if (FAILED(IDXGISwapChain1_ResizeBuffers(gSwap, 0, (UINT)gWinW, (UINT)gWinH, DXGI_FORMAT_UNKNOWN, 0)) || !MakeBackUav()) break;
            gW = gWinW; gH = gWinH;
            if (!CreateTargets(0, 0) || !CreateOverlay(gW, gH)) break;
        }

        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - prev.QuadPart) / (double)freq.QuadPart);
        prev = now;
        if (dt > 0.2f) dt = 0.2f;
        if (dt > 1e-4f) fps += (1.0f / dt - fps) * 0.08f;
        helpT += dt;
        titleT += dt;
        if (titleT > 0.25f) {
            titleT = 0.0f;
            char title[160];
            if (gGalaxy) snprintf(title, sizeof title, "Milky Way - GPU ray tracer  [fixed view, gpu %.1f ms, %.0f fps]", gGpuMs, fps);
            else snprintf(title, sizeof title, "Black hole - GPU ray tracer  [d=%.1f rs, el %.0f deg, gpu %.1f ms, %.0f fps]", gDist, gEl * 57.2958f, gGpuMs, fps);
            SetWindowTextA(gHwnd, title);
        }

        // dynamic resolution: keep the GPU time under the target
        warmFrames++;                                    // the first frames are slow while everything warms up: ignore them
        if (warmFrames < 90) slow = fast = 0;
        else if (gGpuMs > kTargetGpuMs * 1.15f) { slow++; fast = 0; }
        else if (gGpuMs > 0.0f && gGpuMs < kTargetGpuMs * 0.62f) { fast++; slow = 0; }
        else { slow = fast = 0; }
        if (slow >= 4 && level > 0) { level--; slow = 0; }
        if (fast >= 90 && level < 5) { level++; fast = 0; }
        gIW = (int)((float)gW * kScale[level] + 0.5f);
        gIH = (int)((float)gH * kScale[level] + 0.5f);
        if (gIW < 64) gIW = 64;
        if (gIH < 36) gIH = 36;
        gExtraOct = level >= 4 ? 2 : (level >= 2 ? 1 : 0);

        StepWorld(dt, 0);

        OverlayBegin();
        if (gLabels) {
            Lab labs[16];
            int n = BuildLabels(labs);
            for (int i = 0; i < n; i++) {
                int x = (int)labs[i].x, y = (int)labs[i].y;
                SIZE sz;
                GetTextExtentPoint32A(gDc, labs[i].text, (int)strlen(labs[i].text), &sz);
                if (labs[i].centered) x -= sz.cx / 2; else x += 12;
                y -= sz.cy / 2;
                OverlayText(x, y, labs[i].text);
            }
        }
        if (hudOn) {
            int fh = gH / 52 > 14 ? gH / 52 : 14;
            char hud[256];
            if (gGalaxy)
                snprintf(hud, sizeof hud, "Milky Way  |  fixed view, nothing to navigate  |  M = back to the black hole  |  %dx%d at %.0f%% , %dx AA  |  GPU %.1f ms  %.0f fps",
                         gW, gH, kScale[level] * 100.0f, kSsaa[level] * kSsaa[level], gGpuMs, fps);
            else
                snprintf(hud, sizeof hud, "d = %.1f rs   incl %.0f deg   disk spin x%.2f%s%s  |  %dx%d at %.0f%% , %dx AA  |  GPU %.1f ms  %.0f fps",
                         gDist, gEl * 57.2958f, gSpin, gPaused ? " (paused)" : "", gAuto ? "   auto-orbit" : "", gW, gH,
                         kScale[level] * 100.0f, kSsaa[level] * kSsaa[level], gGpuMs, fps);
            OverlayText(14, gH - fh - 26, hud);
            if (helpT < 14.0f)
                OverlayText(14, gH - fh - 56, gGalaxy ? "M back to the black hole, L labels, H hide this, F11 fullscreen, Q quit"
                                                     : "drag = orbit, wheel = zoom, 1-5 views, T auto-orbit, [ ] disk spin, M Milky Way, L labels, Space pause, H hide this, F11 fullscreen, Q quit");
        }
        OverlayEnd();

        CB cb;
        FillCB(&cb, kSsaa[level], 1);
        GpuTimingBegin();
        GpuScene(&cb);
        GpuCompose(gBackUav);
        GpuTimingEnd();
        IDXGISwapChain1_Present(gSwap, 1, 0);
    }

    timeEndPeriod(1);
    ReleaseTargets();
    FreeOverlay();
    REL(gBackUav);
    REL(gBackTex);
    REL(gSwap);
    DestroyWindow(gHwnd);
    return 0;
}

// ================================================================= entry
int main(int argc, char **argv)
{
    int testFrames = 0, forceC = 0, forceR = 0, preset = -1, galaxy = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) testFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &forceC, &forceR);
        else if (!strcmp(argv[i], "--view") && i + 1 < argc) preset = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--galaxy")) galaxy = 1;
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) { int f = atoi(argv[++i]); gTickMs = f > 0 ? 1000 / f : 16; }
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            ++i;
            gModePref = !strcmp(argv[i], "half") ? M_HALF : !strcmp(argv[i], "quad") ? M_QUAD : M_SEXT;
        }
    }

    if (!InitGpu()) {
        fputs("GPU init failed (needs a Direct3D 11 GPU with compute shaders, and blackhole.hlsl next to the exe)\n", stderr);
        return 1;
    }
#ifdef WINDOWED
    (void)testFrames; (void)forceC; (void)forceR;
    return RunWindow(preset, galaxy);
#else
    return RunTerminal(preset, galaxy, testFrames, forceC, forceR);
#endif
}
