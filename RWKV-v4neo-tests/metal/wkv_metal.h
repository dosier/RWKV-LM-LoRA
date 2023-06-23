/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The shader code for the custom operation.
*/

#pragma once

// Defines the Metal soft shrink custom kernel.
static char *CUSTOM_KERNEL = R"MPS_WKV(
#include <metal_stdlib>
using namespace metal;


// Additional forward kernel
constant float MIN_VALUE = -1.0E38;
constant int T_MAX = 1024*1;

template<typename T>
kernel void backward_kernel(constant T* _w [[buffer(0)]],
                            constant T* _u [[buffer(1)]],
                            constant T* _k [[buffer(2)]],
                            constant T* _v [[buffer(3)]],
                            device T* _y [[buffer(4)]],
                            device T* _gy [[buffer(5)]],
                            device T* _gw [[buffer(6)]],
                            device T* _gu [[buffer(7)]],
                            device T* _gk [[buffer(8)]],
                            device T* _gv [[buffer(9)]],
                            constant int& ctxLength [[buffer(10)]],
                            constant int& batchSize [[buffer(11)]],
                            constant int& seqLength [[buffer(12)]],
                            constant int& channelCount [[buffer(13)]],
                            device int* errorBuffer [[buffer(14)]],
                            uint tid [[thread_position_in_grid]]) {

    if(tid >= static_cast<uint>(batchSize * channelCount)) {
        errorBuffer[tid] = 1;
        return;
    }
    if (ctxLength < T_MAX) {
        errorBuffer[tid] = 2;
        return;
    }

    int _b = tid / channelCount;
    int _c = tid % channelCount;
    int _offset = _b * seqLength * channelCount + _c;

    T u = _u[_c];
    T w = _w[_c];
    constant T* k = _k + _offset;
    constant T* v = _v + _offset;
    device T* y = _y + _offset;
    device T* gy = _gy + _offset;
    device T* gk = _gk + _offset;
    device T* gv = _gv + _offset;

    T q[T_MAX];
    T r[T_MAX];

    // similar to the CUDA code, this is a rough attempt
    T gw = 0, gu = 0, aa = 0, bb = 0, ga = 0, gb = 0, pp = MIN_VALUE;
    for(int i = 0; i < seqLength; i++) {
        int ii = i * channelCount;
        T kk = k[ii];
        T vv = v[ii];
        T yy = y[ii];

        T ww = u + kk;
        T p = max(pp, ww);
        T e1 = exp(pp - p);
        T e2 = exp(ww - p);
        T qq = gy[ii] / (e1 * bb + e2);
        gw += (ga - gb * yy) * e1 * qq;
        gu += (vv - yy) * e2 * qq;
        q[i] = qq;
        r[i] = ww - p;

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    const int _offsetBC = _b * channelCount + _c;
    _gw[_offsetBC] = gw * _w[_c]; // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = gu;

    aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = seqLength - 1; i >= 0; i--) {
        const int ii = i * channelCount;
        T kk = k[ii];
        T vv = v[ii];
        T yy = y[ii];
        T qq = q[i];
        T rr = r[i];

        T e1 = qq * exp(rr);
        T e2 = exp(kk + pp);
        gk[ii] = e1 * (vv - yy) + e2 * (aa * vv + bb);
        gv[ii] = e1 + e2 * aa;

        T ww = w + pp;
        T www = rr - u - kk;
        T p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        aa = e1 * aa + e2;
        bb = e1 * bb - e2 * yy;
        pp = p;
    }
}

template
[[host_name("backward_kernel_half")]]
kernel void backward_kernel<half>(constant half* _w [[buffer(0)]],
                                  constant half* _u [[buffer(1)]],
                                  constant half* _k [[buffer(2)]],
                                  constant half* _v [[buffer(3)]],
                                  device half* _y [[buffer(4)]],
                                  device half* _gy [[buffer(5)]],
                                  device half* _gw [[buffer(6)]],
                                  device half* _gu [[buffer(7)]],
                                  device half* _gk [[buffer(8)]],
                                  device half* _gv [[buffer(9)]],
                                  constant int& ctxLength [[buffer(10)]],
                                  constant int& batchSize [[buffer(11)]],
                                  constant int& seqLength [[buffer(12)]],
                                  constant int& channelCount [[buffer(13)]],
                                  device int* errorBuffer [[buffer(14)]],
                                  uint tid [[thread_position_in_grid]]);

template
[[host_name("backward_kernel_float")]]
kernel void backward_kernel<float>(constant float* _w [[buffer(0)]],
                                   constant float* _u [[buffer(1)]],
                                   constant float* _k [[buffer(2)]],
                                   constant float* _v [[buffer(3)]],
                                   device float* _y [[buffer(4)]],
                                   device float* _gy [[buffer(5)]],
                                   device float* _gw [[buffer(6)]],
                                   device float* _gu [[buffer(7)]],
                                   device float* _gk [[buffer(8)]],
                                   device float* _gv [[buffer(9)]],
                                   constant int& ctxLength [[buffer(10)]],
                                   constant int& batchSize [[buffer(11)]],
                                   constant int& seqLength [[buffer(12)]],
                                   constant int& channelCount [[buffer(13)]],
                                   device int* errorBuffer [[buffer(14)]],
                                   uint tid [[thread_position_in_grid]]);

template<typename T>
kernel void forward_kernel(constant T* _w [[buffer(0)]],
                           constant T* _u [[buffer(1)]],
                           constant T* _k [[buffer(2)]],
                           constant T* _v [[buffer(3)]],
                           device T* _y [[buffer(4)]],
                           constant int& batchSize [[buffer(5)]],
                           constant int& seqLength [[buffer(6)]],
                           constant int& channelCount [[buffer(7)]],
                           uint tid [[thread_position_in_grid]]) {
   if(tid >= static_cast<uint>(batchSize * channelCount)) {
        return;
    }

    int _b = tid / channelCount;
    int _c = tid % channelCount;
    int _offset = _b * seqLength * channelCount + _c;

    T u = _u[_c];
    T w = _w[_c];
    constant T* k = _k + _offset;
    constant T* v = _v + _offset;
    device T* y = _y + _offset;

    T aa = 0, bb = 0, pp = MIN_VALUE;
    for(int i = 0; i < seqLength; i++) {
        int ii = i * channelCount;
        T kk = k[ii];
        T vv = v[ii];

        T ww = u + kk;
        T p = max(pp, ww);
        T e1 = exp(pp - p);
        T e2 = exp(ww - p);
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);

        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
}


template
[[host_name("forward_kernel_half")]]
kernel void forward_kernel<half>(constant half* _w [[buffer(0)]],
                                 constant half* _u [[buffer(1)]],
                                 constant half* _k [[buffer(2)]],
                                 constant half* _v [[buffer(3)]],
                                 device half* _y [[buffer(4)]],
                                 constant int& batchSize [[buffer(5)]],
                                 constant int& seqLength [[buffer(6)]],
                                 constant int& channelCount [[buffer(7)]],
                                 uint tid [[thread_position_in_grid]]);

template
[[host_name("forward_kernel_float")]]
kernel void forward_kernel<float>(constant float* _w [[buffer(0)]],
                                  constant float* _u [[buffer(1)]],
                                  constant float* _k [[buffer(2)]],
                                  constant float* _v [[buffer(3)]],
                                  device float* _y [[buffer(4)]],
                                  constant int& batchSize [[buffer(5)]],
                                  constant int& seqLength [[buffer(6)]],
                                  constant int& channelCount [[buffer(7)]],
                                  uint tid [[thread_position_in_grid]]);

)MPS_WKV";