import torch
MIN_VALUE = -1e38

def rwkv_backward_cpu(Tmax, B, T, C, w, u, k, v, y, gy, gw, gu, gk, gv):
    w = w.view(C)
    u = u.view(C)
    k = k.view(B, T, C)
    v = v.view(B, T, C)
    y = y.view(B, T, C)
    gy = gy.view(B, T, C)
    gw = gw.view(B, C)
    gu = gu.view(B, C)
    gk = gk.view(B, T, C)
    gv = gv.view(B, T, C)


    for _b in range(B):
        for _c in range(C):
            u_c = u[_c]
            w_c = w[_c]
            _offset = _b * T * C + _c
            k_offset = k[_b, :, _c]
            v_offset = v[_b, :, _c]
            y_offset = y[_b, :, _c]
            gy_offset = gy[_b, :, _c]
            gk_offset = gk[_b, :, _c]
            gv_offset = gv[_b, :, _c]

            q = torch.zeros(Tmax)
            r = torch.zeros(Tmax)

            gw_c = 0
            gu_c = 0
            aa = 0
            bb = 0
            ga = 0
            gb = 0
            pp = MIN_VALUE
            for i in range(T):
                kk = k_offset[i]
                vv = v_offset[i]
                yy = y_offset[i]

                ww = u_c + kk
                p = max(pp, ww)
                e1 = torch.exp(pp - p)
                e2 = torch.exp(ww - p)
                qq = gy_offset[i] / (e1 * bb + e2)
                gw_c += (ga - gb * yy) * e1 * qq
                gu_c += (vv - yy) * e2 * qq
                q[i] = qq
                r[i] = ww - p

                ww = w_c + pp
                p = max(ww, kk)
                e1 = torch.exp(ww - p)
                e2 = torch.exp(kk - p)
                ga = e1 * (aa + ga)
                gb = e1 * (bb + gb)
                aa = e1 * aa + e2 * vv
                bb = e1 * bb + e2
                pp = p

            gw[_b, _c] = gw_c * w[_c]
            gu[_b, _c] = gu_c

            aa = 0
            bb = 0
            pp = MIN_VALUE
            for i in reversed(range(T)):
                kk = k_offset[i]
                vv = v_offset[i]
                yy = y_offset[i]
                qq = q[i]
                rr = r[i]

                e1 = qq * torch.exp(rr)
                e2 = torch.exp(kk + pp)
                gk_offset[i] = e1 * (vv - yy) + e2 * (aa * vv + bb)
                gv_offset[i] = e1 + e2 * aa

                ww = w_c + pp
                www = rr - u_c - kk
                p = max(ww, www)
                e1 = torch.exp(ww - p)
                e2 = qq * torch.exp(www - p)
                aa = e1 * aa + e2
                bb = e1 * bb - e2 * yy
                pp = p
