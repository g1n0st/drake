#ifndef MATH_TOOLS_H
#define MATH_TOOLS_H

#include <string>
#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>

#include "multibody/gpu_mpm/sifakis_svd.cuh"

// A[n,m], B[m,l] ==> C[n,l]
template <int n, int m, int l, class T>
inline __host__ __device__ void matmul(const T* a, const T* b, T* c)
{
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < l; ++k) {
            c[i * l + k] = 0;
            for (int j = 0; j < m; ++j)
                c[i * l + k] += a[i * m + j] * b[j * l + k];
        }
}

// A[n,m], B[l,m] ==> C[n,l]
template <int n, int m, int l, class T>
inline __host__ __device__ void matmulT(const T* a, const T* b, T* c)
{
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < l; ++k) {
            c[i * l + k] = 0;
            for (int j = 0; j < m; ++j)
                c[i * l + k] += a[i * m + j] * b[k * m + j];
        }
}

template <int n, int m, class T>
inline __host__ __device__ void transpose(const T* a, T* aT)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            aT[j * n + i] = a[i * m + j];
}

template <int n, class T>
inline __host__ __device__ T norm_sqr(const T* x)
{
    T result = 0;
    for (int i = 0; i < n; ++i)
        result += x[i] * x[i];
    return result;
}
template <int n, class T>
inline __host__ __device__ T norm(const T* x)
{
    return sqrt(norm_sqr<n>(x));
}

template <int n, class T>
inline __host__ __device__ T distance(const T* x, const T* y)
{
    T xy[n];
    for (int i = 0; i < n; ++i) 
        xy[i] = x[i] - y[i];
    return sqrt(norm_sqr<n>(xy));
}

template <int n, class T>
inline __host__ __device__ void normalize(T* v)
{
    T norm_inv = 1. / norm<n>(v);
    for (int i = 0; i < n; ++i)
        v[i] *= norm_inv;
}

template <int n, class T>
inline __host__ __device__ T dot(const T* x, const T* y)
{
    T result = 0;
    for (int i = 0; i < n; ++i)
        result += x[i] * y[i];
    return result;
}

template <class T>
inline __host__ __device__ void cross_product3(const T* v1, const T* v2, T* result)
{
    result[0] = v1[1] * v2[2] - v2[1] * v1[2];
    result[1] = v1[2] * v2[0] - v2[2] * v1[0];
    result[2] = v1[0] * v2[1] - v2[0] * v1[1];
}

template <int n, class T>
inline __host__ __device__ void outer_product(T* a, T* b, T* ab)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            ab[i * n + j] = a[i] * b[j];
}

template <class T>
inline __host__ __device__ T determinant3(const T* m)
{
    return m[0 * 3 + 0] * (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2])
        - m[1 * 3 + 0] * (m[0 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[0 * 3 + 2])
        + m[2 * 3 + 0] * (m[0 * 3 + 1] * m[1 * 3 + 2] - m[1 * 3 + 1] * m[0 * 3 + 2]);
}

template <class T>
inline __host__ __device__ void inverse3(T* m, T* m_inv)
{
    T det_inv = 1. / determinant3(m);
    m_inv[0 * 3 + 0] = (m[1 * 3 + 1] * m[2 * 3 + 2] - m[1 * 3 + 2] * m[2 * 3 + 1]) * det_inv;
    m_inv[1 * 3 + 0] = (m[1 * 3 + 2] * m[2 * 3 + 0] - m[1 * 3 + 0] * m[2 * 3 + 2]) * det_inv;
    m_inv[2 * 3 + 0] = (m[1 * 3 + 0] * m[2 * 3 + 1] - m[1 * 3 + 1] * m[2 * 3 + 0]) * det_inv;
    m_inv[0 * 3 + 1] = (m[0 * 3 + 2] * m[2 * 3 + 1] - m[0 * 3 + 1] * m[2 * 3 + 2]) * det_inv;
    m_inv[1 * 3 + 1] = (m[0 * 3 + 0] * m[2 * 3 + 2] - m[0 * 3 + 2] * m[2 * 3 + 0]) * det_inv;
    m_inv[2 * 3 + 1] = (m[0 * 3 + 1] * m[2 * 3 + 0] - m[0 * 3 + 0] * m[2 * 3 + 1]) * det_inv;
    m_inv[0 * 3 + 2] = (m[0 * 3 + 1] * m[1 * 3 + 2] - m[0 * 3 + 2] * m[1 * 3 + 1]) * det_inv;
    m_inv[1 * 3 + 2] = (m[0 * 3 + 2] * m[1 * 3 + 0] - m[0 * 3 + 0] * m[1 * 3 + 2]) * det_inv;
    m_inv[2 * 3 + 2] = (m[0 * 3 + 0] * m[1 * 3 + 1] - m[0 * 3 + 1] * m[1 * 3 + 0]) * det_inv;
}

template <class T>
inline __host__ __device__ T determinant2(const T* m)
{
    return m[0 * 2 + 0] * m[1 * 2 + 1] - m[0 * 2 + 1] * m[1 * 2 + 0];
}

template <class T>
inline __host__ __device__ void inverse2(const T* m, T* m_inv)
{
    T det_inv = 1. / determinant2(m);
    m_inv[0 * 2 + 0] =  m[1 * 2 + 1] * det_inv;
    m_inv[0 * 2 + 1] = -m[0 * 2 + 1] * det_inv;
    m_inv[1 * 2 + 0] = -m[1 * 2 + 0] * det_inv;
    m_inv[1 * 2 + 1] =  m[0 * 2 + 0] * det_inv;
}

template <class T>
inline __host__ __device__ void dsytrd3(const T *A, T *Q, T *d, T *e) {
    Q[0*3+0] = 1.0;
    Q[1*3+1] = 1.0;
    Q[2*3+2] = 1.0;
    e[0] = 0.0;
    e[1] = 0.0;
    e[2] = 0.0;
    d[0] = 0.0;
    d[1] = 0.0;
    d[2] = 0.0;
    T u[3] = {0.0, 0.0, 0.0};
    T q[3] = {0.0, 0.0, 0.0};
    T h = A[0*3+1] * A[0*3+1] + A[0*3+2] * A[0*3+2];
    T g = 0.0;
    if (A[0*3+1] > 0) {
        g = -sqrt(h);
    }
    else {
        g = sqrt(h);
    }
    e[0] = g;
    T f = g * A[0*3+1];
    u[1] = A[0*3+1] - g;
    u[2] = A[0*3+2];
    T omega = h - f;
    if (omega > 0.0) {
        omega = 1.0 / omega;
        T K = 0.0;
        f = A[1*3+1] * u[1] + A[1*3+2] * u[2];
        q[1] = omega * f;  // p
        K += u[1] * f;  // u* A u

        f = A[1*3+2] * u[1] + A[2*3+2] * u[2];
        q[2] = omega * f;  // p
        K += u[2] * f;  // u* A u

        K *= 0.5 * omega * omega;

        q[1] = q[1] - K * u[1];
        q[2] = q[2] - K * u[2];

        d[0] = A[0*3+0];
        d[1] = A[1*3+1] - 2.0 * q[1] * u[1];
        d[2] = A[2*3+2] - 2.0 * q[2] * u[2];

        for (int j = 1; j < 3; j++) {
            f = omega * u[j];
            for (int i = 1; i < 3; i++)
                Q[i*3+j] = Q[i*3+j] - f * u[i];
        }

        // Calculate updated A[1, 2] and store it in e[1]
        e[1] = A[1*3+2] - q[1] * u[2] - u[1] * q[2];
    }
    else {
        d[0] = A[0*3+0];
        d[1] = A[1*3+1];
        d[2] = A[2*3+2];
        e[1] = A[1*3+2];
    }
}

template<class T>
inline __host__ __device__ void dsyevq3(const T *A, const T *Q0, const T *w0, T *Q, T *w) {
    for (int i = 0; i < 9; i++) Q[i] = Q0[i];
    for (int i = 0; i < 3; i++) w[i] = w0[i];
    T e[3] = {0.0, 0.0, 0.0};
    dsytrd3(A, Q, w, e);

    for (int l = 0; l < 2; l++) {
        int nIter = 0;
        while (true) {
            // Check for convergence and exit iteration loop if off-diagonal
            // element e(l) is zero
            int m = 0;
            for (int i = l; i < 2; i++) {
                m = i;
                T g = abs(w[m]) + abs(w[m + 1]);
                if (abs(e[m]) + g == g) {
                    break;
                }
            }
            if (m == l) {
                break;
            }

            nIter += 1;
            assert(nIter <= 30);

            // Calculate g = d_m - k
            T g = (w[l + 1] - w[l]) / (e[l] + e[l]);
            T r = sqrt(g * g + 1.0);
            if (g > 0) {
                g = w[m] - w[l] + e[l] / (g + r);
            }
            else {
                g = w[m] - w[l] + e[l] / (g - r);
            }

            T s = 1.0;
            T c = 1.0;
            T p = 0.0;
            int i = m - 1;
            while (i >= l) {
                T f = s * e[i];
                T b = c * e[i];
                if (abs(f) > abs(g)) {
                    c = g / f;
                    r = sqrt(c * c + 1.0);
                    e[i + 1] = f * r;
                    s = 1.0 / r;
                    c *= s;
                }
                else {
                    s = f / g;
                    r = sqrt(s * s + 1.0);
                    e[i + 1] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }

                g = w[i + 1] - p;
                r = (w[i] - g) * s + 2.0 * c * b;
                p = s * r;
                w[i + 1] = g + p;
                g = c * r - b;

                for (int k = 0; k < 3; k++) {
                    T t = Q[k*3+(i + 1)];
                    Q[k*3+(i + 1)] = s * Q[k*3+i] + c * t;
                    Q[k*3+i] = c * Q[k*3+i] - s * t;
                }

                i -= 1;
            }
            w[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

template<class T>
inline __host__ __device__ void sym_eig3x3(const T *A, T *eigenvalues, T *Q) {
    T M_SQRT3 = 1.73205080756887729352744634151;
    T m = A[0*3+0]+A[1*3+1]+A[2*3+2];
    T dd = A[0*3+1] * A[0*3+1];
    T ee = A[1*3+2] * A[1*3+2];
    T ff = A[0*3+2] * A[0*3+2];
    T c1 = A[0*3+0] * A[1*3+1] + A[0*3+0] * A[2*3+2] + A[1*3+1] * A[2*3+2] - (dd + ee + ff);
    T c0 = A[2*3+2] * dd + A[0*3+0] * ee + A[1*3+1] * ff - A[0*3+0] * A[1*3+1] * A[2*3+2] - 2.0 * A[0*3+2] * A[0*3+1] * A[1*3+2];

    T p = m * m - 3.0 * c1;
    T q = m * (p - 1.5 * c1) - 13.5 * c0;
    T sqrt_p = sqrt(abs(p));
    T phi = 27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0));
    phi = (1.0 / 3.0) * atan2(sqrt(abs(phi)), q);

    T c = sqrt_p * cos(phi);
    T s = (1.0 / M_SQRT3) * sqrt_p * sin(phi);
    T eigenvalues_final[3] = {0., 0., 0.};
    eigenvalues[1] = (1.0 / 3.0) * (m - c);
    eigenvalues[2] = eigenvalues[1] + s;
    eigenvalues[0] = eigenvalues[1] + c;
    eigenvalues[1] = eigenvalues[1] - s;

    T t = abs(eigenvalues[0]);
    T u = abs(eigenvalues[1]);
    if (u > t) {
        t = u;
    }
    u = abs(eigenvalues[2]);
    if (u > t) {
        t = u;
    }
    if (t < 1.0) {
        u = t;
    }
    else {
        u = t * t;
    }
    T error = 256.0 * 2.2204460492503131e-16 * u * u;
    for (int i = 0; i < 9; i++) Q[i] = 0;
    T Q_final[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    Q[0*3+1] = A[0*3+1] * A[1*3+2] - A[0*3+2] * A[1*3+1];
    Q[1*3+1] = A[0*3+2] * A[0*3+1] - A[1*3+2] * A[0*3+0];
    Q[2*3+1] = A[0*3+1] * A[0*3+1];

    Q[0*3+0] = Q[0*3+1] + A[0*3+2] * eigenvalues[0];
    Q[1*3+0] = Q[1*3+1] + A[1*3+2] * eigenvalues[0];
    Q[2*3+0] = (A[0*3+0] - eigenvalues[0]) * (A[1*3+1] - eigenvalues[0]) - Q[2*3+1];
    T norm = Q[0*3+0] * Q[0*3+0] + Q[1*3+0] * Q[1*3+0] + Q[2*3+0] * Q[2*3+0];
    int early_ret = 0;
    if (norm <= error) {
        dsyevq3<T>(A, Q, eigenvalues, Q_final, eigenvalues_final);
        early_ret = 1;
    }
    else {
        norm = sqrt(1.0 / norm);
        Q[0*3+0] *= norm;
        Q[1*3+0] *= norm;
        Q[2*3+0] *= norm;
    }
    if (!early_ret) {
        Q[0*3+1] = Q[0*3+1] + A[0*3+2] * eigenvalues[1];
        Q[1*3+1] = Q[1*3+1] + A[1*3+2] * eigenvalues[1];
        Q[2*3+1] = (A[0*3+0] - eigenvalues[1]) * (A[1*3+1] - eigenvalues[1]) - Q[2*3+1];
        norm = Q[0*3+1] * Q[0*3+1] + Q[1*3+1] * Q[1*3+1] + Q[2*3+1] * Q[2*3+1];
        if (norm <= error) {
            dsyevq3<T>(A, Q, eigenvalues, Q_final, eigenvalues_final);
            early_ret = 1;
        }
        else {
            norm = sqrt(1.0 / norm);
            Q[0*3+1] *= norm;
            Q[1*3+1] *= norm;
            Q[2*3+1] *= norm;
        }

        Q[0*3+2] = Q[1*3+0] * Q[2*3+1] - Q[2*3+0] * Q[1*3+1];
        Q[1*3+2] = Q[2*3+0] * Q[0*3+1] - Q[0*3+0] * Q[2*3+1];
        Q[2*3+2] = Q[0*3+0] * Q[1*3+1] - Q[1*3+0] * Q[0*3+1];
    }

    if (early_ret) {
        for (int i = 0; i < 9; i++) Q[i] = Q_final[i];
        for (int i = 0; i < 3; i++) eigenvalues[i] = eigenvalues_final[i];
    }

    if (eigenvalues[1] < eigenvalues[0]) {
        T tmp = eigenvalues[0];
        eigenvalues[0] = eigenvalues[1];
        eigenvalues[1] = tmp;

        T tmp2[3];
        for (int i = 0; i < 3; i++) tmp2[i] = Q[i*3+0];
        for (int i = 0; i < 3; i++) Q[i*3+0] = Q[i*3+1];
        for (int i = 0; i < 3; i++) Q[i*3+1] = tmp2[i];
    }

    if (eigenvalues[2] < eigenvalues[0]) {
        T tmp = eigenvalues[0];
        eigenvalues[0] = eigenvalues[2];
        eigenvalues[2] = tmp;

        T tmp2[3];
        for (int i = 0; i < 3; i++) tmp2[i] = Q[i*3+0];
        for (int i = 0; i < 3; i++) Q[i*3+0] = Q[i*3+2];
        for (int i = 0; i < 3; i++) Q[i*3+2] = tmp2[i];
    }

    if (eigenvalues[2] < eigenvalues[1]) {
        T tmp = eigenvalues[1];
        eigenvalues[1] = eigenvalues[2];
        eigenvalues[2] = tmp;

        T tmp2[3];
        for (int i = 0; i < 3; i++) tmp2[i] = Q[i*3+1];
        for (int i = 0; i < 3; i++) Q[i*3+1] = Q[i*3+2];
        for (int i = 0; i < 3; i++) Q[i*3+2] = tmp2[i];
    }
}

template<class T>
inline __host__ __device__ void svd3x3(const T *A, T *U, T *sigma, T *V) {
    for (int i = 0; i < 9; i++) sigma[i] = 0;
    SifakisSVD::svd(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8],
                    U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
                    V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8],
                    sigma[0*3+0], sigma[1*3+1], sigma[2*3+2]);
}

template<class T>
inline __host__ __device__ void ssvd3x3(const T *A, T *U, T *sigma, T *V) {
    svd3x3(A, U, sigma, V);
    if (determinant3(U) < 0) {
        for (int i = 0; i < 3; i++) U[i*3+2] *= -1.;
        sigma[2*3+2] = -sigma[2*3+2];
    }
    if (determinant3(V) < 0) {
        for (int i = 0; i < 3; i++) V[i*3+2] *= -1.;
        sigma[2*3+2] = -sigma[2*3+2];
    }
}

template<int n, int m, class T>
inline __host__ __device__ void givens_QR(const T *A, T *Q, T *R) {
    #pragma unroll
    for (int i = 0; i < 9; i++) R[i] = A[i]; // R [n, m]
    Q[0] = T(1); Q[1] = T(0); Q[2] = T(0);   // Q [n, n]
    Q[3] = T(0); Q[4] = T(1); Q[5] = T(0);
    Q[6] = T(0); Q[7] = T(0); Q[8] = T(1);
    
    #pragma unroll
    for (int j = 0; j < m; ++j) {
        for (int i = n - 1; i > j; i--) {
            int rowi = i - 1;
            int rowk = i;
            const T &a = R[rowi * m + j];
            const T &b = R[rowk * m + j];
            T d = a * a + b * b;
            T c = T(1);
            T s = T(0);
            T sqrtd = sqrt(d);
            if (sqrtd > 0) {
                T t = 1. / sqrtd;
                c = a * t;
                s = -b * t;
            }

            #pragma unroll
            for (int jj = 0; jj < m; ++jj) {
                T tau1 = R[rowi * m + jj];
                T tau2 = R[rowk * m + jj];
                R[rowi * m + jj] = c * tau1 - s * tau2;
                R[rowk * m + jj] = s * tau1 + c * tau2;
            }

            #pragma unroll
            for (int jj = 0; jj < m; ++jj) {
                T tau1 = Q[rowi * n + jj];
                T tau2 = Q[rowk * n + jj];
                Q[rowi * n + jj] = c * tau1 - s * tau2;
                Q[rowk * n + jj] = s * tau1 + c * tau2;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            T tmp = Q[i * n + j];
            Q[i * n + j] = Q[j * n + i];
            Q[j * n + i] = tmp;
        }
    }
}

template<class T>
inline __host__ __device__ void polar_decompose2x2(const T *A, T *U, T *P) {
    U[0] = T(1); U[1] = T(0);
    U[2] = T(0); U[3] = T(1);
    P[0] = A[0]; P[1] = A[1];
    P[2] = A[2]; P[3] = A[3];

    // if A is a zero matrix we simply return the pair (I, A)
    if (A[0] == 0 && A[1] == 0 && A[2] == 0 && A[3] == 0) {

    } else {
        T detA = determinant2(A);
        T adetA = abs(detA);
        T B[4] = {
            A[0] + A[3], A[1] - A[2],
            A[2] - A[1], A[3] + A[0]
        };

        if (detA < 0) {
            B[0] = A[0] - A[3];
            B[1] = A[1] + A[2];
            B[2] = A[2] + A[1];
            B[3] = A[3] - A[0];
        }
        
        // here det(B) != 0 if A is not the zero matrix
        T adetB = abs(determinant2(B));
        T k = 1. / sqrt(adetB);
        U[0] = B[0] * k;
        U[1] = B[1] * k;
        U[2] = B[2] * k;
        U[3] = B[3] * k;
        P[0] = (A[0] * A[0] + A[2] * A[2] + adetA) * k;
        P[1] = (A[0] * A[1] + A[2] * A[3]) * k;
        P[2] = (A[0] * A[1] + A[2] * A[3]) * k;
        P[3] = (A[1] * A[1] + A[3] * A[3] + adetA) * k;
    }
}

template<class T>
inline __host__ __device__ void svd2x2(const T *A, T *U, T *sigma, T *V) {
    T R[4], S[4];
    polar_decompose2x2(A, R, S);
    T c = 0;
    T s = 0;
    T s1 = 0;
    T s2 = 0;
    if (abs(S[1]) < T(1e-5)) {
        c = 1.;
        s = 0.;
        s1 = S[0];
        s2 = S[3];
    } else {
        T tao = T(.5) * (S[0] - S[3]);
        T w = sqrt(tao * tao + S[1] * S[1]);
        T t = 0;
        if (tao > 0) {
            t = S[1] / (tao + w);
        } else {
            t = S[1] / (tao - w);
        }
        c = T(1.) / sqrt(t * t + T(1.));
        s = -t * c;
        s1 = c * c * S[0] - T(2.) * c * s * S[1] + s * s * S[3];
        s2 = s * s * S[0] + T(2.) * c * s * S[1] + c * c * S[3];
    }
    if (s1 < s2) {
        T tmp = s1;
        s1 = s2;
        s2 = tmp;
        V[0] = -s;
        V[1] = c;
        V[2] = -c;
        V[3] = -s;
    } else {
        V[0] = c;
        V[1] = s;
        V[2] = -s;
        V[3] = -c;
    }
    matmul<2, 2, 2, T>(R, V, U);
    sigma[0] = s1;
    sigma[1] = 0;
    sigma[2] = 0;
    sigma[3] = s2;
}


template<int n, int m, typename T>
__device__ __host__
inline void watch(const char *name, const T *M) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx == 0) {
        printf("%s\n", name);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; j++) {
                printf("%.3f ", M[i*m+j]);
            }
            printf("\n");
        }
    }
}

#endif