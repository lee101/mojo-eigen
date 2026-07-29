"""Dense small-matrix and sparse iterative kernels derived from Eigen."""

from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import abs, atan2, cos, sin, sqrt
from std.sys import has_accelerator
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof


comptime PARALLEL_BATCH = 512
comptime ITEMS_PER_WORKER = 256


def ptr[
    dtype: DType
](address: Int) -> UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]]:
    return UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]](
        unsafe_from_address=address
    )


def iptr(address: Int) -> UnsafePointer[Int64, AnyOrigin[mut=True]]:
    return UnsafePointer[Int64, AnyOrigin[mut=True]](
        unsafe_from_address=address
    )


@always_inline
def scalar[dtype: DType](value: Float64) -> Scalar[dtype]:
    return Scalar[dtype](value)


@always_inline
def epsilon[dtype: DType]() -> Scalar[dtype]:
    if dtype == DType.float64:
        return scalar[dtype](2.220446049250313e-16)
    return scalar[dtype](1.1920928955078125e-7)


@always_inline
def real_min[dtype: DType]() -> Scalar[dtype]:
    if dtype == DType.float64:
        return scalar[dtype](2.2250738585072014e-308)
    return scalar[dtype](1.1754943508222875e-38)


@always_inline
def real_max[dtype: DType]() -> Scalar[dtype]:
    if dtype == DType.float64:
        return scalar[dtype](1.7976931348623157e308)
    return scalar[dtype](3.4028234663852886e38)


@always_inline
def finite[dtype: DType](value: Scalar[dtype]) -> Bool:
    return value == value and abs(value) <= real_max[dtype]()


@always_inline
def simd_copy[
    dtype: DType
](
    source: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    destination: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    count: Int,
):
    comptime W = simdwidthof[dtype]()
    var i = 0
    while i + W <= count:
        destination.store(i, source.load[width=W](i))
        i += W
    while i < count:
        destination[i] = source[i]
        i += 1


# Eigen: Eigen/src/Core/products/GeneralMatrixMatrix.h general_matrix_matrix_product
def matmul[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    b: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    for item in range(batch):
        var base = item * n * n
        for i in range(n):
            for j in range(n):
                var total = scalar[dtype](0.0)
                for k in range(n):
                    total += a[base + i * n + k] * b[base + k * n + j]
                dst[base + i * n + j] = total


# Eigen: Eigen/src/Core/GeneralProduct.h gemv_dense_selector
def matvec[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    x: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    for item in range(batch):
        var matrix_base = item * n * n
        var vector_base = item * n
        for i in range(n):
            var total = scalar[dtype](0.0)
            for j in range(n):
                total += a[matrix_base + i * n + j] * x[vector_base + j]
            dst[vector_base + i] = total


# Eigen: Eigen/src/LU/PartialPivLU.h generic_partial_lu_impl::unblocked_lu
def lu_solve[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solution: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    pivots: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
) -> Bool:
    if n == 0:
        return True
    for i in range(n * n):
        work[i] = matrix[i]
    for i in range(n):
        solution[i] = rhs[i]
    for k in range(n):
        var pivot = k
        var biggest = abs(work[k * n + k])
        for i in range(k + 1, n):
            var candidate = abs(work[i * n + k])
            if candidate > biggest:
                biggest = candidate
                pivot = i
        pivots[k] = Int64(pivot)
        if biggest == scalar[dtype](0.0) or not finite[dtype](biggest):
            return False
        if pivot != k:
            for j in range(n):
                var temporary = work[k * n + j]
                work[k * n + j] = work[pivot * n + j]
                work[pivot * n + j] = temporary
            var rhs_temporary = solution[k]
            solution[k] = solution[pivot]
            solution[pivot] = rhs_temporary
        for i in range(k + 1, n):
            work[i * n + k] /= work[k * n + k]
            var multiplier = work[i * n + k]
            for j in range(k + 1, n):
                work[i * n + j] -= multiplier * work[k * n + j]
    for i in range(n):
        var value = solution[i]
        for j in range(i):
            value -= work[i * n + j] * solution[j]
        solution[i] = value
    for rr in range(n):
        var i = n - 1 - rr
        var value = solution[i]
        for j in range(i + 1, n):
            value -= work[i * n + j] * solution[j]
        solution[i] = value / work[i * n + i]
    return True


# Eigen: Eigen/src/Cholesky/LLT.h llt_inplace<Scalar, Lower>::unblocked
def llt_solve[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solution: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
) -> Bool:
    if n == 0:
        return True
    for i in range(n * n):
        work[i] = matrix[i]
    for k in range(n):
        var diagonal = work[k * n + k]
        for j in range(k):
            diagonal -= work[k * n + j] * work[k * n + j]
        if diagonal <= scalar[dtype](0.0) or not finite[dtype](diagonal):
            return False
        diagonal = sqrt(diagonal)
        work[k * n + k] = diagonal
        for i in range(k + 1, n):
            var value = work[i * n + k]
            for j in range(k):
                value -= work[i * n + j] * work[k * n + j]
            work[i * n + k] = value / diagonal
    for i in range(n):
        var value = rhs[i]
        for j in range(i):
            value -= work[i * n + j] * solution[j]
        solution[i] = value / work[i * n + i]
    for rr in range(n):
        var i = n - 1 - rr
        var value = solution[i]
        for j in range(i + 1, n):
            value -= work[j * n + i] * solution[j]
        solution[i] = value / work[i * n + i]
    return True


# Eigen: Eigen/src/Cholesky/LDLT.h ldlt_inplace<Lower>::unblocked
def ldlt_solve[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solution: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    temporary: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    pivots: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
) -> Bool:
    if n == 0:
        return True
    simd_copy[dtype](matrix, work, n * n)
    var found_zero = False
    for k in range(n):
        var pivot = k
        var biggest = abs(work[k * n + k])
        for i in range(k + 1, n):
            var candidate = abs(work[i * n + i])
            if candidate > biggest:
                biggest = candidate
                pivot = i
        pivots[k] = Int64(pivot)
        if pivot != k:
            for j in range(n):
                var value = work[k * n + j]
                work[k * n + j] = work[pivot * n + j]
                work[pivot * n + j] = value
            for i in range(n):
                var value = work[i * n + k]
                work[i * n + k] = work[i * n + pivot]
                work[i * n + pivot] = value
        for j in range(k):
            temporary[j] = work[j * n + j] * work[k * n + j]
        var diagonal = work[k * n + k]
        for j in range(k):
            diagonal -= work[k * n + j] * temporary[j]
        work[k * n + k] = diagonal
        var valid = abs(diagonal) > scalar[dtype](0.0)
        if found_zero and valid:
            return False
        if not valid:
            found_zero = True
        for i in range(k + 1, n):
            var value = work[i * n + k]
            for j in range(k):
                value -= work[i * n + j] * temporary[j]
            if valid:
                work[i * n + k] = value / diagonal
            elif value != scalar[dtype](0.0):
                return False
            else:
                work[i * n + k] = scalar[dtype](0.0)
    for i in range(n):
        solution[i] = rhs[i]
    for k in range(n):
        var pivot = Int(pivots[k])
        if pivot != k:
            var value = solution[k]
            solution[k] = solution[pivot]
            solution[pivot] = value
    for i in range(n):
        var value = solution[i]
        for j in range(i):
            value -= work[i * n + j] * solution[j]
        solution[i] = value
    for i in range(n):
        var diagonal = work[i * n + i]
        if diagonal == scalar[dtype](0.0):
            return False
        solution[i] /= diagonal
    for rr in range(n):
        var i = n - 1 - rr
        var value = solution[i]
        for j in range(i + 1, n):
            value -= work[j * n + i] * solution[j]
        solution[i] = value
    for rr in range(n):
        var k = n - 1 - rr
        var pivot = Int(pivots[k])
        if pivot != k:
            var value = solution[k]
            solution[k] = solution[pivot]
            solution[pivot] = value
    return True


# Eigen: Eigen/src/Householder/Householder.h MatrixBase::makeHouseholder
# Eigen: Eigen/src/QR/HouseholderQR.h householder_qr_inplace_unblocked
def qr_solve[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solution: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    transformed_rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    m: Int,
    n: Int,
) -> Bool:
    if n == 0:
        return True
    if m < n:
        return False
    for i in range(m * n):
        work[i] = matrix[i]
    for i in range(m):
        transformed_rhs[i] = rhs[i]
    for k in range(n):
        var tail_squared = scalar[dtype](0.0)
        for i in range(k + 1, m):
            tail_squared += work[i * n + k] * work[i * n + k]
        var c0 = work[k * n + k]
        var tau = scalar[dtype](0.0)
        var beta = c0
        if tail_squared > real_min[dtype]():
            beta = sqrt(c0 * c0 + tail_squared)
            if c0 >= scalar[dtype](0.0):
                beta = -beta
            var denominator = c0 - beta
            for i in range(k + 1, m):
                work[i * n + k] /= denominator
            tau = (beta - c0) / beta
        else:
            for i in range(k + 1, m):
                work[i * n + k] = scalar[dtype](0.0)
        work[k * n + k] = beta
        if tau != scalar[dtype](0.0):
            for j in range(k + 1, n):
                var dot = work[k * n + j]
                for i in range(k + 1, m):
                    dot += work[i * n + k] * work[i * n + j]
                dot *= tau
                work[k * n + j] -= dot
                for i in range(k + 1, m):
                    work[i * n + j] -= work[i * n + k] * dot
            var rhs_dot = transformed_rhs[k]
            for i in range(k + 1, m):
                rhs_dot += work[i * n + k] * transformed_rhs[i]
            rhs_dot *= tau
            transformed_rhs[k] -= rhs_dot
            for i in range(k + 1, m):
                transformed_rhs[i] -= work[i * n + k] * rhs_dot
    for rr in range(n):
        var i = n - 1 - rr
        var diagonal = work[i * n + i]
        if diagonal == scalar[dtype](0.0) or not finite[dtype](diagonal):
            return False
        var value = transformed_rhs[i]
        for j in range(i + 1, n):
            value -= work[i * n + j] * solution[j]
        solution[i] = value / diagonal
    return True


@always_inline
def apply_left[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    p: Int,
    q: Int,
    c: Scalar[dtype],
    s: Scalar[dtype],
):
    comptime W = simdwidthof[dtype]()
    var j = 0
    while j + W <= n:
        var x = matrix.load[width=W](p * n + j)
        var y = matrix.load[width=W](q * n + j)
        matrix.store(p * n + j, c * x + s * y)
        matrix.store(q * n + j, -s * x + c * y)
        j += W
    while j < n:
        var x = matrix[p * n + j]
        var y = matrix[q * n + j]
        matrix[p * n + j] = c * x + s * y
        matrix[q * n + j] = -s * x + c * y
        j += 1


@always_inline
def apply_right[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    p: Int,
    q: Int,
    c: Scalar[dtype],
    s: Scalar[dtype],
):
    for i in range(n):
        var x = matrix[i * n + p]
        var y = matrix[i * n + q]
        matrix[i * n + p] = c * x - s * y
        matrix[i * n + q] = s * x + c * y


# Eigen: Eigen/src/Jacobi/Jacobi.h real_2x2_jacobi_svd
def svd_rotation[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    p: Int,
    q: Int,
    rotations: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
):
    var m00 = matrix[p * n + p]
    var m01 = matrix[p * n + q]
    var m10 = matrix[q * n + p]
    var m11 = matrix[q * n + q]
    var t = m00 + m11
    var d = m10 - m01
    var c1 = scalar[dtype](1.0)
    var s1 = scalar[dtype](0.0)
    if abs(d) >= real_min[dtype]():
        var u = t / d
        s1 = scalar[dtype](1.0) / sqrt(scalar[dtype](1.0) + u * u)
        c1 = u * s1
    var a00 = c1 * m00 + s1 * m10
    var a01 = c1 * m01 + s1 * m11
    var a11 = -s1 * m01 + c1 * m11
    var deno = scalar[dtype](2.0) * abs(a01)
    var jr_c = scalar[dtype](1.0)
    var jr_s = scalar[dtype](0.0)
    if deno >= real_min[dtype]():
        var tau = (a00 - a11) / deno
        var w = sqrt(tau * tau + scalar[dtype](1.0))
        var tangent: Scalar[dtype]
        if tau > scalar[dtype](0.0):
            tangent = scalar[dtype](1.0) / (tau + w)
        else:
            tangent = scalar[dtype](1.0) / (tau - w)
        var sign_t = (
            scalar[dtype](1.0)
            if tangent > scalar[dtype](0.0)
            else scalar[dtype](-1.0)
        )
        var normalizer = scalar[dtype](1.0) / sqrt(
            tangent * tangent + scalar[dtype](1.0)
        )
        jr_s = -sign_t * abs(tangent) * normalizer
        jr_c = normalizer
    rotations[0] = c1 * jr_c + s1 * jr_s
    rotations[1] = s1 * jr_c - c1 * jr_s
    rotations[2] = jr_c
    rotations[3] = jr_s


# Eigen: Eigen/src/SVD/JacobiSVD.h JacobiSVD::compute_impl
def jacobi_svd[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    singular_values: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    matrix_u: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    matrix_v: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rotations: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
) -> Bool:
    if n != 3 and n != 4:
        return False
    var scale = scalar[dtype](0.0)
    for i in range(n * n):
        scale = max(scale, abs(matrix[i]))
    if not finite[dtype](scale):
        return False
    if scale == scalar[dtype](0.0):
        scale = scalar[dtype](1.0)
    comptime W = simdwidthof[dtype]()
    var i = 0
    while i + W <= n * n:
        work.store(i, matrix.load[width=W](i) / scale)
        matrix_u.store(i, SIMD[dtype, W](0.0))
        matrix_v.store(i, SIMD[dtype, W](0.0))
        i += W
    while i < n * n:
        work[i] = matrix[i] / scale
        matrix_u[i] = scalar[dtype](0.0)
        matrix_v[i] = scalar[dtype](0.0)
        i += 1
    for i in range(n):
        matrix_u[i * n + i] = scalar[dtype](1.0)
        matrix_v[i * n + i] = scalar[dtype](1.0)
    var max_diagonal = scalar[dtype](0.0)
    for i in range(n):
        max_diagonal = max(max_diagonal, abs(work[i * n + i]))
    var precision = scalar[dtype](2.0) * epsilon[dtype]()
    var sweep = 0
    while sweep < 128:
        var finished = True
        var threshold = max(real_min[dtype](), precision * max_diagonal)
        for p in range(1, n):
            for q in range(p):
                if (
                    abs(work[p * n + q]) > threshold
                    or abs(work[q * n + p]) > threshold
                ):
                    finished = False
                    svd_rotation[dtype](work, n, p, q, rotations)
                    var left_c = rotations[0]
                    var left_s = rotations[1]
                    var right_c = rotations[2]
                    var right_s = rotations[3]
                    apply_left[dtype](work, n, p, q, left_c, left_s)
                    apply_right[dtype](
                        matrix_u, n, p, q, left_c, -left_s
                    )
                    apply_right[dtype](
                        work, n, p, q, right_c, right_s
                    )
                    apply_right[dtype](
                        matrix_v, n, p, q, right_c, right_s
                    )
                    max_diagonal = max(
                        max_diagonal,
                        max(
                            abs(work[p * n + p]),
                            abs(work[q * n + q]),
                        ),
                    )
                    threshold = max(
                        real_min[dtype](), precision * max_diagonal
                    )
        if finished:
            break
        sweep += 1
    if sweep == 128:
        return False
    for i in range(n):
        var diagonal = work[i * n + i]
        singular_values[i] = abs(diagonal) * scale
        if diagonal < scalar[dtype](0.0):
            for row in range(n):
                matrix_u[row * n + i] = -matrix_u[row * n + i]
    for i in range(n):
        var position = i
        var biggest = singular_values[i]
        for j in range(i + 1, n):
            if singular_values[j] > biggest:
                biggest = singular_values[j]
                position = j
        if position != i:
            singular_values[position] = singular_values[i]
            singular_values[i] = biggest
            for row in range(n):
                var value = matrix_u[row * n + i]
                matrix_u[row * n + i] = matrix_u[row * n + position]
                matrix_u[row * n + position] = value
                value = matrix_v[row * n + i]
                matrix_v[row * n + i] = matrix_v[row * n + position]
                matrix_v[row * n + position] = value
    return True


# Eigen: Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h direct_selfadjoint_eigenvalues<3>::extract_kernel
@always_inline
def cross_column[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvalue: Scalar[dtype],
    column_a: Int,
    column_b: Int,
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    dst_column: Int,
):
    var ax = matrix[column_a]
    var ay = matrix[3 + column_a]
    var az = matrix[6 + column_a]
    var bx = matrix[column_b]
    var by = matrix[3 + column_b]
    var bz = matrix[6 + column_b]
    if column_a == 0:
        ax -= eigenvalue
    elif column_a == 1:
        ay -= eigenvalue
    else:
        az -= eigenvalue
    if column_b == 0:
        bx -= eigenvalue
    elif column_b == 1:
        by -= eigenvalue
    else:
        bz -= eigenvalue
    dst[dst_column] = ay * bz - az * by
    dst[3 + dst_column] = az * bx - ax * bz
    dst[6 + dst_column] = ax * by - ay * bx


# Eigen: Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h direct_selfadjoint_eigenvalues<3>::run
def direct_eigh3[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvalues: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvectors: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
) -> Bool:
    var candidate_vector = work + 9
    var shift = (
        matrix[0] + matrix[4] + matrix[8]
    ) / scalar[dtype](3.0)
    var scale = scalar[dtype](0.0)
    for i in range(3):
        for j in range(3):
            var value = matrix[max(i, j) * 3 + min(i, j)]
            if i == j:
                value -= shift
            work[i * 3 + j] = value
            scale = max(scale, abs(value))
    if not finite[dtype](scale):
        return False
    if scale > scalar[dtype](0.0):
        for i in range(9):
            work[i] /= scale
    var c0 = (
        work[0] * work[4] * work[8]
        + scalar[dtype](2.0) * work[3] * work[6] * work[7]
        - work[0] * work[7] * work[7]
        - work[4] * work[6] * work[6]
        - work[8] * work[3] * work[3]
    )
    var c1 = (
        work[0] * work[4] - work[3] * work[3]
        + work[0] * work[8] - work[6] * work[6]
        + work[4] * work[8] - work[7] * work[7]
    )
    var c2 = work[0] + work[4] + work[8]
    var third = scalar[dtype](1.0) / scalar[dtype](3.0)
    var c2_third = c2 * third
    var a_third = max(
        (c2 * c2_third - c1) * third, scalar[dtype](0.0)
    )
    var half_b = scalar[dtype](0.5) * (
        c0
        + c2_third
        * (
            scalar[dtype](2.0) * c2_third * c2_third
            - c1
        )
    )
    var q = max(
        a_third * a_third * a_third - half_b * half_b,
        scalar[dtype](0.0),
    )
    var rho = sqrt(a_third)
    var theta: Scalar[dtype]
    if dtype == DType.float64:
        theta = scalar[dtype](
            atan2(Float64(sqrt(q)), Float64(half_b))
        ) * third
    else:
        theta = scalar[dtype](
            Float64(atan2(Float32(sqrt(q)), Float32(half_b)))
        ) * third
    var cos_theta: Scalar[dtype]
    var sin_theta: Scalar[dtype]
    if dtype == DType.float64:
        cos_theta = scalar[dtype](cos(Float64(theta)))
        sin_theta = scalar[dtype](sin(Float64(theta)))
    else:
        cos_theta = scalar[dtype](Float64(cos(Float32(theta))))
        sin_theta = scalar[dtype](Float64(sin(Float32(theta))))
    var sqrt_three = sqrt(scalar[dtype](3.0))
    eigenvalues[0] = c2_third - rho * (
        cos_theta + sqrt_three * sin_theta
    )
    eigenvalues[1] = c2_third - rho * (
        cos_theta - sqrt_three * sin_theta
    )
    eigenvalues[2] = c2_third + scalar[dtype](2.0) * rho * cos_theta
    for i in range(2):
        for j in range(i + 1, 3):
            if eigenvalues[i] > eigenvalues[j]:
                var value = eigenvalues[i]
                eigenvalues[i] = eigenvalues[j]
                eigenvalues[j] = value
    for i in range(9):
        eigenvectors[i] = scalar[dtype](0.0)
    if eigenvalues[2] - eigenvalues[0] <= epsilon[dtype]():
        eigenvectors[0] = scalar[dtype](1.0)
        eigenvectors[4] = scalar[dtype](1.0)
        eigenvectors[8] = scalar[dtype](1.0)
    else:
        var d0 = eigenvalues[2] - eigenvalues[1]
        var d1 = eigenvalues[1] - eigenvalues[0]
        var k = 0
        var l = 2
        if d0 > d1:
            k = 2
            l = 0
            d0 = d1
        var i0 = 0
        var biggest = abs(work[0] - eigenvalues[k])
        for i in range(1, 3):
            var candidate = abs(work[i * 3 + i] - eigenvalues[k])
            if candidate > biggest:
                biggest = candidate
                i0 = i
        for row in range(3):
            eigenvectors[row * 3 + l] = work[row * 3 + i0]
        eigenvectors[i0 * 3 + l] -= eigenvalues[k]
        cross_column[dtype](
            work, eigenvalues[k], i0, (i0 + 1) % 3, eigenvectors, k
        )
        var norm0 = (
            eigenvectors[k] * eigenvectors[k]
            + eigenvectors[3 + k] * eigenvectors[3 + k]
            + eigenvectors[6 + k] * eigenvectors[6 + k]
        )
        cross_column[dtype](
            work,
            eigenvalues[k],
            i0,
            (i0 + 2) % 3,
            candidate_vector,
            0,
        )
        var norm1 = (
            candidate_vector[0] * candidate_vector[0]
            + candidate_vector[3] * candidate_vector[3]
            + candidate_vector[6] * candidate_vector[6]
        )
        if norm1 > norm0:
            for row in range(3):
                eigenvectors[row * 3 + k] = candidate_vector[row * 3]
            norm0 = norm1
        var inverse_norm = scalar[dtype](1.0) / sqrt(norm0)
        for row in range(3):
            eigenvectors[row * 3 + k] *= inverse_norm
        if d0 <= scalar[dtype](2.0) * epsilon[dtype]() * d1:
            var dot = scalar[dtype](0.0)
            for row in range(3):
                dot += (
                    eigenvectors[row * 3 + k]
                    * eigenvectors[row * 3 + l]
                )
            var norm = scalar[dtype](0.0)
            for row in range(3):
                eigenvectors[row * 3 + l] -= (
                    dot * eigenvectors[row * 3 + k]
                )
                norm += (
                    eigenvectors[row * 3 + l]
                    * eigenvectors[row * 3 + l]
                )
            norm = scalar[dtype](1.0) / sqrt(norm)
            for row in range(3):
                eigenvectors[row * 3 + l] *= norm
        else:
            i0 = 0
            biggest = abs(work[0] - eigenvalues[l])
            for i in range(1, 3):
                var candidate = abs(work[i * 3 + i] - eigenvalues[l])
                if candidate > biggest:
                    biggest = candidate
                    i0 = i
            cross_column[dtype](
                work,
                eigenvalues[l],
                i0,
                (i0 + 1) % 3,
                eigenvectors,
                l,
            )
            var norm = (
                eigenvectors[l] * eigenvectors[l]
                + eigenvectors[3 + l] * eigenvectors[3 + l]
                + eigenvectors[6 + l] * eigenvectors[6 + l]
            )
            cross_column[dtype](
                work,
                eigenvalues[l],
                i0,
                (i0 + 2) % 3,
                candidate_vector,
                0,
            )
            var other_norm = (
                candidate_vector[0] * candidate_vector[0]
                + candidate_vector[3] * candidate_vector[3]
                + candidate_vector[6] * candidate_vector[6]
            )
            if other_norm > norm:
                for row in range(3):
                    eigenvectors[row * 3 + l] = candidate_vector[row * 3]
                norm = other_norm
            norm = scalar[dtype](1.0) / sqrt(norm)
            for row in range(3):
                eigenvectors[row * 3 + l] *= norm
        eigenvectors[1] = (
            eigenvectors[5] * eigenvectors[6]
            - eigenvectors[8] * eigenvectors[3]
        )
        eigenvectors[4] = (
            eigenvectors[8] * eigenvectors[0]
            - eigenvectors[2] * eigenvectors[6]
        )
        eigenvectors[7] = (
            eigenvectors[2] * eigenvectors[3]
            - eigenvectors[5] * eigenvectors[0]
        )
        var norm = sqrt(
            eigenvectors[1] * eigenvectors[1]
            + eigenvectors[4] * eigenvectors[4]
            + eigenvectors[7] * eigenvectors[7]
        )
        for row in range(3):
            eigenvectors[row * 3 + 1] /= norm
    for i in range(3):
        eigenvalues[i] = eigenvalues[i] * scale + shift
    return True


# Eigen: Eigen/src/Eigenvalues/Tridiagonalization.h tridiagonalization_inplace_unblocked
# Eigen: Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h computeFromTridiagonal_impl
def tridiagonal_eigh4[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvalues: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvectors: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
) -> Bool:
    comptime n = 4
    var vectors = work + 16
    var helper = work + 20
    var subdiag = work + 24
    var scale = scalar[dtype](0.0)
    for i in range(n):
        for j in range(n):
            var value = matrix[max(i, j) * n + min(i, j)]
            work[i * n + j] = value
            scale = max(scale, abs(value))
            eigenvectors[i * n + j] = (
                scalar[dtype](1.0)
                if i == j
                else scalar[dtype](0.0)
            )
    if not finite[dtype](scale):
        return False
    if scale == scalar[dtype](0.0):
        scale = scalar[dtype](1.0)
    for i in range(16):
        work[i] /= scale
    for column in range(n - 1):
        var remaining = n - column - 1
        var tail_squared = scalar[dtype](0.0)
        for j in range(1, remaining):
            var value = work[(column + 1 + j) * n + column]
            tail_squared += value * value
        var c0 = work[(column + 1) * n + column]
        var tau = scalar[dtype](0.0)
        var beta = c0
        if tail_squared > real_min[dtype]():
            beta = sqrt(c0 * c0 + tail_squared)
            if c0 >= scalar[dtype](0.0):
                beta = -beta
            var denominator = c0 - beta
            vectors[0] = scalar[dtype](1.0)
            for j in range(1, remaining):
                vectors[j] = (
                    work[(column + 1 + j) * n + column] / denominator
                )
            tau = (beta - c0) / beta
        else:
            vectors[0] = scalar[dtype](1.0)
            for j in range(1, remaining):
                vectors[j] = scalar[dtype](0.0)
        for i in range(remaining):
            var value = scalar[dtype](0.0)
            for j in range(remaining):
                value += (
                    work[(column + 1 + i) * n + column + 1 + j]
                    * vectors[j]
                )
            helper[i] = tau * value
        var dot = scalar[dtype](0.0)
        for i in range(remaining):
            dot += helper[i] * vectors[i]
        var alpha = -scalar[dtype](0.5) * tau * dot
        for i in range(remaining):
            helper[i] += alpha * vectors[i]
        for i in range(remaining):
            for j in range(i + 1):
                var value = (
                    vectors[i] * helper[j] + helper[i] * vectors[j]
                )
                work[(column + 1 + i) * n + column + 1 + j] -= value
                work[(column + 1 + j) * n + column + 1 + i] = (
                    work[(column + 1 + i) * n + column + 1 + j]
                )
        if tau != scalar[dtype](0.0):
            for row in range(n):
                var value = scalar[dtype](0.0)
                for j in range(remaining):
                    value += (
                        eigenvectors[row * n + column + 1 + j]
                        * vectors[j]
                    )
                value *= tau
                for j in range(remaining):
                    eigenvectors[row * n + column + 1 + j] -= (
                        value * vectors[j]
                    )
        work[(column + 1) * n + column] = beta
        work[column * n + column + 1] = beta
    for i in range(n):
        eigenvalues[i] = work[i * n + i]
    for i in range(n - 1):
        subdiag[i] = work[(i + 1) * n + i]
    var end = n - 1
    var start = 0
    var iterations = 0
    var precision_inverse = scalar[dtype](1.0) / epsilon[dtype]()
    while end > 0:
        for i in range(start, end):
            if abs(subdiag[i]) < real_min[dtype]():
                subdiag[i] = scalar[dtype](0.0)
            else:
                var scaled = precision_inverse * subdiag[i]
                if scaled * scaled <= abs(eigenvalues[i]) + abs(
                    eigenvalues[i + 1]
                ):
                    subdiag[i] = scalar[dtype](0.0)
        while end > 0 and subdiag[end - 1] == scalar[dtype](0.0):
            end -= 1
        if end <= 0:
            break
        iterations += 1
        if iterations > 30 * n:
            return False
        start = end - 1
        while start > 0 and subdiag[start - 1] != scalar[dtype](0.0):
            start -= 1
        var td = (
            eigenvalues[end - 1] - eigenvalues[end]
        ) * scalar[dtype](0.5)
        var edge = subdiag[end - 1]
        var mu = eigenvalues[end]
        if td == scalar[dtype](0.0):
            mu -= abs(edge)
        elif edge != scalar[dtype](0.0):
            var h = sqrt(td * td + edge * edge)
            mu -= edge * edge / (
                td + (h if td > scalar[dtype](0.0) else -h)
            )
        var x = eigenvalues[start] - mu
        var z = subdiag[start]
        for k in range(start, end):
            if z == scalar[dtype](0.0):
                break
            var c: Scalar[dtype]
            var s: Scalar[dtype]
            if x == scalar[dtype](0.0):
                c = scalar[dtype](0.0)
                s = (
                    scalar[dtype](1.0)
                    if z < scalar[dtype](0.0)
                    else scalar[dtype](-1.0)
                )
            elif abs(x) > abs(z):
                var tangent = z / x
                var u = sqrt(scalar[dtype](1.0) + tangent * tangent)
                if x < scalar[dtype](0.0):
                    u = -u
                c = scalar[dtype](1.0) / u
                s = -tangent * c
            else:
                var tangent = x / z
                var u = sqrt(scalar[dtype](1.0) + tangent * tangent)
                if z < scalar[dtype](0.0):
                    u = -u
                s = -scalar[dtype](1.0) / u
                c = -tangent * s
            var difference = eigenvalues[k] - eigenvalues[k + 1]
            var delta = s * (
                s * difference
                + scalar[dtype](2.0) * c * subdiag[k]
            )
            subdiag[k] = (
                c * s * difference
                + (c * c - s * s) * subdiag[k]
            )
            eigenvalues[k] -= delta
            eigenvalues[k + 1] += delta
            if k > start:
                subdiag[k - 1] = c * subdiag[k - 1] - s * z
            x = subdiag[k]
            if k < end - 1:
                z = -s * subdiag[k + 1]
                subdiag[k + 1] = c * subdiag[k + 1]
            apply_right[dtype](eigenvectors, n, k, k + 1, c, s)
    for i in range(n):
        eigenvalues[i] *= scale
    for i in range(n - 1):
        var position = i
        var smallest = eigenvalues[i]
        for j in range(i + 1, n):
            if eigenvalues[j] < smallest:
                smallest = eigenvalues[j]
                position = j
        if position != i:
            eigenvalues[position] = eigenvalues[i]
            eigenvalues[i] = smallest
            for row in range(n):
                var value = eigenvectors[row * n + i]
                eigenvectors[row * n + i] = eigenvectors[row * n + position]
                eigenvectors[row * n + position] = value
    return True


def selfadjoint_eigh[
    dtype: DType
](
    matrix: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvalues: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvectors: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
) -> Bool:
    if n == 3:
        return direct_eigh3[dtype](matrix, eigenvalues, eigenvectors, work)
    if n == 4:
        return tridiagonal_eigh4[dtype](
            matrix, eigenvalues, eigenvectors, work
        )
    return False


def batch_ldlt_solve[
    dtype: DType
](
    matrices: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    right_hand_sides: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solutions: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    temporary: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    pivots: UnsafePointer[Int64, AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    var workers = 1
    if batch >= PARALLEL_BATCH:
        workers = min(
            num_physical_cores(),
            (batch + ITEMS_PER_WORKER - 1) // ITEMS_PER_WORKER,
        )

    @parameter
    def process(worker: Int):
        var start = worker * batch // workers
        var end = (worker + 1) * batch // workers
        for item in range(start, end):
            statuses[item] = Int64(
                1
                if ldlt_solve[dtype](
                    matrices + item * n * n,
                    right_hand_sides + item * n,
                    solutions + item * n,
                    work + item * n * n,
                    temporary + item * n,
                    pivots + item * n,
                    n,
                )
                else 0
            )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def batch_selfadjoint_eigh[
    dtype: DType
](
    matrices: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvalues: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    eigenvectors: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    var workspace_size = 28 if n == 4 else 18
    var workers = 1
    if batch >= PARALLEL_BATCH:
        workers = min(
            num_physical_cores(),
            (batch + ITEMS_PER_WORKER - 1) // ITEMS_PER_WORKER,
        )

    @parameter
    def process(worker: Int):
        var start = worker * batch // workers
        var end = (worker + 1) * batch // workers
        for item in range(start, end):
            statuses[item] = Int64(
                1
                if selfadjoint_eigh[dtype](
                    matrices + item * n * n,
                    eigenvalues + item * n,
                    eigenvectors + item * n * n,
                    work + item * workspace_size,
                    n,
                )
                else 0
            )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def gpu_eigh4_f64(
    matrices: UnsafePointer[Float64, AnyOrigin[mut=True]],
    eigenvalues: UnsafePointer[Float64, AnyOrigin[mut=True]],
    eigenvectors: UnsafePointer[Float64, AnyOrigin[mut=True]],
    work: UnsafePointer[Float64, AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    var item = Int(global_idx.x)
    if item >= batch:
        return
    statuses[item] = Int64(
        1
        if tridiagonal_eigh4[DType.float64](
            matrices + item * n * n,
            eigenvalues + item * n,
            eigenvectors + item * n * n,
            work + item * 28,
        )
        else 0
    )


def batch_selfadjoint_eigh_gpu_f64(
    matrices: UnsafePointer[Float64, AnyOrigin[mut=True]],
    eigenvalues: UnsafePointer[Float64, AnyOrigin[mut=True]],
    eigenvectors: UnsafePointer[Float64, AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
) -> Bool:
    comptime if not has_accelerator():
        return False
    else:
        if n != 4:
            return False
        try:
            var ctx = DeviceContext()
            var matrices_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n * n
            )
            var values_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n
            )
            var vectors_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n * n
            )
            var work_device = ctx.enqueue_create_buffer[DType.float64](
                batch * 28
            )
            var statuses_device = ctx.enqueue_create_buffer[DType.int64](batch)
            ctx.enqueue_copy(dst_buf=matrices_device, src_ptr=matrices)
            var block_size = 256
            ctx.enqueue_function[gpu_eigh4_f64](
                matrices_device,
                values_device,
                vectors_device,
                work_device,
                statuses_device,
                n,
                batch,
                grid_dim=(batch + block_size - 1) // block_size,
                block_dim=block_size,
            )
            ctx.enqueue_copy(dst_ptr=eigenvalues, src_buf=values_device)
            ctx.enqueue_copy(dst_ptr=eigenvectors, src_buf=vectors_device)
            ctx.enqueue_copy(dst_ptr=statuses, src_buf=statuses_device)
            ctx.synchronize()
            return True
        except:
            return False


def batch_jacobi_svd[
    dtype: DType
](
    matrices: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    singular_values: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    matrices_u: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    matrices_v: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    rotations: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    var workers = 1
    if batch >= PARALLEL_BATCH:
        workers = min(
            num_physical_cores(),
            (batch + ITEMS_PER_WORKER - 1) // ITEMS_PER_WORKER,
        )

    @parameter
    def process(worker: Int):
        var start = worker * batch // workers
        var end = (worker + 1) * batch // workers
        for item in range(start, end):
            statuses[item] = Int64(
                1
                if jacobi_svd[dtype](
                    matrices + item * n * n,
                    singular_values + item * n,
                    matrices_u + item * n * n,
                    matrices_v + item * n * n,
                    work + item * n * n,
                    rotations + item * 4,
                    n,
                )
                else 0
            )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def gpu_svd_f64(
    matrices: UnsafePointer[Float64, AnyOrigin[mut=True]],
    singular_values: UnsafePointer[Float64, AnyOrigin[mut=True]],
    matrices_u: UnsafePointer[Float64, AnyOrigin[mut=True]],
    matrices_v: UnsafePointer[Float64, AnyOrigin[mut=True]],
    work: UnsafePointer[Float64, AnyOrigin[mut=True]],
    rotations: UnsafePointer[Float64, AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
):
    var item = Int(global_idx.x)
    if item >= batch:
        return
    statuses[item] = Int64(
        1
        if jacobi_svd[DType.float64](
            matrices + item * n * n,
            singular_values + item * n,
            matrices_u + item * n * n,
            matrices_v + item * n * n,
            work + item * n * n,
            rotations + item * 4,
            n,
        )
        else 0
    )


def batch_jacobi_svd_gpu_f64(
    matrices: UnsafePointer[Float64, AnyOrigin[mut=True]],
    singular_values: UnsafePointer[Float64, AnyOrigin[mut=True]],
    matrices_u: UnsafePointer[Float64, AnyOrigin[mut=True]],
    matrices_v: UnsafePointer[Float64, AnyOrigin[mut=True]],
    statuses: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int,
    batch: Int,
) -> Bool:
    comptime if not has_accelerator():
        return False
    else:
        try:
            var ctx = DeviceContext()
            var matrices_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n * n
            )
            var values_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n
            )
            var matrices_u_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n * n
            )
            var matrices_v_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n * n
            )
            var work_device = ctx.enqueue_create_buffer[DType.float64](
                batch * n * n
            )
            var rotations_device = ctx.enqueue_create_buffer[DType.float64](
                batch * 4
            )
            var statuses_device = ctx.enqueue_create_buffer[DType.int64](batch)
            ctx.enqueue_copy(dst_buf=matrices_device, src_ptr=matrices)
            var block_size = 256
            ctx.enqueue_function[gpu_svd_f64](
                matrices_device,
                values_device,
                matrices_u_device,
                matrices_v_device,
                work_device,
                rotations_device,
                statuses_device,
                n,
                batch,
                grid_dim=(batch + block_size - 1) // block_size,
                block_dim=block_size,
            )
            ctx.enqueue_copy(dst_ptr=singular_values, src_buf=values_device)
            ctx.enqueue_copy(dst_ptr=matrices_u, src_buf=matrices_u_device)
            ctx.enqueue_copy(dst_ptr=matrices_v, src_buf=matrices_v_device)
            ctx.enqueue_copy(dst_ptr=statuses, src_buf=statuses_device)
            ctx.synchronize()
            return True
        except:
            return False


# Eigen: Eigen/src/Core/StableNorm.h stable_norm_impl
@always_inline
def stable_norm[
    dtype: DType
](
    vector: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], n: Int
) -> Scalar[dtype]:
    var scale = scalar[dtype](0.0)
    var sum_squares = scalar[dtype](1.0)
    for i in range(n):
        var value = abs(vector[i])
        if value != scalar[dtype](0.0):
            if scale < value:
                var ratio = scale / value
                sum_squares = scalar[dtype](1.0) + sum_squares * ratio * ratio
                scale = value
            else:
                var ratio = value / scale
                sum_squares += ratio * ratio
    if scale == scalar[dtype](0.0):
        return scalar[dtype](0.0)
    return scale * sqrt(sum_squares)


def csr_matvec[
    dtype: DType
](
    values: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    indices: UnsafePointer[Int64, AnyOrigin[mut=True]],
    indptr: UnsafePointer[Int64, AnyOrigin[mut=True]],
    x: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    dst: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
):
    for row in range(n):
        var total = scalar[dtype](0.0)
        for position in range(Int(indptr[row]), Int(indptr[row + 1])):
            total += values[position] * x[Int(indices[position])]
        dst[row] = total


# Eigen: Eigen/src/IterativeLinearSolvers/BasicPreconditioners.h DiagonalPreconditioner::factorize
# Eigen: Eigen/src/IterativeLinearSolvers/ConjugateGradient.h internal::conjugate_gradient
def conjugate_gradient[
    dtype: DType
](
    values: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    indices: UnsafePointer[Int64, AnyOrigin[mut=True]],
    indptr: UnsafePointer[Int64, AnyOrigin[mut=True]],
    rhs: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    solution: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    work: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    error: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    n: Int,
    max_iterations: Int,
    tolerance: Scalar[dtype],
) -> Int:
    if n == 0:
        error[0] = scalar[dtype](0.0)
        return 0
    var residual = work
    var direction = work + n
    var preconditioned = work + 2 * n
    var product = work + 3 * n
    var inverse_diagonal = work + 4 * n
    csr_matvec[dtype](values, indices, indptr, solution, product, n)
    for i in range(n):
        residual[i] = rhs[i] - product[i]
        inverse_diagonal[i] = scalar[dtype](1.0)
        for position in range(Int(indptr[i]), Int(indptr[i + 1])):
            if Int(indices[position]) == i:
                if values[position] != scalar[dtype](0.0):
                    inverse_diagonal[i] = (
                        scalar[dtype](1.0) / values[position]
                    )
                break
    var rhs_norm = stable_norm[dtype](rhs, n)
    if rhs_norm == scalar[dtype](0.0):
        for i in range(n):
            solution[i] = scalar[dtype](0.0)
        error[0] = scalar[dtype](0.0)
        return 0
    var threshold = max(tolerance * rhs_norm, real_min[dtype]())
    var residual_norm = stable_norm[dtype](residual, n)
    if residual_norm < threshold:
        error[0] = residual_norm / rhs_norm
        return 0
    var residual_scale = scalar[dtype](1.0)
    var sqrt_min = sqrt(real_min[dtype]())
    var sqrt_max = sqrt(real_max[dtype]())
    if residual_norm < sqrt_min or residual_norm > sqrt_max:
        residual_scale = residual_norm
        for i in range(n):
            residual[i] /= residual_scale
        threshold /= residual_scale
    var abs_new = scalar[dtype](0.0)
    for i in range(n):
        direction[i] = inverse_diagonal[i] * residual[i]
        abs_new += residual[i] * direction[i]
    var iteration = 0
    while iteration < max_iterations:
        csr_matvec[dtype](
            values, indices, indptr, direction, product, n
        )
        var denominator = scalar[dtype](0.0)
        for i in range(n):
            denominator += direction[i] * product[i]
        if denominator == scalar[dtype](0.0) or not finite[dtype](denominator):
            error[0] = real_max[dtype]()
            return iteration
        var alpha = abs_new / denominator
        for i in range(n):
            solution[i] += residual_scale * alpha * direction[i]
            residual[i] -= alpha * product[i]
        residual_norm = stable_norm[dtype](residual, n)
        if residual_norm < threshold:
            error[0] = residual_norm / (rhs_norm / residual_scale)
            return iteration + 1
        var abs_old = abs_new
        abs_new = scalar[dtype](0.0)
        for i in range(n):
            preconditioned[i] = inverse_diagonal[i] * residual[i]
            abs_new += residual[i] * preconditioned[i]
        var beta = abs_new / abs_old
        for i in range(n):
            direction[i] = preconditioned[i] + beta * direction[i]
        iteration += 1
    error[0] = residual_norm / (rhs_norm / residual_scale)
    return iteration


@export("me_matmul_f64")
def me_matmul_f64(a: Int, b: Int, dst: Int, n: Int, batch: Int) abi("C"):
    matmul[DType.float64](ptr[DType.float64](a), ptr[DType.float64](b), ptr[DType.float64](dst), n, batch)


@export("me_matmul_f32")
def me_matmul_f32(a: Int, b: Int, dst: Int, n: Int, batch: Int) abi("C"):
    matmul[DType.float32](ptr[DType.float32](a), ptr[DType.float32](b), ptr[DType.float32](dst), n, batch)


@export("me_matvec_f64")
def me_matvec_f64(a: Int, x: Int, dst: Int, n: Int, batch: Int) abi("C"):
    matvec[DType.float64](ptr[DType.float64](a), ptr[DType.float64](x), ptr[DType.float64](dst), n, batch)


@export("me_matvec_f32")
def me_matvec_f32(a: Int, x: Int, dst: Int, n: Int, batch: Int) abi("C"):
    matvec[DType.float32](ptr[DType.float32](a), ptr[DType.float32](x), ptr[DType.float32](dst), n, batch)


@export("me_lu_f64")
def me_lu_f64(a: Int, b: Int, x: Int, work: Int, pivots: Int, n: Int) abi("C") -> Int:
    return 1 if lu_solve[DType.float64](ptr[DType.float64](a), ptr[DType.float64](b), ptr[DType.float64](x), ptr[DType.float64](work), iptr(pivots), n) else 0


@export("me_lu_f32")
def me_lu_f32(a: Int, b: Int, x: Int, work: Int, pivots: Int, n: Int) abi("C") -> Int:
    return 1 if lu_solve[DType.float32](ptr[DType.float32](a), ptr[DType.float32](b), ptr[DType.float32](x), ptr[DType.float32](work), iptr(pivots), n) else 0


@export("me_llt_f64")
def me_llt_f64(a: Int, b: Int, x: Int, work: Int, n: Int) abi("C") -> Int:
    return 1 if llt_solve[DType.float64](ptr[DType.float64](a), ptr[DType.float64](b), ptr[DType.float64](x), ptr[DType.float64](work), n) else 0


@export("me_llt_f32")
def me_llt_f32(a: Int, b: Int, x: Int, work: Int, n: Int) abi("C") -> Int:
    return 1 if llt_solve[DType.float32](ptr[DType.float32](a), ptr[DType.float32](b), ptr[DType.float32](x), ptr[DType.float32](work), n) else 0


@export("me_ldlt_f64")
def me_ldlt_f64(a: Int, b: Int, x: Int, work: Int, temp: Int, pivots: Int, n: Int) abi("C") -> Int:
    return 1 if ldlt_solve[DType.float64](ptr[DType.float64](a), ptr[DType.float64](b), ptr[DType.float64](x), ptr[DType.float64](work), ptr[DType.float64](temp), iptr(pivots), n) else 0


@export("me_ldlt_f32")
def me_ldlt_f32(a: Int, b: Int, x: Int, work: Int, temp: Int, pivots: Int, n: Int) abi("C") -> Int:
    return 1 if ldlt_solve[DType.float32](ptr[DType.float32](a), ptr[DType.float32](b), ptr[DType.float32](x), ptr[DType.float32](work), ptr[DType.float32](temp), iptr(pivots), n) else 0


@export("me_ldlt_batch_f64")
def me_ldlt_batch_f64(a: Int, b: Int, x: Int, work: Int, temp: Int, pivots: Int, statuses: Int, n: Int, batch: Int) abi("C"):
    batch_ldlt_solve[DType.float64](ptr[DType.float64](a), ptr[DType.float64](b), ptr[DType.float64](x), ptr[DType.float64](work), ptr[DType.float64](temp), iptr(pivots), iptr(statuses), n, batch)


@export("me_ldlt_batch_f32")
def me_ldlt_batch_f32(a: Int, b: Int, x: Int, work: Int, temp: Int, pivots: Int, statuses: Int, n: Int, batch: Int) abi("C"):
    batch_ldlt_solve[DType.float32](ptr[DType.float32](a), ptr[DType.float32](b), ptr[DType.float32](x), ptr[DType.float32](work), ptr[DType.float32](temp), iptr(pivots), iptr(statuses), n, batch)


@export("me_qr_f64")
def me_qr_f64(a: Int, b: Int, x: Int, work: Int, transformed: Int, m: Int, n: Int) abi("C") -> Int:
    return 1 if qr_solve[DType.float64](ptr[DType.float64](a), ptr[DType.float64](b), ptr[DType.float64](x), ptr[DType.float64](work), ptr[DType.float64](transformed), m, n) else 0


@export("me_qr_f32")
def me_qr_f32(a: Int, b: Int, x: Int, work: Int, transformed: Int, m: Int, n: Int) abi("C") -> Int:
    return 1 if qr_solve[DType.float32](ptr[DType.float32](a), ptr[DType.float32](b), ptr[DType.float32](x), ptr[DType.float32](work), ptr[DType.float32](transformed), m, n) else 0


@export("me_eigh_f64")
def me_eigh_f64(a: Int, values: Int, vectors: Int, work: Int, n: Int) abi("C") -> Int:
    return 1 if selfadjoint_eigh[DType.float64](ptr[DType.float64](a), ptr[DType.float64](values), ptr[DType.float64](vectors), ptr[DType.float64](work), n) else 0


@export("me_eigh_f32")
def me_eigh_f32(a: Int, values: Int, vectors: Int, work: Int, n: Int) abi("C") -> Int:
    return 1 if selfadjoint_eigh[DType.float32](ptr[DType.float32](a), ptr[DType.float32](values), ptr[DType.float32](vectors), ptr[DType.float32](work), n) else 0


@export("me_eigh_batch_f64")
def me_eigh_batch_f64(a: Int, values: Int, vectors: Int, work: Int, statuses: Int, n: Int, batch: Int) abi("C"):
    batch_selfadjoint_eigh[DType.float64](ptr[DType.float64](a), ptr[DType.float64](values), ptr[DType.float64](vectors), ptr[DType.float64](work), iptr(statuses), n, batch)


@export("me_eigh_batch_f32")
def me_eigh_batch_f32(a: Int, values: Int, vectors: Int, work: Int, statuses: Int, n: Int, batch: Int) abi("C"):
    batch_selfadjoint_eigh[DType.float32](ptr[DType.float32](a), ptr[DType.float32](values), ptr[DType.float32](vectors), ptr[DType.float32](work), iptr(statuses), n, batch)


@export("me_eigh_batch_gpu_f64")
def me_eigh_batch_gpu_f64(a: Int, values: Int, vectors: Int, statuses: Int, n: Int, batch: Int) abi("C") -> Int:
    return 1 if batch_selfadjoint_eigh_gpu_f64(ptr[DType.float64](a), ptr[DType.float64](values), ptr[DType.float64](vectors), iptr(statuses), n, batch) else 0


@export("me_svd_f64")
def me_svd_f64(a: Int, values: Int, u: Int, v: Int, work: Int, rotations: Int, n: Int) abi("C") -> Int:
    return 1 if jacobi_svd[DType.float64](ptr[DType.float64](a), ptr[DType.float64](values), ptr[DType.float64](u), ptr[DType.float64](v), ptr[DType.float64](work), ptr[DType.float64](rotations), n) else 0


@export("me_svd_f32")
def me_svd_f32(a: Int, values: Int, u: Int, v: Int, work: Int, rotations: Int, n: Int) abi("C") -> Int:
    return 1 if jacobi_svd[DType.float32](ptr[DType.float32](a), ptr[DType.float32](values), ptr[DType.float32](u), ptr[DType.float32](v), ptr[DType.float32](work), ptr[DType.float32](rotations), n) else 0


@export("me_svd_batch_f64")
def me_svd_batch_f64(a: Int, values: Int, u: Int, v: Int, work: Int, rotations: Int, statuses: Int, n: Int, batch: Int) abi("C"):
    batch_jacobi_svd[DType.float64](ptr[DType.float64](a), ptr[DType.float64](values), ptr[DType.float64](u), ptr[DType.float64](v), ptr[DType.float64](work), ptr[DType.float64](rotations), iptr(statuses), n, batch)


@export("me_svd_batch_f32")
def me_svd_batch_f32(a: Int, values: Int, u: Int, v: Int, work: Int, rotations: Int, statuses: Int, n: Int, batch: Int) abi("C"):
    batch_jacobi_svd[DType.float32](ptr[DType.float32](a), ptr[DType.float32](values), ptr[DType.float32](u), ptr[DType.float32](v), ptr[DType.float32](work), ptr[DType.float32](rotations), iptr(statuses), n, batch)


@export("me_svd_batch_gpu_f64")
def me_svd_batch_gpu_f64(a: Int, values: Int, u: Int, v: Int, statuses: Int, n: Int, batch: Int) abi("C") -> Int:
    return 1 if batch_jacobi_svd_gpu_f64(ptr[DType.float64](a), ptr[DType.float64](values), ptr[DType.float64](u), ptr[DType.float64](v), iptr(statuses), n, batch) else 0


@export("me_cg_f64")
def me_cg_f64(values: Int, indices: Int, indptr: Int, rhs: Int, x: Int, work: Int, error: Int, n: Int, max_iterations: Int, tolerance: Float64) abi("C") -> Int:
    return conjugate_gradient[DType.float64](ptr[DType.float64](values), iptr(indices), iptr(indptr), ptr[DType.float64](rhs), ptr[DType.float64](x), ptr[DType.float64](work), ptr[DType.float64](error), n, max_iterations, tolerance)


@export("me_cg_f32")
def me_cg_f32(values: Int, indices: Int, indptr: Int, rhs: Int, x: Int, work: Int, error: Int, n: Int, max_iterations: Int, tolerance: Float32) abi("C") -> Int:
    return conjugate_gradient[DType.float32](ptr[DType.float32](values), iptr(indices), iptr(indptr), ptr[DType.float32](rhs), ptr[DType.float32](x), ptr[DType.float32](work), ptr[DType.float32](error), n, max_iterations, tolerance)
