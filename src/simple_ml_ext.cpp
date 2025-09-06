#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t start = 0; start < m; start += batch) {
        size_t actual_batch = std::min(batch, m - start);
        std::vector<float> exp_logits(actual_batch * k);
        std::vector<float> grad(n * k, 0.0f);
        // Compute exp of logits
        for (size_t i = 0; i < actual_batch; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float tmp = 0.0f;
                for (size_t l = 0; l < n; ++l) {
                    tmp += X[(start + i) * n + l] * theta[l * k + j];
                }
                exp_logits[i * k + j] = std::exp(tmp);
            }
        }
        // Compute softmax probabilities
        for (size_t i = 0; i < actual_batch; ++i) {
            float sum_exp = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                sum_exp += exp_logits[i * k + j];
            }
            for (size_t j = 0; j < k; ++j) {
                exp_logits[i * k + j] /= sum_exp;
            }
        }
        // Compute one hot encoding
        for (size_t i = 0; i < actual_batch; ++i) {
            exp_logits[i * k + y[start + i]] -= 1.0f;
        }
        // Accumulate gradients
        for (size_t i = 0; i < actual_batch; ++i) {
            for (size_t j = 0; j < k; ++j) {
                for (size_t l = 0; l < n; ++l) {
                    grad[l * k + j] += X[(start + i) * n + l] * exp_logits[i * k + j];
                }
            }
        }
        // Update theta
        for (size_t j = 0; j < n * k; ++j) {
            theta[j] -= (lr / actual_batch) * grad[j];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
