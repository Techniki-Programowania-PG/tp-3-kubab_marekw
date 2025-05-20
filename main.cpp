#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <matplot/matplot.h>
#include <vector>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;
using namespace matplot;

std::vector<double> sine(double freq, double rate, int n) {
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i)
        y[i] = std::sin(2 * M_PI * freq * (i / rate));
    return y;
}

std::vector<double> cosine(double freq, double rate, int n) {
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i)
        y[i] = std::cos(2 * M_PI * freq * (i / rate));
    return y;
}

std::vector<double> square_wave(double freq, double rate, int n) {
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i)
        y[i] = std::sin(2 * M_PI * freq * (i / rate)) >= 0 ? 1.0 : -1.0;
    return y;
}

std::vector<double> sawtooth(double freq, double rate, int n) {
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double t = i / rate;
        double T = 1.0 / freq;
        y[i] = 2.0 * (t / T - std::floor(0.5 + t / T));
    }
    return y;
}

std::vector<double> noise(const std::vector<double>& signal, double level) {
    std::vector<double> noisy(signal.size());
    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, level);
    for (size_t i = 0; i < signal.size(); ++i)
        noisy[i] = signal[i] + dist(gen);
    return noisy;
}


std::vector<double> filt1d(const std::vector<double>& signal, const std::vector<double>& kernel) {
    size_t N = signal.size(), K = kernel.size();
    std::vector<double> result(N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < K; ++j) {
            int idx = static_cast<int>(i) - static_cast<int>(j);
            if (idx >= 0 && idx < static_cast<int>(N))
                result[i] += signal[idx] * kernel[j];
        }
    return result;
}

std::vector<std::vector<double>> filt2d(const std::vector<std::vector<double>>& mtx, const std::vector<std::vector<double>>& k) {
    size_t r = mtx.size(), c = mtx[0].size();
    size_t kr = k.size(), kc = k[0].size();
    size_t pr = kr / 2, pc = kc / 2;
    std::vector<std::vector<double>> out(r, std::vector<double>(c, 0.0));

    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            for (size_t m = 0; m < kr; ++m)
                for (size_t n = 0; n < kc; ++n) {
                    int x = static_cast<int>(i) + m - pr;
                    int y = static_cast<int>(j) + n - pc;
                    if (x >= 0 && x < static_cast<int>(r) && y >= 0 && y < static_cast<int>(c))
                        out[i][j] += mtx[x][y] * k[m][n];
                }

    return out;
}


std::vector<std::vector<double>> dft(const std::vector<double>& signal) {
    size_t N = signal.size();
    std::vector<std::vector<double>> output(N, std::vector<double>(2, 0.0));

    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum = 0;
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            sum += signal[n] * std::exp(std::complex<double>(0, angle));
        }
        output[k][0] = sum.real();
        output[k][1] = sum.imag();
    }
    return output;
}

std::vector<double> idft(const std::vector<std::vector<double>>& spectrum) {
    size_t N = spectrum.size();
    std::vector<double> signal(N, 0.0);

    for (size_t n = 0; n < N; ++n) {
        std::complex<double> sum = 0;
        for (size_t k = 0; k < N; ++k) {
            std::complex<double> X(spectrum[k][0], spectrum[k][1]);
            double angle = 2.0 * M_PI * k * n / N;
            sum += X * std::exp(std::complex<double>(0, angle));
        }
        signal[n] = sum.real() / N;
    }
    return signal;
}

void plot_dft(const std::vector<double>& signal, const std::string& title = "Widmo DFT") {
    size_t N = signal.size();
    std::vector<double> magnitude(N);

    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum = 0;
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            sum += signal[n] * std::exp(std::complex<double>(0, angle));
        }
        magnitude[k] = std::abs(sum);
    }

    std::vector<double> t(N);
    for (size_t i = 0; i < N; ++i) t[i] = i;

    figure(true); plot(t, magnitude); ::title(title); grid(on); show();
}


void plot1d(const std::vector<double>& y, const std::string& title = "") {
    std::vector<double> t(y.size());
    for (size_t i = 0; i < y.size(); ++i) t[i] = i;
    figure(true); plot(t, y); ::title(title); grid(on); show();
}

void plot2d(const std::vector<std::vector<double>>& mtx, const std::string& title = "") {
    figure(true);
    for (const auto& row : mtx) plot(row);
    ::title(title); grid(on); show();
}


PYBIND11_MODULE(signalgen, m) {
    m.def("sine", &sine);
    m.def("cos", &cosine);
    m.def("sq", &square_wave);
    m.def("saw", &sawtooth);
    m.def("noise", &noise);
    m.def("filt1d", &filt1d);
    m.def("filt2d", &filt2d);
    m.def("dft", &dft);
    m.def("idft", &idft);
    m.def("plot_dft", &plot_dft, py::arg("signal"), py::arg("title") = "Widmo DFT");
    m.def("plot1d", &plot1d, py::arg("y"), py::arg("title") = "");
    m.def("plot2d", &plot2d, py::arg("mtx"), py::arg("title") = "");
}
