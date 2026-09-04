#include "clarity_learner.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

// ---- BetaPosterior ----

BetaPosterior::BetaPosterior(double alpha_, double beta_)
    : alpha(alpha_), beta(beta_) {}

void BetaPosterior::update(bool correct) {
    if (correct) alpha += 1.0;
    else beta += 1.0;
}

double BetaPosterior::mean() const {
    return alpha / (alpha + beta);
}

double BetaPosterior::variance() const {
    double ab = alpha + beta;
    return (alpha * beta) / (ab * ab * (ab + 1.0));
}

double BetaPosterior::sample() {
    if (alpha > 20.0 && beta > 20.0) {
        double m = mean();
        double v = variance();
        double s = std::sqrt(std::max(v, 1e-10));
        double z = sample_normal();
        double samp = m + s * z;
        return std::max(0.01, std::min(0.99, samp));
    }
    double x = sample_gamma(alpha);
    double y = sample_gamma(beta);
    if (x + y < 1e-10) return 0.5;
    return std::max(0.01, std::min(0.99, x / (x + y)));
}

double BetaPosterior::sample_gamma(double shape) {
    if (shape < 1.0) {
        return sample_gamma(shape + 1.0) * std::pow((double)std::rand() / RAND_MAX, 1.0 / shape);
    }
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    while (true) {
        double x;
        do { x = sample_normal(); } while (false);
        double v = 1.0 + c * x;
        if (v <= 0) continue;
        v = v * v * v;
        double u = (double)std::rand() / RAND_MAX;
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v;
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
    }
}

double BetaPosterior::sample_normal() {
    double u1 = std::max((double)std::rand() / RAND_MAX, 1e-10);
    double u2 = (double)std::rand() / RAND_MAX;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// ---- ClarityLearner ----

ClarityLearner::ClarityLearner(int n_actions_)
    : n_actions(n_actions_) {
    posteriors = new BetaPosterior[n_actions];
}

ClarityLearner::~ClarityLearner() {
    delete[] posteriors;
}

void ClarityLearner::reset() {
    for (int i = 0; i < n_actions; i++) {
        posteriors[i] = BetaPosterior(2.0, 2.0);
    }
}

void ClarityLearner::reset_posterior(int action_id) {
    if (action_id >= 0 && action_id < n_actions) {
        posteriors[action_id] = BetaPosterior(2.0, 2.0);
    }
}

double ClarityLearner::sample(int action_id) {
    return posteriors[action_id].sample();
}

void ClarityLearner::update(int action_id, bool correct) {
    posteriors[action_id].update(correct);
}

double ClarityLearner::mean(int action_id) const {
    return posteriors[action_id].mean();
}
