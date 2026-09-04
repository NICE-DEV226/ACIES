#include "belief.h"
#include <cmath>
#include <algorithm>

BeliefState::BeliefState(double prior_, double temperature_)
    : prior(prior_), belief(prior_), temperature(temperature_),
      min_belief(0.001), max_belief(0.999), n_updates(0) {}

void BeliefState::reset() {
    belief = prior;
    n_updates = 0;
}

void BeliefState::update(int obs, double clarity) {
    double p = clarity;
    if (temperature != 1.0) {
        double logit = std::log(std::max(p / (1.0 - p), 1e-10));
        double logit_scaled = logit / temperature;
        p = 1.0 / (1.0 + std::exp(-logit_scaled));
    }

    double p_obs_y1, p_obs_y0;
    if (obs == 1) {
        p_obs_y1 = p;
        p_obs_y0 = 1.0 - p;
    } else {
        p_obs_y1 = 1.0 - p;
        p_obs_y0 = p;
    }

    double p_obs = p_obs_y1 * belief + p_obs_y0 * (1.0 - belief);
    if (p_obs < 1e-15) return;

    double posterior = (p_obs_y1 * belief) / p_obs;
    belief = std::max(min_belief, std::min(max_belief, posterior));
    n_updates++;
}

double BeliefState::risk() const {
    return 10.0 * std::min(belief, 1.0 - belief);
}

double BeliefState::confidence() const {
    return std::max(belief, 1.0 - belief);
}

int BeliefState::decision() const {
    return (belief < 0.5) ? 0 : 1;
}

double BeliefState::risk_after_action(double clarity) const {
    double expected_risk = 0.0;
    for (int obs = 0; obs <= 1; obs++) {
        double p_obs, new_belief;
        if (obs == 1) {
            p_obs = clarity * belief + (1.0 - clarity) * (1.0 - belief);
            new_belief = (clarity * belief) / std::max(p_obs, 1e-15);
        } else {
            p_obs = (1.0 - clarity) * belief + clarity * (1.0 - belief);
            new_belief = ((1.0 - clarity) * belief) / std::max(p_obs, 1e-15);
        }
        new_belief = std::max(min_belief, std::min(max_belief, new_belief));
        double r = 10.0 * std::min(new_belief, 1.0 - new_belief);
        expected_risk += p_obs * r;
    }
    return expected_risk;
}

double BeliefState::delta_risk(double clarity) const {
    return risk() - risk_after_action(clarity);
}

double BeliefState::delta_risk_efficiency(double clarity, double cost) const {
    if (cost <= 0.0) return 0.0;
    return delta_risk(clarity) / cost;
}
