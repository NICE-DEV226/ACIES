#include "belief.h"
#include "clarity_learner.h"
#include "acies.h"

// ---- BeliefState ----

acies_belief_t acies_belief_create(double prior, double temperature) {
    return new BeliefState(prior, temperature);
}

void acies_belief_destroy(acies_belief_t b) {
    delete static_cast<BeliefState*>(b);
}

void acies_belief_reset(acies_belief_t b) {
    static_cast<BeliefState*>(b)->reset();
}

void acies_belief_update(acies_belief_t b, int obs, double clarity) {
    static_cast<BeliefState*>(b)->update(obs, clarity);
}

double acies_belief_risk(acies_belief_t b) {
    return static_cast<BeliefState*>(b)->risk();
}

double acies_belief_confidence(acies_belief_t b) {
    return static_cast<BeliefState*>(b)->confidence();
}

int acies_belief_decision(acies_belief_t b) {
    return static_cast<BeliefState*>(b)->decision();
}

double acies_belief_delta_risk(acies_belief_t b, double clarity) {
    return static_cast<BeliefState*>(b)->delta_risk(clarity);
}

double acies_belief_delta_risk_efficiency(acies_belief_t b, double clarity, double cost) {
    return static_cast<BeliefState*>(b)->delta_risk_efficiency(clarity, cost);
}

// ---- ClarityLearner ----

acies_learner_t acies_learner_create(int n_actions) {
    return new ClarityLearner(n_actions);
}

void acies_learner_destroy(acies_learner_t l) {
    delete static_cast<ClarityLearner*>(l);
}

void acies_learner_reset(acies_learner_t l) {
    static_cast<ClarityLearner*>(l)->reset();
}

void acies_learner_reset_posterior(acies_learner_t l, int action_id) {
    static_cast<ClarityLearner*>(l)->reset_posterior(action_id);
}

double acies_learner_sample(acies_learner_t l, int action_id) {
    return static_cast<ClarityLearner*>(l)->sample(action_id);
}

void acies_learner_update(acies_learner_t l, int action_id, int correct) {
    static_cast<ClarityLearner*>(l)->update(action_id, correct != 0);
}

double acies_learner_mean(acies_learner_t l, int action_id) {
    return static_cast<ClarityLearner*>(l)->mean(action_id);
}
