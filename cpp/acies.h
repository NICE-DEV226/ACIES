#ifndef ACIES_H
#define ACIES_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef void* acies_belief_t;
typedef void* acies_learner_t;

// BeliefState
acies_belief_t acies_belief_create(double prior, double temperature);
void acies_belief_destroy(acies_belief_t b);
void acies_belief_reset(acies_belief_t b);
void acies_belief_update(acies_belief_t b, int obs, double clarity);
double acies_belief_risk(acies_belief_t b);
double acies_belief_confidence(acies_belief_t b);
int acies_belief_decision(acies_belief_t b);
double acies_belief_delta_risk(acies_belief_t b, double clarity);
double acies_belief_delta_risk_efficiency(acies_belief_t b, double clarity, double cost);

// ClarityLearner
acies_learner_t acies_learner_create(int n_actions);
void acies_learner_destroy(acies_learner_t l);
void acies_learner_reset(acies_learner_t l);
void acies_learner_reset_posterior(acies_learner_t l, int action_id);
double acies_learner_sample(acies_learner_t l, int action_id);
void acies_learner_update(acies_learner_t l, int action_id, int correct);
double acies_learner_mean(acies_learner_t l, int action_id);

#ifdef __cplusplus
}
#endif

#endif
