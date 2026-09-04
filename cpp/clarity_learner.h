#ifndef ACIES_CLARITY_LEARNER_H
#define ACIES_CLARITY_LEARNER_H

struct BetaPosterior {
    double alpha;
    double beta;

    BetaPosterior(double alpha_ = 2.0, double beta_ = 2.0);
    void update(bool correct);
    double sample();
    double mean() const;
    double variance() const;

private:
    double sample_gamma(double shape);
    double sample_normal();
};

struct ClarityLearner {
    int n_actions;
    BetaPosterior* posteriors;

    ClarityLearner(int n_actions_);
    ~ClarityLearner();
    void reset();
    void reset_posterior(int action_id);
    double sample(int action_id);
    void update(int action_id, bool correct);
    double mean(int action_id) const;
};

#endif
