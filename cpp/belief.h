#ifndef ACIES_BELIEF_H
#define ACIES_BELIEF_H

struct BeliefState {
    double prior;
    double belief;
    double temperature;
    double min_belief;
    double max_belief;
    int n_updates;

    BeliefState(double prior_ = 0.5, double temperature_ = 1.0);
    void reset();
    void update(int obs, double clarity);
    double risk() const;
    double confidence() const;
    int decision() const;
    double delta_risk(double clarity) const;
    double delta_risk_efficiency(double clarity, double cost) const;

private:
    double risk_after_action(double clarity) const;
};

#endif
