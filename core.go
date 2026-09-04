package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// BeliefState — Bayesian belief tracker for binary classification
type BeliefState struct {
	Prior       float64
	Belief      float64
	Temperature float64
	MinBelief   float64
	MaxBelief   float64
	NUpdates    int
}

func NewBelief(prior, temperature float64) *BeliefState {
	return &BeliefState{
		Prior:       prior,
		Belief:      prior,
		Temperature: temperature,
		MinBelief:   0.001,
		MaxBelief:   0.999,
	}
}

func (b *BeliefState) Reset() {
	b.Belief = b.Prior
	b.NUpdates = 0
}

func (b *BeliefState) Update(obs int, clarity float64) {
	p := clarity
	if b.Temperature != 1.0 {
		logit := math.Log(math.Max(p/(1-p), 1e-10))
		logitScaled := logit / b.Temperature
		p = 1.0 / (1.0 + math.Exp(-logitScaled))
	}

	var pObsY1, pObsY0 float64
	if obs == 1 {
		pObsY1 = p
		pObsY0 = 1 - p
	} else {
		pObsY1 = 1 - p
		pObsY0 = p
	}

	pObs := pObsY1*b.Belief + pObsY0*(1-b.Belief)
	if pObs < 1e-15 {
		return
	}

	posterior := (pObsY1 * b.Belief) / pObs
	b.Belief = math.Max(b.MinBelief, math.Min(b.MaxBelief, posterior))
	b.NUpdates++
}

func (b *BeliefState) Risk() float64 {
	return 10.0 * math.Min(b.Belief, 1-b.Belief)
}

func (b *BeliefState) Confidence() float64 {
	return math.Max(b.Belief, 1-b.Belief)
}

func (b *BeliefState) Decision() int {
	if b.Belief < 0.5 {
		return 0
	}
	return 1
}

func (b *BeliefState) RiskAfterAction(clarity float64) float64 {
	expectedRisk := 0.0
	for obs := 0; obs <= 1; obs++ {
		var pObs, newBelief float64
		if obs == 1 {
			pObs = clarity*b.Belief + (1-clarity)*(1-b.Belief)
			newBelief = (clarity * b.Belief) / math.Max(pObs, 1e-15)
		} else {
			pObs = (1-clarity)*b.Belief + clarity*(1-b.Belief)
			newBelief = ((1 - clarity) * b.Belief) / math.Max(pObs, 1e-15)
		}
		newBelief = math.Max(b.MinBelief, math.Min(b.MaxBelief, newBelief))
		r := 10.0 * math.Min(newBelief, 1-newBelief)
		expectedRisk += pObs * r
	}
	return expectedRisk
}

func (b *BeliefState) DeltaRisk(clarity float64) float64 {
	return b.Risk() - b.RiskAfterAction(clarity)
}

func (b *BeliefState) DeltaRiskEfficiency(clarity, cost float64) float64 {
	if cost <= 0 {
		return 0
	}
	return b.DeltaRisk(clarity) / cost
}

// ============================================================
// Actions
// ============================================================

type ActionType int

const (
	Resolution ActionType = iota
	Crop
)

type Action struct {
	ID          int
	Name        string
	Type        ActionType
	Resolution  int
	CropArea    float64
	BaseLatency float64 // ms
	BaseEnergy  float64 // mJ
	BaseMemory  float64 // MB
	PixelRatio  float64
}

type HardwareProfile struct {
	Name          string
	LatencyWeight float64
	EnergyWeight  float64
	MemoryWeight  float64
	LatencyScale  float64
	EnergyScale   float64
	MemoryScale   float64
}

func (a *Action) Cost(p HardwareProfile) float64 {
	return p.LatencyWeight*a.BaseLatency*p.LatencyScale +
		p.EnergyWeight*a.BaseEnergy*p.EnergyScale +
		p.MemoryWeight*a.BaseMemory*p.MemoryScale
}

func DefaultActions() []Action {
	actions := make([]Action, 0, 9)
	idx := 0

	type resEntry struct {
		res, lat, mem int
		energy        float64
	}
	resolutions := []resEntry{
		{64, 2, 8, 0.5},
		{128, 5, 16, 2},
		{224, 12, 32, 6},
		{320, 25, 64, 13},
		{512, 60, 128, 35},
		{1024, 200, 256, 140},
	}
	for _, r := range resolutions {
		pixelRatio := float64(r.res*r.res) / float64(1024*1024)
		actions = append(actions, Action{
			ID: idx, Name: fmt.Sprintf("%dp", r.res), Type: Resolution,
			Resolution: r.res, CropArea: 1.0,
			BaseLatency: float64(r.lat), BaseEnergy: r.energy,
			BaseMemory: float64(r.mem), PixelRatio: pixelRatio,
		})
		idx++
	}

	type cropEntry struct {
		res, lat, mem int
		area, energy  float64
	}
	crops := []cropEntry{
		{224, 8, 24, 0.05, 4},
		{320, 15, 40, 0.08, 8},
		{512, 35, 80, 0.12, 20},
	}
	for _, c := range crops {
		pixelRatio := c.area * float64(c.res*c.res) / float64(1024*1024)
		actions = append(actions, Action{
			ID: idx, Name: fmt.Sprintf("crop_%d", c.res), Type: Crop,
			Resolution: c.res, CropArea: c.area,
			BaseLatency: float64(c.lat), BaseEnergy: c.energy,
			BaseMemory: float64(c.mem), PixelRatio: pixelRatio,
		})
		idx++
	}

	return actions
}

func HardwareProfiles() map[string]HardwareProfile {
	return map[string]HardwareProfile{
		"default": {"Default", 0.4, 0.4, 0.2, 1.0, 1.0, 1.0},
		"jetson":  {"Jetson Orin Nano", 0.5, 0.3, 0.2, 0.6, 0.8, 1.0},
		"rpi":     {"Raspberry Pi 5", 0.3, 0.5, 0.2, 2.5, 0.4, 0.8},
		"gpu":     {"Desktop GPU (RTX 4090)", 0.6, 0.2, 0.2, 0.1, 3.0, 2.0},
		"tpu":     {"Edge TPU (Coral)", 0.4, 0.5, 0.1, 0.3, 0.1, 0.3},
	}
}

// ============================================================
// Clarity Learner (Thompson Sampling)
// ============================================================

type BetaPosterior struct {
	Alpha, Beta float64
}

func NewBetaPosterior(alpha, beta float64) BetaPosterior {
	return BetaPosterior{Alpha: alpha, Beta: beta}
}

func (bp *BetaPosterior) Update(correct bool) {
	if correct {
		bp.Alpha++
	} else {
		bp.Beta++
	}
}

func (bp *BetaPosterior) Sample() float64 {
	if bp.Alpha > 20 && bp.Beta > 20 {
		mean := bp.Alpha / (bp.Alpha + bp.Beta)
		ab := bp.Alpha + bp.Beta
		v := (bp.Alpha * bp.Beta) / (ab * ab * (ab + 1))
		std := math.Sqrt(math.Max(v, 1e-10))
		z := rand.NormFloat64()
		s := mean + std*z
		return math.Max(0.01, math.Min(0.99, s))
	}
	x := bp.sampleGamma(bp.Alpha)
	y := bp.sampleGamma(bp.Beta)
	if x+y < 1e-10 {
		return 0.5
	}
	return math.Max(0.01, math.Min(0.99, x/(x+y)))
}

func (bp *BetaPosterior) sampleGamma(shape float64) float64 {
	if shape < 1.0 {
		return bp.sampleGamma(shape+1) * math.Pow(rand.Float64(), 1.0/shape)
	}
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		var x float64
		for {
			x = rand.NormFloat64()
			v := 1.0 + c*x
			if v > 0 {
				vv := v * v * v
				u := rand.Float64()
				if u < 1.0-0.0331*x*x*x*x {
					return d * vv
				}
				if math.Log(u) < 0.5*x*x+d*(1-vv+math.Log(vv)) {
					return d * vv
				}
			}
		}
	}
}

func (bp *BetaPosterior) Mean() float64 {
	return bp.Alpha / (bp.Alpha + bp.Beta)
}

type ClarityLearner struct {
	NActions  int
	Posteriors []BetaPosterior
}

func NewClarityLearner(nActions int) *ClarityLearner {
	p := make([]BetaPosterior, nActions)
	for i := range p {
		p[i] = NewBetaPosterior(2.0, 2.0)
	}
	return &ClarityLearner{NActions: nActions, Posteriors: p}
}

func (l *ClarityLearner) Reset() {
	for i := range l.Posteriors {
		l.Posteriors[i] = NewBetaPosterior(2.0, 2.0)
	}
}

func (l *ClarityLearner) ResetPosterior(id int) {
	l.Posteriors[id] = NewBetaPosterior(2.0, 2.0)
}

func (l *ClarityLearner) Sample(id int) float64 {
	return l.Posteriors[id].Sample()
}

func (l *ClarityLearner) Update(id int, correct bool) {
	l.Posteriors[id].Update(correct)
}

func (l *ClarityLearner) Mean(id int) float64 {
	return l.Posteriors[id].Mean()
}

// ============================================================
// Safety Layer
// ============================================================

type SafetyConfig struct {
	MaxRisk            float64
	EmergencyRisk      float64
	MinObservations    int
	ConfidenceThreshold float64
	RiskMargin         float64
}

func DefaultSafetyConfig() SafetyConfig {
	return SafetyConfig{
		MaxRisk: 2.0, EmergencyRisk: 4.0,
		MinObservations: 1, ConfidenceThreshold: 0.95, RiskMargin: 0.6,
	}
}

type SafetyState struct {
	NViolations int
	NEmergency  int
	RiskHistory []float64
}

type SafetyLayer struct {
	Config SafetyConfig
	State  SafetyState
}

func NewSafetyLayer(cfg SafetyConfig) *SafetyLayer {
	return &SafetyLayer{Config: cfg}
}

func (s *SafetyLayer) Reset() {
	s.State = SafetyState{}
}

func (s *SafetyLayer) Select(belief *BeliefState, candidates []ScoredAction, nObs int, clarityEst map[int]float64) *Action {
	currentRisk := belief.Risk()

	if currentRisk >= s.Config.EmergencyRisk {
		s.State.NEmergency++
		return s.emergencyAction(belief, candidates)
	}
	if nObs < s.Config.MinObservations {
		return &candidates[0].Action
	}
	if belief.Confidence() >= s.Config.ConfidenceThreshold {
		return nil
	}

	safe := make([]ScoredAction, 0)
	for _, c := range candidates {
		clarityEstVal := 0.5
		if v, ok := clarityEst[c.Action.ID]; ok {
			clarityEstVal = v
		}
		expectedRisk := belief.RiskAfterAction(clarityEstVal)
		minClarity := 0.5
		if currentRisk >= s.Config.MaxRisk*0.7 {
			minClarity = 0.65
		}
		if expectedRisk <= s.Config.MaxRisk && clarityEstVal >= minClarity {
			safe = append(safe, c)
		}
	}

	if len(safe) == 0 {
		s.State.NEmergency++
		return s.emergencyAction(belief, candidates)
	}

	action := safe[0].Action
	s.State.RiskHistory = append(s.State.RiskHistory, currentRisk)
	return &action
}

func (s *SafetyLayer) emergencyAction(belief *BeliefState, candidates []ScoredAction) *Action {
	best := candidates[0]
	for _, c := range candidates {
		dr := belief.DeltaRisk(0.5 + 0.49*c.Action.PixelRatio)
		drBest := belief.DeltaRisk(0.5 + 0.49*best.Action.PixelRatio)
		if dr > drBest {
			best = c
		}
	}
	return &best.Action
}

func (s *SafetyLayer) CheckPostAction(belief *BeliefState) bool {
	risk := belief.Risk()
	s.State.RiskHistory = append(s.State.RiskHistory, risk)
	if risk > 4.5 {
		s.State.NViolations++
		return false
	}
	return true
}

func (s *SafetyLayer) ShouldAbstain(belief *BeliefState, nObs int) bool {
	if nObs < s.Config.MinObservations {
		return true
	}
	return belief.Confidence() < s.Config.ConfidenceThreshold
}

// ============================================================
// Controller
// ============================================================

type ScoredAction struct {
	Action  Action
	Score   float64
	Clarity float64
}

type APCConfig struct {
	Prior       float64
	Temperature float64
	ConfidenceThreshold float64
	MaxSteps    int
	MaxRisk     float64
	EmergencyRisk float64
	Hardware    HardwareProfile
}

func DefaultConfig() APCConfig {
	return APCConfig{
		Prior: 0.5, Temperature: 1.0,
		ConfidenceThreshold: 0.95, MaxSteps: 6,
		MaxRisk: 2.0, EmergencyRisk: 4.0,
		Hardware: HardwareProfiles()["default"],
	}
}

type Step struct {
	Action      Action
	Observation int
	BeliefAfter float64
	Cost        float64
	ClarityTrue float64
}

type Result struct {
	Decision    int
	Correct     bool
	TotalCost   float64
	NSteps      int
	Steps       []Step
	FinalBelief float64
	FinalRisk   float64
}

func Run(cfg APCConfig, trueClass int, clarityFn func(Action) float64) Result {
	belief := NewBelief(cfg.Prior, cfg.Temperature)
	learner := NewClarityLearner(9)
	safety := NewSafetyLayer(DefaultSafetyConfig())
	safety.Config.MaxRisk = cfg.MaxRisk
	safety.Config.EmergencyRisk = cfg.EmergencyRisk
	safety.Config.ConfidenceThreshold = cfg.ConfidenceThreshold
	actions := DefaultActions()

	totalCost := 0.0
	steps := make([]Step, 0)

	for step := 0; step < cfg.MaxSteps; step++ {
		// Score actions
		sampled := make([]float64, len(actions))
		for i := range actions {
			sampled[i] = learner.Sample(i)
		}
		totalObs := 0
		for _, p := range learner.Posteriors {
			totalObs += int(p.Alpha + p.Beta - 2)
			if totalObs < 1 {
				totalObs = 1
			}
		}

		scored := make([]ScoredAction, len(actions))
		for i, a := range actions {
			cost := a.Cost(cfg.Hardware)
			dr := belief.DeltaRiskEfficiency(sampled[i], cost)
			nI := float64(learner.Posteriors[i].Alpha+learner.Posteriors[i].Beta-2)
			if nI < 1 {
				nI = 1
			}
			exploration := 3.0 * math.Sqrt(math.Log(math.Max(float64(totalObs), 1)+1)/nI)
			if nI <= 1 {
				exploration *= 5.0
			}
			scored[i] = ScoredAction{Action: a, Score: dr + exploration, Clarity: sampled[i]}
		}

		// Sort by score desc
		for i := 0; i < len(scored); i++ {
			for j := i + 1; j < len(scored); j++ {
				if scored[j].Score > scored[i].Score {
					scored[i], scored[j] = scored[j], scored[i]
				}
			}
		}

		// Safety select
		clarityEst := make(map[int]float64)
		for i, a := range actions {
			clarityEst[a.ID] = learner.Mean(i)
		}
		safeAction := safety.Select(belief, scored, len(steps), clarityEst)
		if safeAction == nil {
			break
		}

		// Execute
		clarityTrue := clarityFn(*safeAction)
		cost := safeAction.Cost(cfg.Hardware)

		// Generate observation
		var obs int
		if trueClass == 1 {
			if rand.Float64() < clarityTrue {
				obs = 1
			} else {
				obs = 0
			}
		} else {
			if rand.Float64() < clarityTrue {
				obs = 0
			} else {
				obs = 1
			}
		}

		belief.Update(obs, clarityTrue)
		learner.Update(safeAction.ID, obs == trueClass)
		safety.CheckPostAction(belief)
		totalCost += cost

		steps = append(steps, Step{
			Action: *safeAction, Observation: obs,
			BeliefAfter: belief.Belief, Cost: cost, ClarityTrue: clarityTrue,
		})
	}

	return Result{
		Decision: belief.Decision(),
		Correct:  belief.Decision() == trueClass,
		TotalCost: totalCost,
		NSteps:   len(steps),
		Steps:    steps,
		FinalBelief: belief.Belief,
		FinalRisk: belief.Risk(),
	}
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
