package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "run":
		cmdRun()
	case "bench":
		cmdBench()
	case "config":
		cmdConfig()
	case "version":
		fmt.Println("ACIES v0.1.4")
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`ACIES — Adaptive Perception Control

Usage:
  acies run [options]     Run APC on a task
  acies bench [options]   Run benchmark
  acies config [profile]  Show hardware profile
  acies version           Show version

Examples:
  acies run --hardware jetson --threshold 0.92
  acies bench --iterations 1000 --hardware rpi
  acies config jetson`)
}

// ============================================================
// acies run
// ============================================================

func cmdRun() {
	fs := newRunFlags()
	if err := fs.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}
	hardware := fs.hardware
	threshold := fs.threshold
	maxSteps := fs.maxSteps
	verbose := fs.verbose
	difficulty := fs.difficulty

	profile, ok := HardwareProfiles()[*hardware]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown hardware profile: %s\n", *hardware)
		fmt.Fprintf(os.Stderr, "Available: default, jetson, rpi, gpu, tpu\n")
		os.Exit(1)
	}

	cfg := APCConfig{
		Prior: 0.5, Temperature: 1.0,
		ConfidenceThreshold: *threshold, MaxSteps: *maxSteps,
		MaxRisk: 2.0, EmergencyRisk: 4.0,
		Hardware: profile,
	}

	clarities := map[string]float64{
		"64p": 0.55, "128p": 0.65, "224p": 0.75,
		"320p": 0.82, "512p": 0.88, "1024p": 0.93,
		"crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
	}

	clarityFn := func(a Action) float64 {
		base := clarities[a.Name]
		adjusted := base * (1.0 - *difficulty*0.4)
		adjusted += (rand.Float64() - 0.5) * 0.05
		return math.Max(0.01, math.Min(0.99, adjusted))
	}

	if *verbose {
		fmt.Printf("Profile: %s\n", profile.Name)
		fmt.Printf("Threshold: %.2f | Max steps: %d | Difficulty: %.1f\n",
			*threshold, *maxSteps, *difficulty)
		fmt.Println(strings.Repeat("─", 50))
	}

	trueClass := rand.Intn(2)
	result := Run(cfg, trueClass, clarityFn)

	fmt.Printf("Decision: %d (true: %d) | %s\n", result.Decision, trueClass,
		boolStr(result.Correct, "CORRECT", "WRONG"))
	fmt.Printf("Cost: %.1f | Steps: %d | Final belief: %.3f | Risk: %.3f\n",
		result.TotalCost, result.NSteps, result.FinalBelief, result.FinalRisk)

	if *verbose {
		fmt.Println()
		for i, s := range result.Steps {
			fmt.Printf("  Step %d: %-10s clarity=%.2f cost=%.1f → obs=%d belief=%.3f\n",
				i, s.Action.Name, s.ClarityTrue, s.Cost, s.Observation, s.BeliefAfter)
		}
	}
}

// ============================================================
// acies bench
// ============================================================

func cmdBench() {
	fs := newBenchFlags()
	if err := fs.Parse(os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}
	hardware := fs.hardware
	iterations := fs.iterations
	threshold := fs.threshold

	profile, ok := HardwareProfiles()[*hardware]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown hardware profile: %s\n", *hardware)
		os.Exit(1)
	}

	cfg := APCConfig{
		Prior: 0.5, Temperature: 1.0,
		ConfidenceThreshold: *threshold, MaxSteps: 6,
		MaxRisk: 2.0, EmergencyRisk: 4.0,
		Hardware: profile,
	}

	clarities := map[string]float64{
		"64p": 0.55, "128p": 0.65, "224p": 0.75,
		"320p": 0.82, "512p": 0.88, "1024p": 0.93,
		"crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
	}

	fmt.Printf("ACIES Benchmark — %s — %d iterations\n", profile.Name, *iterations)
	fmt.Println(strings.Repeat("─", 55))

	start := time.Now()
	correct := 0
	totalCost := 0.0
	totalSteps := 0
	maxCost := 0.0

	for i := 0; i < *iterations; i++ {
		trueClass := rand.Intn(2)
		clarityFn := func(a Action) float64 {
			base := clarities[a.Name]
			adjusted := base * 0.85 // slight difficulty
			adjusted += (rand.Float64() - 0.5) * 0.05
			return math.Max(0.01, math.Min(0.99, adjusted))
		}
		r := Run(cfg, trueClass, clarityFn)
		if r.Correct {
			correct++
		}
		totalCost += r.TotalCost
		totalSteps += r.NSteps
		if r.TotalCost > maxCost {
			maxCost = r.TotalCost
		}
	}
	elapsed := time.Since(start)

	acc := float64(correct) / float64(*iterations)
	avgCost := totalCost / float64(*iterations)
	avgSteps := float64(totalSteps) / float64(*iterations)
	epc := avgCost / math.Max(acc, 1e-10)
	runsPerSec := float64(*iterations) / elapsed.Seconds()

	fmt.Printf("Accuracy:    %.1f%%\n", acc*100)
	fmt.Printf("Avg cost:    %.1f\n", avgCost)
	fmt.Printf("Max cost:    %.1f\n", maxCost)
	fmt.Printf("Avg steps:   %.1f\n", avgSteps)
	fmt.Printf("EPC:         %.1f\n", epc)
	fmt.Printf("Time:        %s (%.0f runs/sec)\n", elapsed.Truncate(time.Millisecond), runsPerSec)

	// Comparison: always 1024p
	cost1024 := float64(200)*profile.LatencyScale*profile.LatencyWeight +
		float64(140)*profile.EnergyScale*profile.EnergyWeight +
		float64(256)*profile.MemoryScale*profile.MemoryWeight
	savings := (1.0 - avgCost/cost1024) * 100

	fmt.Printf("\nvs Fixed 1024p (cost=%.0f): ", cost1024)
	if savings > 0 {
		fmt.Printf("%.0f%% savings\n", savings)
	} else {
		fmt.Printf("%.0f%% more expensive\n", -savings)
	}
}

// ============================================================
// acies config
// ============================================================

func cmdConfig() {
	if len(os.Args) < 3 {
		fmt.Println("Available hardware profiles:")
		for name, p := range HardwareProfiles() {
			fmt.Printf("  %-10s %s\n", name, p.Name)
		}
		fmt.Println("\nUsage: acies config <profile>")
		return
	}

	profile, ok := HardwareProfiles()[os.Args[2]]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown profile: %s\n", os.Args[2])
		os.Exit(1)
	}

	data, _ := json.MarshalIndent(profile, "", "  ")
	fmt.Println(string(data))
}

// ============================================================
// Helpers
// ============================================================

func boolStr(b bool, trueStr, falseStr string) string {
	if b {
		return trueStr
	}
	return falseStr
}

// Minimal flag set (no external deps)
// ============================================================
// Flag parsing (standard library)
// ============================================================

type runFlags struct {
	fs         *flag.FlagSet
	hardware   *string
	threshold  *float64
	maxSteps   *int
	verbose    *bool
	difficulty *float64
}

func newRunFlags() *runFlags {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	r := &runFlags{fs: fs}
	r.hardware = fs.String("hardware", "default", "Hardware profile (default/jetson/rpi/gpu/tpu)")
	r.threshold = fs.Float64("threshold", 0.95, "Confidence threshold")
	r.maxSteps = fs.Int("max-steps", 6, "Maximum perception steps")
	r.verbose = fs.Bool("verbose", false, "Verbose output")
	r.difficulty = fs.Float64("difficulty", 0.0, "Task difficulty (0=easy, 1=impossible)")
	return r
}

func (r *runFlags) Parse(args []string) error {
	return r.fs.Parse(args)
}

type benchFlags struct {
	fs         *flag.FlagSet
	hardware   *string
	iterations *int
	threshold  *float64
}

func newBenchFlags() *benchFlags {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	b := &benchFlags{fs: fs}
	b.hardware = fs.String("hardware", "default", "Hardware profile")
	b.iterations = fs.Int("iterations", 1000, "Number of iterations")
	b.threshold = fs.Float64("threshold", 0.95, "Confidence threshold")
	return b
}

func (b *benchFlags) Parse(args []string) error {
	return b.fs.Parse(args)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func baseDir() string {
	ex, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(ex)
}
