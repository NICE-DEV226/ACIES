package main

import (
	"strings"
	"testing"
)

// TestFlagParsing exercises every invocation form described in issue #12,
// plus an unknown-flag case. It fails on the pre-fix hand-written parser
// because boolean flags swallow the next argument.
func TestFlagParsing(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		wantVerbose   bool
		wantHardware  string
		wantErrSubstr string
	}{
		{
			name:         "verbose alone",
			args:         []string{"run", "--verbose"},
			wantVerbose:  true,
			wantHardware: "default",
		},
		{
			name:         "hardware then verbose",
			args:         []string{"run", "--hardware", "jetson", "--verbose"},
			wantVerbose:  true,
			wantHardware: "jetson",
		},
		{
			name:         "verbose then hardware",
			args:         []string{"run", "--verbose", "--hardware", "jetson"},
			wantVerbose:  true,
			wantHardware: "jetson",
		},
		{
			name:         "verbose equals true",
			args:         []string{"run", "--verbose=true", "--hardware", "jetson"},
			wantVerbose:  true,
			wantHardware: "jetson",
		},
		{
			name:         "hardware then verbose equals true",
			args:         []string{"run", "--hardware", "jetson", "--verbose", "true"},
			wantVerbose:  true,
			wantHardware: "jetson",
		},
		{
			name:          "unknown flag",
			args:          []string{"run", "--nope"},
			wantErrSubstr: "flag provided but not defined",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fs := newRunFlags()
			err := fs.Parse(tt.args[1:])

			if tt.wantErrSubstr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrSubstr) {
					t.Fatalf("expected error containing %q, got %v", tt.wantErrSubstr, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if *fs.verbose != tt.wantVerbose {
				t.Errorf("verbose = %v, want %v", *fs.verbose, tt.wantVerbose)
			}
			if *fs.hardware != tt.wantHardware {
				t.Errorf("hardware = %q, want %q", *fs.hardware, tt.wantHardware)
			}
		})
	}
}
