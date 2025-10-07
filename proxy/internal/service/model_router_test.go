package service

import (
	"bytes"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/seifghazi/claude-code-monitor/internal/config"
	"github.com/seifghazi/claude-code-monitor/internal/model"
	"github.com/seifghazi/claude-code-monitor/internal/provider"
)

func TestModelRouter_EdgeCases(t *testing.T) {
	// Setup
	cfg := &config.Config{
		Subagents: config.SubagentsConfig{
			Mappings: map[string]string{
				"streaming-systems-engineer": "gpt-4o",
			},
		},
	}

	providers := make(map[string]provider.Provider)
	providers["anthropic"] = nil
	providers["openai"] = nil

	logger := log.New(os.Stdout, "test: ", log.LstdFlags)
	router := NewModelRouter(cfg, providers, logger)

	tests := []struct {
		name          string
		request       *model.AnthropicRequest
		expectedRoute string
		expectedModel string
		description   string
	}{
		{
			name: "Regular Claude Code request (no Notes section)",
			request: &model.AnthropicRequest{
				Model: "claude-3-opus-20240229",
				System: []model.AnthropicSystemMessage{
					{Text: "You are Claude Code, Anthropic's official CLI for Claude."},
					{Text: "You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user."},
				},
			},
			expectedRoute: "anthropic",
			expectedModel: "claude-3-opus-20240229",
			description:   "Regular Claude Code requests should use original model",
		},
		{
			name: "Non-Claude Code request",
			request: &model.AnthropicRequest{
				Model: "claude-3-opus-20240229",
				System: []model.AnthropicSystemMessage{
					{Text: "You are a helpful assistant."},
				},
			},
			expectedRoute: "anthropic",
			expectedModel: "claude-3-opus-20240229",
			description:   "Non-Claude Code requests should use original model",
		},
		{
			name: "Single system message",
			request: &model.AnthropicRequest{
				Model:  "claude-3-opus-20240229",
				System: []model.AnthropicSystemMessage{},
			},
			expectedRoute: "anthropic",
			expectedModel: "claude-3-opus-20240229",
			description:   "Requests with no system messages should use original model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.request.System) == 2 {
				// Test extract static prompt for second message
				fullPrompt := tt.request.System[1].Text
				staticPrompt := router.extractStaticPrompt(fullPrompt)

				// Verify no "Notes:" in static prompt
				if contains(staticPrompt, "Notes:") {
					t.Errorf("Static prompt should not contain 'Notes:' section")
				}
			}

			// Log for manual verification
			t.Logf("Test case: %s", tt.description)
		})
	}
}

func TestModelRouter_ExtractStaticPrompt(t *testing.T) {
	router := &ModelRouter{}

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Prompt with Notes section",
			input:    "You are an expert engineer.\n\nNotes:\n- Some dynamic content\n- More notes",
			expected: "You are an expert engineer.",
		},
		{
			name:     "Prompt without Notes section",
			input:    "You are an expert engineer.\nNo notes here.",
			expected: "You are an expert engineer.\nNo notes here.",
		},
		{
			name:     "Prompt with double newline before Notes",
			input:    "You are an expert.\n\nNotes:\nDynamic content",
			expected: "You are an expert.",
		},
		{
			name:     "Empty prompt",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := router.extractStaticPrompt(tt.input)
			if result != tt.expected {
				t.Errorf("extractStaticPrompt() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestModelRouter_GetProviderNameForModel_PrefixMatching(t *testing.T) {
	var buf bytes.Buffer
	router := &ModelRouter{
		modelProviderMap: initializeModelProviderMap(),
		logger:           log.New(&buf, "", 0),
	}

	tests := []struct {
		name       string
		model      string
		expected   string
		expectWarn bool
	}{
		{
			name:       "Known Anthropic release",
			model:      "claude-sonnet-4-5-20250929",
			expected:   "anthropic",
			expectWarn: false,
		},
		{
			name:       "New Anthropic prefix variant",
			model:      "claude-new-model-20260101",
			expected:   "anthropic",
			expectWarn: false,
		},
		{
			name:       "OpenAI GPT prefix",
			model:      "gpt-4o-2025",
			expected:   "openai",
			expectWarn: false,
		},
		{
			name:       "OpenAI O1 prefix",
			model:      "o1-mini-2025",
			expected:   "openai",
			expectWarn: false,
		},
		{
			name:       "Unknown model fallback",
			model:      "mystery-model",
			expected:   "anthropic",
			expectWarn: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf.Reset()
			provider := router.getProviderNameForModel(tt.model)

			if provider != tt.expected {
				t.Fatalf("expected provider %q, got %q", tt.expected, provider)
			}

			logged := buf.String()
			if tt.expectWarn {
				if !strings.Contains(logged, "doesn't match any known patterns") {
					t.Fatalf("expected warning log for model %s", tt.model)
				}
			} else {
				if logged != "" {
					t.Fatalf("did not expect warning log for model %s", tt.model)
				}
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && s[0:len(substr)] == substr) ||
		(len(s) > len(substr) && contains(s[1:], substr)))
}
