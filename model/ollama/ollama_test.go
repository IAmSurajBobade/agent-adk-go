// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/model/ollama/api"
)

func TestNewModel(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		baseURL   string
		opts      *Options
		wantErr   bool
	}{
		{
			name:      "basic model creation",
			modelName: "llama3.2",
			baseURL:   "http://localhost:11434",
			opts:      nil,
			wantErr:   false,
		},
		{
			name:      "model with options",
			modelName: "mistral",
			baseURL:   "http://localhost:11434",
			opts: &Options{
				Temperature: float32Ptr(0.7),
				TopK:        float32Ptr(40),
			},
			wantErr: false,
		},
		{
			name:      "empty base URL uses default",
			modelName: "llama3.2",
			baseURL:   "",
			opts:      nil,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewModel(tt.modelName, tt.baseURL, tt.opts)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewModel() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got == nil {
				t.Error("NewModel() returned nil model")
			}
			if !tt.wantErr && got != nil {
				if got.Name() != tt.modelName {
					t.Errorf("Model.Name() = %v, want %v", got.Name(), tt.modelName)
				}
			}
		})
	}
}

func TestModel_Generate(t *testing.T) {
	tests := []struct {
		name          string
		modelName     string
		req           *model.LLMRequest
		serverResp    api.ChatResponse
		want          *model.LLMResponse
		wantErr       bool
		validateOpts  func(*testing.T, map[string]interface{})
	}{
		{
			name:      "simple text generation",
			modelName: "llama3.2",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("What is 2+2?", "user"),
				},
				Config: &genai.GenerateContentConfig{
					Temperature: float32Ptr(0.0),
				},
			},
			serverResp: api.ChatResponse{
				Model:     "llama3.2",
				CreatedAt: time.Now(),
				Message: api.Message{
					Role:    "assistant",
					Content: "4",
				},
				Done:              true,
				TotalDuration:     1000000000,
				LoadDuration:      500000000,
				PromptEvalCount:   10,
				PromptEvalDuration: 200000000,
				EvalCount:         5,
				EvalDuration:      300000000,
			},
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{Text: "4"},
					},
				},
				TurnComplete: true,
				FinishReason: genai.FinishReasonStop,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 5,
					TotalTokenCount:      15,
				},
			},
			validateOpts: func(t *testing.T, opts map[string]interface{}) {
				if temp, ok := opts["temperature"]; !ok {
					t.Errorf("temperature option missing")
				} else if fmt.Sprintf("%v", temp) != fmt.Sprintf("%v", float32(0.0)) {
					t.Errorf("temperature = %v, want 0.0", temp)
				}
			},
		},
		{
			name:      "generation with all options",
			modelName: "mistral",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hello", "user"),
				},
				Config: &genai.GenerateContentConfig{
					Temperature:      float32Ptr(0.8),
					TopK:             float32Ptr(40),
					TopP:             float32Ptr(0.95),
					MaxOutputTokens:  100,
					StopSequences:    []string{"\n", "END"},
					PresencePenalty:  float32Ptr(0.5),
					FrequencyPenalty: float32Ptr(0.5),
				},
			},
			serverResp: api.ChatResponse{
				Model: "mistral",
				Message: api.Message{
					Role:    "assistant",
					Content: "Hi there!",
				},
				Done:            true,
				PromptEvalCount: 5,
				EvalCount:       3,
			},
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{Text: "Hi there!"},
					},
				},
				TurnComplete: true,
				FinishReason: genai.FinishReasonStop,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     5,
					CandidatesTokenCount: 3,
					TotalTokenCount:      8,
				},
			},
			validateOpts: func(t *testing.T, opts map[string]interface{}) {
				expectedOpts := map[string]interface{}{
					"temperature":       float32(0.8),
					"top_k":             float32(40),
					"top_p":             float32(0.95),
					"num_predict":       int32(100),
					"presence_penalty":  float32(0.5),
					"frequency_penalty": float32(0.5),
				}
				for key, want := range expectedOpts {
					if got, ok := opts[key]; !ok {
						t.Errorf("option %s missing", key)
					} else if fmt.Sprintf("%v", got) != fmt.Sprintf("%v", want) {
						t.Errorf("option %s = %v, want %v", key, got, want)
					}
				}
				// Verify stop sequences
				if stop, ok := opts["stop"].([]string); ok {
					if len(stop) != 2 {
						t.Errorf("stop sequences count = %d, want 2", len(stop))
					}
				} else if stopInterface, ok := opts["stop"].([]interface{}); ok {
					if len(stopInterface) != 2 {
						t.Errorf("stop sequences count = %d, want 2", len(stopInterface))
					}
				}
			},
		},
		{
			name:      "multi-turn conversation",
			modelName: "llama3.2",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hi", "user"),
					genai.NewContentFromText("Hello! How can I help?", genai.RoleModel),
					genai.NewContentFromText("What's the weather?", "user"),
				},
			},
			serverResp: api.ChatResponse{
				Model: "llama3.2",
				Message: api.Message{
					Role:    "assistant",
					Content: "I don't have access to weather data.",
				},
				Done:            true,
				PromptEvalCount: 20,
				EvalCount:       10,
			},
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{Text: "I don't have access to weather data."},
					},
				},
				TurnComplete: true,
				FinishReason: genai.FinishReasonStop,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     20,
					CandidatesTokenCount: 10,
					TotalTokenCount:      30,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock server
			var capturedRequest *api.ChatRequest
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/chat" {
					// Capture and decode request
					capturedRequest = &api.ChatRequest{}
					if err := json.NewDecoder(r.Body).Decode(capturedRequest); err != nil {
						t.Errorf("failed to decode request: %v", err)
					}

					// Send response
					w.Header().Set("Content-Type", "application/json")
					if err := json.NewEncoder(w).Encode(tt.serverResp); err != nil {
						t.Errorf("failed to encode response: %v", err)
					}
				}
			}))
			defer server.Close()

			// Create model with mock server
			testModel, err := NewModel(tt.modelName, server.URL, nil)
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}

			// Call generate
			ctx := context.Background()
			for got, err := range testModel.GenerateContent(ctx, tt.req, false) {
				if (err != nil) != tt.wantErr {
					t.Errorf("Model.Generate() error = %v, wantErr %v", err, tt.wantErr)
					return
				}

				if !tt.wantErr {
					// Validate options were passed correctly
					if tt.validateOpts != nil && capturedRequest != nil {
						tt.validateOpts(t, capturedRequest.Options)
					}

					// Compare response
					ignoreFields := cmpopts.IgnoreFields(model.LLMResponse{}, "CustomMetadata", "AvgLogprobs", "Partial")
					if diff := cmp.Diff(tt.want, got, ignoreFields); diff != "" {
						t.Errorf("Model.Generate() mismatch (-want +got):\n%s", diff)
					}

					// Verify custom metadata exists
					if got.CustomMetadata == nil {
						t.Error("CustomMetadata should not be nil")
					}
				}
			}
		})
	}
}

func TestModel_GenerateStream(t *testing.T) {
	tests := []struct {
		name         string
		modelName    string
		req          *model.LLMRequest
		serverResps  []api.ChatResponse
		wantText     string
		wantPartials int
		wantErr      bool
	}{
		{
			name:      "streaming simple response",
			modelName: "llama3.2",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Count to 3", "user"),
				},
			},
			serverResps: []api.ChatResponse{
				{Model: "llama3.2", Message: api.Message{Role: "assistant", Content: "1"}, Done: false},
				{Model: "llama3.2", Message: api.Message{Role: "assistant", Content: " 2"}, Done: false},
				{Model: "llama3.2", Message: api.Message{Role: "assistant", Content: " 3"}, Done: false},
				{Model: "llama3.2", Message: api.Message{Role: "assistant", Content: ""}, Done: true, PromptEvalCount: 5, EvalCount: 3},
			},
			wantText:     "1 2 3",
			wantPartials: 3,
		},
		{
			name:      "streaming with final aggregation",
			modelName: "mistral",
			req: &model.LLMRequest{
				Contents: []*genai.Content{
					genai.NewContentFromText("Hi", "user"),
				},
			},
			serverResps: []api.ChatResponse{
				{Model: "mistral", Message: api.Message{Role: "assistant", Content: "Hello"}, Done: false},
				{Model: "mistral", Message: api.Message{Role: "assistant", Content: " there"}, Done: false},
				{Model: "mistral", Message: api.Message{Role: "assistant", Content: "!"}, Done: false},
				{Model: "mistral", Message: api.Message{Role: "assistant", Content: ""}, Done: true, PromptEvalCount: 2, EvalCount: 3},
			},
			wantText:     "Hello there!",
			wantPartials: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock streaming server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/chat" {
					w.Header().Set("Content-Type", "application/x-ndjson")

					// Send each response as newline-delimited JSON
					for _, resp := range tt.serverResps {
						data, err := json.Marshal(resp)
						if err != nil {
							t.Errorf("failed to marshal response: %v", err)
							return
						}
						w.Write(data)
						w.Write([]byte("\n"))
						if f, ok := w.(http.Flusher); ok {
							f.Flush()
						}
					}
				}
			}))
			defer server.Close()

			testModel, err := NewModel(tt.modelName, server.URL, nil)
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}

			// Collect streaming responses
			ctx := context.Background()
			var partialText strings.Builder
			var finalText strings.Builder
			var partialCount int
			var finalResp *model.LLMResponse

			for resp, err := range testModel.GenerateContent(ctx, tt.req, true) {
				if (err != nil) != tt.wantErr {
					t.Errorf("Model.GenerateStream() error = %v, wantErr %v", err, tt.wantErr)
					return
				}

				if !tt.wantErr && resp != nil {
					if resp.Content != nil && len(resp.Content.Parts) > 0 {
						text := resp.Content.Parts[0].Text
						if resp.Partial {
							partialCount++
							partialText.WriteString(text)
						} else {
							finalText.WriteString(text)
						}
					}
					finalResp = resp
				}
			}

			if !tt.wantErr {
				// Check that we got the expected text in either partials or final
				gotPartial := partialText.String()
				gotFinal := finalText.String()

				// The aggregated final text should match expectations
				if gotFinal != tt.wantText {
					t.Errorf("final aggregated text = %q, want %q", gotFinal, tt.wantText)
				}

				// Partials should add up to the same text
				if gotPartial != tt.wantText {
					t.Errorf("partial text = %q, want %q", gotPartial, tt.wantText)
				}

				if partialCount < tt.wantPartials {
					t.Errorf("partial responses = %d, want at least %d", partialCount, tt.wantPartials)
				}

				if finalResp == nil || !finalResp.TurnComplete {
					t.Error("final response should have TurnComplete = true")
				}
			}
		})
	}
}

func TestOptions_AllFields(t *testing.T) {
	// Test that all option fields can be set and are properly converted
	opts := &Options{
		// Sampling
		Temperature: float32Ptr(0.9),
		TopK:        float32Ptr(50),
		TopP:        float32Ptr(0.9),
		MinP:        float32Ptr(0.05),
		TypicalP:    float32Ptr(1.0),
		TFSZ:        float32Ptr(1.0),

		// Context & Generation
		NumCtx:     intPtr(4096),
		NumPredict: intPtr(256),
		NumKeep:    intPtr(5),
		NumBatch:   intPtr(512),
		NumThread:  intPtr(8),

		// Repetition Control
		RepeatLastN:      intPtr(64),
		RepeatPenalty:    float32Ptr(1.1),
		PresencePenalty:  float32Ptr(0.0),
		FrequencyPenalty: float32Ptr(0.0),
		PenalizeNewline:  boolPtr(false),

		// Mirostat
		Mirostat:    intPtr(0),
		MirostatTau: float32Ptr(5.0),
		MirostatEta: float32Ptr(0.1),

		// Hardware
		NumGPU:  intPtr(1),
		MainGPU: intPtr(0),
		NUMA:    boolPtr(false),
		LowVRAM: boolPtr(false),
		UseMMap: boolPtr(true),
		UseMLock: boolPtr(false),
		VocabOnly: boolPtr(false),

		// Other
		Seed: intPtr(42),
		Stop: []string{"END", "\n\n"},

		// Request-level
		KeepAlive: durationPtr(5 * time.Minute),
		Format:    "json",
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chat" {
			var req api.ChatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("failed to decode request: %v", err)
				return
			}

			// Verify all options are present
			expectedOptions := map[string]interface{}{
				"temperature":       float32(0.9),
				"top_k":             float32(50),
				"top_p":             float32(0.9),
				"min_p":             float32(0.05),
				"typical_p":         float32(1.0),
				"tfs_z":             float32(1.0),
				"num_ctx":           4096,
				"num_predict":       256,
				"num_keep":          5,
				"num_batch":         512,
				"num_thread":        8,
				"repeat_last_n":     64,
				"repeat_penalty":    float32(1.1),
				"presence_penalty":  float32(0.0),
				"frequency_penalty": float32(0.0),
				"penalize_newline":  false,
				"mirostat":          0,
				"mirostat_tau":      float32(5.0),
				"mirostat_eta":      float32(0.1),
				"num_gpu":           1,
				"main_gpu":          0,
				"numa":              false,
				"low_vram":          false,
				"use_mmap":          true,
				"use_mlock":         false,
				"vocab_only":        false,
				"seed":              42,
			}

			for key, want := range expectedOptions {
				got, ok := req.Options[key]
				if !ok {
					t.Errorf("option %s not found in request", key)
				} else if fmt.Sprintf("%v", got) != fmt.Sprintf("%v", want) {
					t.Errorf("option %s = %v, want %v", key, got, want)
				}
			}

			// Verify stop sequences
			if stop, ok := req.Options["stop"].([]string); ok {
				if len(stop) != 2 {
					t.Errorf("stop sequences count = %d, want 2", len(stop))
				}
			} else if stopInterface, ok := req.Options["stop"].([]interface{}); ok {
				if len(stopInterface) != 2 {
					t.Errorf("stop sequences count = %d, want 2", len(stopInterface))
				}
			}

			// Verify format and keep_alive
			if req.Format != "json" {
				t.Errorf("format = %v, want json", req.Format)
			}
			if req.KeepAlive == nil || req.KeepAlive.Duration != 5*time.Minute {
				t.Errorf("keep_alive = %v, want 5m", req.KeepAlive)
			}

			// Send response
			resp := api.ChatResponse{
				Model:   "llama3.2",
				Message: api.Message{Role: "assistant", Content: "test"},
				Done:    true,
			}
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer server.Close()

	testModel, err := NewModel("llama3.2", server.URL, opts)
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("test", "user")},
	}

	ctx := context.Background()
	for _, err := range testModel.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Errorf("GenerateContent() error = %v", err)
		}
	}
}

func TestGenaiContentsToOllamaMessages(t *testing.T) {
	tests := []struct {
		name     string
		contents []*genai.Content
		want     []api.Message
		wantErr  bool
	}{
		{
			name: "simple text message",
			contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
			},
			want: []api.Message{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "multi-turn conversation",
			contents: []*genai.Content{
				genai.NewContentFromText("Hi", "user"),
				genai.NewContentFromText("Hello!", genai.RoleModel),
				genai.NewContentFromText("How are you?", "user"),
			},
			want: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "How are you?"},
			},
		},
		{
			name: "message with multiple text parts",
			contents: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "Part 1"},
						{Text: "Part 2"},
					},
				},
			},
			want: []api.Message{
				{Role: "user", Content: "Part 1\nPart 2"},
			},
		},
		{
			name: "function call",
			contents: []*genai.Content{
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "get_weather",
								Args: map[string]interface{}{"city": "Paris"},
							},
						},
					},
				},
			},
			want: []api.Message{
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: json.RawMessage(`{"city":"Paris"}`),
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := genaiContentsToOllamaMessages(tt.contents)
			if (err != nil) != tt.wantErr {
				t.Errorf("genaiContentsToOllamaMessages() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if len(got) != len(tt.want) {
				t.Errorf("message count = %d, want %d", len(got), len(tt.want))
				return
			}

			for i := range got {
				if got[i].Role != tt.want[i].Role {
					t.Errorf("message[%d].Role = %v, want %v", i, got[i].Role, tt.want[i].Role)
				}
				if got[i].Content != tt.want[i].Content {
					t.Errorf("message[%d].Content = %v, want %v", i, got[i].Content, tt.want[i].Content)
				}
				if len(got[i].ToolCalls) != len(tt.want[i].ToolCalls) {
					t.Errorf("message[%d] tool calls count = %d, want %d", i, len(got[i].ToolCalls), len(tt.want[i].ToolCalls))
				}
			}
		})
	}
}

func TestMapRole(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"user", "user"},
		{genai.RoleModel, "assistant"},
		{"system", "system"},
		{"tool", "tool"},
		{"unknown", "user"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := mapRole(tt.input)
			if got != tt.want {
				t.Errorf("mapRole(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// Helper functions

func float32Ptr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

func durationPtr(d time.Duration) *time.Duration {
	return &d
}
