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

// Package ollama implements the [model.LLM] interface for Ollama models.
package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/ollama/api"
)

// ollamaModel implements the model.LLM interface for Ollama.
type ollamaModel struct {
	client *api.Client
	name   string
	config *ClientConfig
}

// ClientConfig contains configuration for the Ollama client.
// All fields are optional and will use Ollama defaults if not specified.
type ClientConfig struct {
	// BaseURL is the Ollama server URL (default: "http://localhost:11434")
	BaseURL string
	// Sampling Parameters
	Temperature *float32 `json:"temperature,omitempty"` // Controls randomness (default: 0.8)
	TopK        *float32 `json:"top_k,omitempty"`       // Top-k sampling
	TopP        *float32 `json:"top_p,omitempty"`       // Top-p (nucleus) sampling
	MinP        *float32 `json:"min_p,omitempty"`       // Minimum probability threshold
	TypicalP    *float32 `json:"typical_p,omitempty"`   // Typical sampling parameter
	TFSZ        *float32 `json:"tfs_z,omitempty"`       // Tail free sampling

	// Context & Generation
	NumCtx     *int `json:"num_ctx,omitempty"`     // Context window size (default: 2048)
	NumPredict *int `json:"num_predict,omitempty"` // Max tokens to generate (default: -1, infinite)
	NumKeep    *int `json:"num_keep,omitempty"`    // Tokens to keep from initial prompt
	NumBatch   *int `json:"num_batch,omitempty"`   // Batch size for processing
	NumThread  *int `json:"num_thread,omitempty"`  // CPU threads to use

	// Repetition Control
	RepeatLastN      *int     `json:"repeat_last_n,omitempty"`     // How far back to prevent repetition
	RepeatPenalty    *float32 `json:"repeat_penalty,omitempty"`    // Penalty for repetitions (default: 1.1)
	PresencePenalty  *float32 `json:"presence_penalty,omitempty"`  // Presence penalty
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"` // Frequency penalty
	PenalizeNewline  *bool    `json:"penalize_newline,omitempty"`  // Penalize newline tokens

	// Mirostat Sampling
	Mirostat    *int     `json:"mirostat,omitempty"`     // Enable Mirostat (0=disabled, 1/2=enabled)
	MirostatTau *float32 `json:"mirostat_tau,omitempty"` // Mirostat target entropy
	MirostatEta *float32 `json:"mirostat_eta,omitempty"` // Mirostat learning rate

	// Hardware & Performance
	NumGPU    *int  `json:"num_gpu,omitempty"`    // Number of GPUs to use
	MainGPU   *int  `json:"main_gpu,omitempty"`   // Main GPU to use
	NUMA      *bool `json:"numa,omitempty"`       // NUMA support
	LowVRAM   *bool `json:"low_vram,omitempty"`   // Low VRAM mode
	UseMMap   *bool `json:"use_mmap,omitempty"`   // Use memory mapping
	UseMLock  *bool `json:"use_mlock,omitempty"`  // Use memory locking
	VocabOnly *bool `json:"vocab_only,omitempty"` // Load vocabulary only

	// Other
	Seed *int     `json:"seed,omitempty"` // Random seed for reproducibility
	Stop []string `json:"stop,omitempty"` // Stop sequences

	// Request-level options
	KeepAlive *time.Duration `json:"-"` // How long to keep model loaded
	Format    string         `json:"-"` // Response format ("json" or JSON schema)
}

// NewModel returns [model.LLM], backed by the Ollama API.
//
// It uses the provided context and configuration to initialize the Ollama client.
// The modelName specifies which Ollama model to use (e.g., "llama3.2", "mistral").
// The cfg parameter configures the client behavior; pass nil to use defaults.
//
// An error is returned if the client fails to initialize.
func NewModel(_ context.Context, modelName string, cfg *ClientConfig) (model.LLM, error) {
	if cfg == nil {
		cfg = &ClientConfig{}
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	if err := client.SetBaseURL(baseURL); err != nil {
		return nil, fmt.Errorf("failed to set base URL: %w", err)
	}

	return &ollamaModel{
		name:   modelName,
		client: client,
		config: cfg,
	}, nil
}

func (m *ollamaModel) Name() string {
	return m.name
}

// GenerateContent calls the underlying Ollama model.
func (m *ollamaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// generate calls the model synchronously.
func (m *ollamaModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	messages, err := genaiContentsToOllamaMessages(req.Contents)
	if err != nil {
		return nil, fmt.Errorf("failed to convert contents: %w", err)
	}

	chatReq := &api.ChatRequest{
		Model:    m.name,
		Messages: messages,
		Stream:   new(bool), // false
		Options:  m.buildOllamaOptions(req),
	}

	if m.config != nil {
		if m.config.KeepAlive != nil {
			chatReq.KeepAlive = &api.Duration{Duration: *m.config.KeepAlive}
		}
		if m.config.Format != "" {
			chatReq.Format = m.config.Format
		}
	}

	// Merge tools from request config if available
	if req.Config != nil && len(req.Config.Tools) > 0 {
		chatReq.Tools = convertGenaiToolsToOllama(req.Config.Tools)
	}

	var response api.ChatResponse
	err = m.client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
		response = resp
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to call Ollama: %w", err)
	}

	return ollamaChatResponseToLLMResponse(&response, false), nil
}

// generateStream returns a stream of responses from the model.
func (m *ollamaModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	aggregator := llminternal.NewStreamingResponseAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		messages, err := genaiContentsToOllamaMessages(req.Contents)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert contents: %w", err))
			return
		}

		chatReq := &api.ChatRequest{
			Model:    m.name,
			Messages: messages,
			Stream:   boolPtr(true),
			Options:  m.buildOllamaOptions(req),
		}

		if m.config != nil {
			if m.config.KeepAlive != nil {
				chatReq.KeepAlive = &api.Duration{Duration: *m.config.KeepAlive}
			}
			if m.config.Format != "" {
				chatReq.Format = m.config.Format
			}
		}

		// Merge tools from request config if available
		if req.Config != nil && len(req.Config.Tools) > 0 {
			chatReq.Tools = convertGenaiToolsToOllama(req.Config.Tools)
		}

		err = m.client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
			// Convert Ollama response to genai format
			genResp := ollamaChatResponseToGenaiResponse(&resp, !resp.Done)

			// Process through aggregator
			for llmResponse, err := range aggregator.ProcessResponse(ctx, genResp) {
				if !yield(llmResponse, err) {
					return fmt.Errorf("consumer stopped")
				}
			}
			return nil
		})

		if err != nil {
			yield(nil, err)
			return
		}

		// Send final aggregated response
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

// buildOllamaOptions converts ClientConfig and LLMRequest config to Ollama API options.
func (m *ollamaModel) buildOllamaOptions(req *model.LLMRequest) map[string]interface{} {
	options := make(map[string]interface{})

	// First, apply client-level options
	if m.config != nil {
		if m.config.Temperature != nil {
			options["temperature"] = *m.config.Temperature
		}
		if m.config.TopK != nil {
			options["top_k"] = *m.config.TopK
		}
		if m.config.TopP != nil {
			options["top_p"] = *m.config.TopP
		}
		if m.config.MinP != nil {
			options["min_p"] = *m.config.MinP
		}
		if m.config.TypicalP != nil {
			options["typical_p"] = *m.config.TypicalP
		}
		if m.config.TFSZ != nil {
			options["tfs_z"] = *m.config.TFSZ
		}
		if m.config.NumCtx != nil {
			options["num_ctx"] = *m.config.NumCtx
		}
		if m.config.NumPredict != nil {
			options["num_predict"] = *m.config.NumPredict
		}
		if m.config.NumKeep != nil {
			options["num_keep"] = *m.config.NumKeep
		}
		if m.config.NumBatch != nil {
			options["num_batch"] = *m.config.NumBatch
		}
		if m.config.NumThread != nil {
			options["num_thread"] = *m.config.NumThread
		}
		if m.config.RepeatLastN != nil {
			options["repeat_last_n"] = *m.config.RepeatLastN
		}
		if m.config.RepeatPenalty != nil {
			options["repeat_penalty"] = *m.config.RepeatPenalty
		}
		if m.config.PresencePenalty != nil {
			options["presence_penalty"] = *m.config.PresencePenalty
		}
		if m.config.FrequencyPenalty != nil {
			options["frequency_penalty"] = *m.config.FrequencyPenalty
		}
		if m.config.PenalizeNewline != nil {
			options["penalize_newline"] = *m.config.PenalizeNewline
		}
		if m.config.Mirostat != nil {
			options["mirostat"] = *m.config.Mirostat
		}
		if m.config.MirostatTau != nil {
			options["mirostat_tau"] = *m.config.MirostatTau
		}
		if m.config.MirostatEta != nil {
			options["mirostat_eta"] = *m.config.MirostatEta
		}
		if m.config.NumGPU != nil {
			options["num_gpu"] = *m.config.NumGPU
		}
		if m.config.MainGPU != nil {
			options["main_gpu"] = *m.config.MainGPU
		}
		if m.config.NUMA != nil {
			options["numa"] = *m.config.NUMA
		}
		if m.config.LowVRAM != nil {
			options["low_vram"] = *m.config.LowVRAM
		}
		if m.config.UseMMap != nil {
			options["use_mmap"] = *m.config.UseMMap
		}
		if m.config.UseMLock != nil {
			options["use_mlock"] = *m.config.UseMLock
		}
		if m.config.VocabOnly != nil {
			options["vocab_only"] = *m.config.VocabOnly
		}
		if m.config.Seed != nil {
			options["seed"] = *m.config.Seed
		}
		if m.config.Stop != nil {
			options["stop"] = m.config.Stop
		}
	}

	// Override with request-level config if present
	if req.Config != nil {
		if req.Config.Temperature != nil {
			options["temperature"] = *req.Config.Temperature
		}
		if req.Config.TopK != nil {
			options["top_k"] = *req.Config.TopK
		}
		if req.Config.TopP != nil {
			options["top_p"] = *req.Config.TopP
		}
		if req.Config.MaxOutputTokens != 0 {
			options["num_predict"] = req.Config.MaxOutputTokens
		}
		if req.Config.StopSequences != nil {
			options["stop"] = req.Config.StopSequences
		}
		if req.Config.PresencePenalty != nil {
			options["presence_penalty"] = *req.Config.PresencePenalty
		}
		if req.Config.FrequencyPenalty != nil {
			options["frequency_penalty"] = *req.Config.FrequencyPenalty
		}
	}

	return options
}

// genaiContentsToOllamaMessages converts genai.Content to Ollama messages.
func genaiContentsToOllamaMessages(contents []*genai.Content) ([]api.Message, error) {
	messages := make([]api.Message, 0, len(contents))

	for _, content := range contents {
		if content == nil {
			continue
		}

		msg := api.Message{
			Role: mapRole(content.Role),
		}

		// Build content from parts
		var textParts []string
		var images []api.ImageData
		var toolCalls []api.ToolCall

		for _, part := range content.Parts {
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
			if part.InlineData != nil {
				images = append(images, api.ImageData(part.InlineData.Data))
			}
			if part.FunctionCall != nil {
				// Convert function call to tool call
				argsJSON, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal function args: %w", err)
				}
				toolCalls = append(toolCalls, api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      part.FunctionCall.Name,
						Arguments: argsJSON,
					},
				})
			}
			if part.FunctionResponse != nil {
				// This is a tool response message, handle separately
				respJSON, err := json.Marshal(part.FunctionResponse.Response)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal function response: %w", err)
				}
				msg.Content = string(respJSON)
			}
		}

		// Set message content
		if len(textParts) > 0 {
			msg.Content = joinTextParts(textParts)
		}
		if len(images) > 0 {
			msg.Images = images
		}
		if len(toolCalls) > 0 {
			msg.ToolCalls = toolCalls
		}

		messages = append(messages, msg)
	}

	return messages, nil
}

// ollamaChatResponseToGenaiResponse converts Ollama response to genai format for aggregator.
func ollamaChatResponseToGenaiResponse(resp *api.ChatResponse, partial bool) *genai.GenerateContentResponse {
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	if resp.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{
			Text: resp.Message.Content,
		})
	}

	// Handle tool calls
	if len(resp.Message.ToolCalls) > 0 {
		for _, tc := range resp.Message.ToolCalls {
			var args map[string]interface{}
			if err := json.Unmarshal(tc.Function.Arguments, &args); err == nil {
				content.Parts = append(content.Parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						Name: tc.Function.Name,
						Args: args,
					},
				})
			}
		}
	}

	candidate := &genai.Candidate{
		Content: content,
	}

	if resp.Done {
		candidate.FinishReason = genai.FinishReasonStop
	}

	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{candidate},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.PromptEvalCount),
			CandidatesTokenCount: int32(resp.EvalCount),
			TotalTokenCount:      int32(resp.PromptEvalCount + resp.EvalCount),
		},
	}
}

// ollamaChatResponseToLLMResponse converts Ollama response directly to LLMResponse.
func ollamaChatResponseToLLMResponse(resp *api.ChatResponse, partial bool) *model.LLMResponse {
	content := &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{},
	}

	if resp.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{
			Text: resp.Message.Content,
		})
	}

	// Handle tool calls
	if len(resp.Message.ToolCalls) > 0 {
		for _, tc := range resp.Message.ToolCalls {
			var args map[string]interface{}
			if err := json.Unmarshal(tc.Function.Arguments, &args); err == nil {
				content.Parts = append(content.Parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						Name: tc.Function.Name,
						Args: args,
					},
				})
			}
		}
	}

	llmResp := &model.LLMResponse{
		Content:      content,
		Partial:      partial,
		TurnComplete: resp.Done,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.PromptEvalCount),
			CandidatesTokenCount: int32(resp.EvalCount),
			TotalTokenCount:      int32(resp.PromptEvalCount + resp.EvalCount),
		},
	}

	if resp.Done {
		llmResp.FinishReason = genai.FinishReasonStop
	}

	// Store custom metadata
	llmResp.CustomMetadata = map[string]any{
		"model":                resp.Model,
		"created_at":           resp.CreatedAt,
		"total_duration":       resp.TotalDuration,
		"load_duration":        resp.LoadDuration,
		"prompt_eval_duration": resp.PromptEvalDuration,
		"eval_duration":        resp.EvalDuration,
	}

	return llmResp
}

// convertGenaiToolsToOllama converts genai tools to Ollama tools.
func convertGenaiToolsToOllama(tools []*genai.Tool) []api.Tool {
	ollamaTools := make([]api.Tool, 0, len(tools))

	for _, tool := range tools {
		if tool == nil || tool.FunctionDeclarations == nil {
			continue
		}

		for _, fd := range tool.FunctionDeclarations {
			ollamaTool := api.Tool{
				Type: "function",
				Function: api.ToolFunction{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters: api.ToolFunctionParams{
						Type:       "object",
						Properties: convertSchemaProperties(fd.Parameters),
						// Required:   fd.Parameters.Required,
					},
				},
			}
			ollamaTools = append(ollamaTools, ollamaTool)
		}
	}

	return ollamaTools
}

// convertSchemaProperties converts genai schema properties to Ollama format.
func convertSchemaProperties(schema *genai.Schema) map[string]api.ToolFunctionParams {
	if schema == nil || schema.Properties == nil {
		return nil
	}

	props := make(map[string]api.ToolFunctionParams)
	for key, prop := range schema.Properties {
		props[key] = api.ToolFunctionParams{
			Type:        string(prop.Type),
			Description: prop.Description,
			Enum:        prop.Enum,
		}
	}
	return props
}

// mapRole maps genai role to Ollama role.
func mapRole(role string) string {
	switch role {
	case genai.RoleUser:
		return "user"
	case genai.RoleModel:
		return "assistant"
	case "system":
		return "system"
	case "tool":
		return "tool"
	default:
		return "user"
	}
}

// joinTextParts joins text parts with newlines.
func joinTextParts(parts []string) string {
	result := ""
	for i, part := range parts {
		if i > 0 {
			result += "\n"
		}
		result += part
	}
	return result
}

// boolPtr returns a pointer to the given bool value.
func boolPtr(b bool) *bool {
	return &b
}
