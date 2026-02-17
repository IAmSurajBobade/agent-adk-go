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

// Package api provides types and client for Ollama API.
// This is a temporary implementation that mirrors the github.com/ollama/ollama/api package.
// In production, replace this with the actual Ollama SDK: github.com/ollama/ollama/api
package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Duration wraps time.Duration for JSON marshaling.
type Duration struct {
	Duration time.Duration
}

// Client provides access to the Ollama API.
type Client struct {
	baseURL string
}

// ChatRequest represents a chat completion request.
type ChatRequest struct {
	Model     string
	Messages  []Message
	Stream    *bool
	Options   map[string]interface{}
	KeepAlive *Duration
	Format    string
	Tools     []Tool
}

// ChatResponse represents a chat completion response.
type ChatResponse struct {
	Model              string
	CreatedAt          time.Time
	Message            Message
	Done               bool
	TotalDuration      time.Duration
	LoadDuration       time.Duration
	PromptEvalCount    int
	PromptEvalDuration time.Duration
	EvalCount          int
	EvalDuration       time.Duration
}

// Message represents a chat message.
type Message struct {
	Role      string
	Content   string
	Images    []ImageData
	ToolCalls []ToolCall
}

// ImageData represents image data for multimodal requests.
type ImageData []byte

// ToolCall represents a function/tool call from the model.
type ToolCall struct {
	Function ToolCallFunction
}

// ToolCallFunction represents the function details in a tool call.
type ToolCallFunction struct {
	Name      string
	Arguments json.RawMessage
}

// Tool represents a tool/function definition.
type Tool struct {
	Type     string
	Function ToolFunction
}

// ToolFunction represents a function definition.
type ToolFunction struct {
	Name        string
	Description string
	Parameters  ToolFunctionParams
}

// ToolFunctionParams represents function parameters schema.
type ToolFunctionParams struct {
	Type        string
	Description string
	Enum        []string
	Properties  map[string]ToolFunctionParams
	Required    []string
}

// ClientFromEnvironment creates a client using environment variables.
func ClientFromEnvironment() (*Client, error) {
	return &Client{baseURL: "http://localhost:11434"}, nil
}

// SetBaseURL sets the base URL for the Ollama API.
func (c *Client) SetBaseURL(url string) error {
	c.baseURL = url
	return nil
}

// Chat sends a chat request to the Ollama API.
// The fn callback is called for each response chunk (streaming) or once (non-streaming).
func (c *Client) Chat(ctx context.Context, req *ChatRequest, fn func(ChatResponse) error) error {
	// This is a mock implementation that makes actual HTTP requests.
	// In production, replace this package with github.com/ollama/ollama/api

	url := c.baseURL + "/api/chat"

	reqBody, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	httpResp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", httpResp.StatusCode)
	}

	// Handle streaming vs non-streaming
	if req.Stream != nil && *req.Stream {
		// Streaming: read newline-delimited JSON
		decoder := json.NewDecoder(httpResp.Body)
		for {
			var resp ChatResponse
			if err := decoder.Decode(&resp); err != nil {
				if err == io.EOF {
					break
				}
				return fmt.Errorf("failed to decode response: %w", err)
			}
			if err := fn(resp); err != nil {
				return err
			}
			if resp.Done {
				break
			}
		}
	} else {
		// Non-streaming: read single JSON response
		var resp ChatResponse
		if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
		if err := fn(resp); err != nil {
			return err
		}
	}

	return nil
}
