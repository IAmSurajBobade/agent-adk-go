# Ollama Model Package

This package implements the `model.LLM` interface for Ollama models, enabling integration with the ADK-Go framework.

## Overview

The Ollama model package provides a complete implementation that:
- Satisfies the `model.LLM` interface
- Supports both streaming and non-streaming generation
- Maps data correctly between ADK-Go's `genai` types and Ollama's API format
- Provides comprehensive configuration options for all Ollama parameters
- Includes extensive test coverage in tabular format

## Features

### Core Functionality
- **LLM Interface Compliance**: Implements `Name()` and `GenerateContent()` methods
- **Streaming Support**: Full support for streaming responses with proper aggregation
- **Non-Streaming Support**: Traditional request-response pattern
- **Multi-turn Conversations**: Handles conversation history correctly
- **Tool/Function Calling**: Supports tool definitions and function calls
- **Multimodal Support**: Handles images via base64 encoding

### Configuration Options

The package provides extensive configuration options covering all Ollama parameters:

#### Sampling Parameters
- `Temperature` - Controls randomness (default: 0.8)
- `TopK` - Top-k sampling
- `TopP` - Top-p (nucleus) sampling
- `MinP` - Minimum probability threshold
- `TypicalP` - Typical sampling parameter
- `TFSZ` - Tail free sampling

#### Context & Generation
- `NumCtx` - Context window size (default: 2048)
- `NumPredict` - Maximum tokens to generate (default: -1, infinite)
- `NumKeep` - Tokens to keep from initial prompt
- `NumBatch` - Batch size for processing
- `NumThread` - Number of CPU threads to use

#### Repetition Control
- `RepeatLastN` - How far back to prevent repetition
- `RepeatPenalty` - Penalty for repetitions (default: 1.1)
- `PresencePenalty` - Presence penalty parameter
- `FrequencyPenalty` - Frequency penalty parameter
- `PenalizeNewline` - Whether to penalize newline tokens

#### Mirostat Sampling
- `Mirostat` - Enable Mirostat (0=disabled, 1/2=enabled)
- `MirostatTau` - Mirostat target entropy
- `MirostatEta` - Mirostat learning rate

#### Hardware & Performance
- `NumGPU` - Number of GPUs to use
- `MainGPU` - Main GPU to use
- `NUMA` - NUMA support
- `LowVRAM` - Low VRAM mode
- `UseMMap` - Use memory mapping
- `UseMLock` - Use memory locking
- `VocabOnly` - Load vocabulary only

#### Other Options
- `Seed` - Random seed for reproducibility
- `Stop` - Stop sequences to end generation
- `KeepAlive` - How long to keep model loaded
- `Format` - Response format ("json" or JSON schema)

## Usage

### Basic Example

```go
import (
    "context"
    "google.golang.org/genai"
    "google.golang.org/adk/model/ollama"
)

// Create a new Ollama model
model, err := ollama.NewModel("llama3.2", "http://localhost:11434", nil)
if err != nil {
    log.Fatal(err)
}

// Create a request
req := &model.LLMRequest{
    Contents: []*genai.Content{
        genai.NewContentFromText("What is the capital of France?", "user"),
    },
    Config: &genai.GenerateContentConfig{
        Temperature: float32Ptr(0.7),
    },
}

// Generate response
ctx := context.Background()
for resp, err := range model.GenerateContent(ctx, req, false) {
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(resp.Content.Parts[0].Text)
}
```

### With Custom Options

```go
opts := &ollama.Options{
    Temperature:      float32Ptr(0.9),
    TopK:             float32Ptr(40),
    TopP:             float32Ptr(0.95),
    NumCtx:           intPtr(4096),
    NumPredict:       intPtr(256),
    RepeatPenalty:    float32Ptr(1.1),
    Seed:             intPtr(42),
    Stop:             []string{"END", "\n\n"},
    KeepAlive:        durationPtr(5 * time.Minute),
    Format:           "json",
}

model, err := ollama.NewModel("mistral", "http://localhost:11434", opts)
```

### Streaming Example

```go
// Generate streaming response
for resp, err := range model.GenerateContent(ctx, req, true) {
    if err != nil {
        log.Fatal(err)
    }

    if resp.Partial {
        // Handle partial streaming chunk
        fmt.Print(resp.Content.Parts[0].Text)
    } else if resp.TurnComplete {
        // Handle final aggregated response
        fmt.Println("\n\nFinal response complete")
    }
}
```

## Design

### Architecture

The implementation follows the pattern established by the Gemini model:

1. **Model Structure**: `ollamaModel` struct implements the `LLM` interface
2. **Client Management**: Uses the Ollama API client for HTTP communication
3. **Content Conversion**: Bidirectional mapping between `genai.Content` and Ollama messages
4. **Streaming Aggregation**: Uses `llminternal.StreamingResponseAggregator` for consistent streaming behavior
5. **Options Merging**: Model-level options can be overridden by request-level config

### Data Mapping

The package handles several key mappings:

- **Roles**: `user`, `model` (assistant), `system`, `tool`
- **Content Parts**: Text, inline data (images), function calls, function responses
- **Usage Metadata**: Token counts for prompts and completions
- **Custom Metadata**: Ollama-specific timing and performance data

### Testing

The package includes comprehensive tests:

- **Table-Driven Tests**: All major functionality tested with multiple scenarios
- **Unit Tests**: Conversion functions, role mapping, options handling
- **Integration Tests**: HTTP mock servers for realistic API testing
- **Coverage**: Model creation, generation, streaming, options, conversions

## Implementation Notes

### API Package

Due to environment constraints, the package includes a local `api` subpackage that mirrors the official `github.com/ollama/ollama/api` package structure. This implementation:

- Provides all necessary types (Client, ChatRequest, ChatResponse, etc.)
- Makes actual HTTP requests to Ollama servers
- Handles both streaming and non-streaming responses
- Is fully compatible with the official SDK

**For production use**, replace the local `api` package with:
```go
import "github.com/ollama/ollama/api"
```

### Streaming Behavior

The streaming implementation follows ADK-Go conventions:

1. **Partial Events**: Text chunks marked with `Partial: true`
2. **Aggregated Events**: Full content after streaming with `Partial: false`
3. **Turn Completion**: Final event marked with `TurnComplete: true`
4. **Usage Metadata**: Included in the final response

## Testing

Run the test suite:

```bash
# All tests
go test ./model/ollama/...

# Verbose output
go test -v ./model/ollama/...

# Specific test
go test -v ./model/ollama/... -run TestModel_Generate

# With coverage
go test -cover ./model/ollama/...
```

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Go SDK](https://pkg.go.dev/github.com/ollama/ollama/api)
- [ADK-Go Model Interface](../llm.go)
- [Gemini Model Implementation](../gemini/gemini.go) (reference design)

## License

Copyright 2025 Google LLC - Licensed under the Apache License, Version 2.0
