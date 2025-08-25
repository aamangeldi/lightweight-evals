"""Tests for model adapters."""

import os
from unittest.mock import Mock, patch

from lightweight_evals.adapters.dummy import DummyAdapter
from lightweight_evals.adapters.openai import OpenAIAdapter


class TestDummyAdapter:
    """Tests for DummyAdapter."""

    def test_init(self):
        """Test DummyAdapter initialization."""
        adapter = DummyAdapter(seed=42)
        assert adapter.name == "dummy"
        assert adapter.version == "1.0"
        assert adapter.seed == 42

    def test_generate_deterministic(self):
        """Test that DummyAdapter generates deterministic responses."""
        adapter1 = DummyAdapter(seed=42)
        adapter2 = DummyAdapter(seed=42)

        prompt = "Test prompt"
        response1 = adapter1.generate(prompt)
        response2 = adapter2.generate(prompt)

        assert response1 == response2
        assert isinstance(response1, str)
        assert len(response1) > 0

    def test_generate_different_prompts(self):
        """Test that different prompts can generate different responses."""
        adapter = DummyAdapter(seed=42)

        response1 = adapter.generate("First prompt")
        response2 = adapter.generate("Second prompt")

        # Different prompts might generate different responses
        # (though not guaranteed due to hash collision)
        assert isinstance(response1, str)
        assert isinstance(response2, str)

    def test_generate_with_parameters(self):
        """Test that generate method accepts all expected parameters."""
        adapter = DummyAdapter()

        response = adapter.generate("Test prompt", max_tokens=100, temperature=0.5)

        assert isinstance(response, str)
        assert len(response) > 0


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    def test_init_with_api_key(self):
        """Test OpenAIAdapter initialization with API key."""
        adapter = OpenAIAdapter(model="gpt-3.5-turbo", api_key="test-key")
        assert adapter.name == "openai"
        assert adapter.version == "1.0"
        assert adapter.model == "gpt-3.5-turbo"

    def test_init_with_env_api_key(self):
        """Test OpenAIAdapter initialization with environment API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
            adapter = OpenAIAdapter()
            assert adapter.model == "gpt-4o-mini"  # default model

    @patch("lightweight_evals.adapters.openai.OpenAI")
    def test_generate_success(self, mock_openai_class):
        """Test successful response generation."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter(api_key="test-key")
        response = adapter.generate("Test prompt")

        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("lightweight_evals.adapters.openai.OpenAI")
    def test_generate_with_parameters(self, mock_openai_class):
        """Test generate method with custom parameters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter(api_key="test-key")
        adapter.generate("Test prompt", max_tokens=100, temperature=0.8)

        # Verify the call was made with correct parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["messages"][0]["content"] == "Test prompt"

    @patch("lightweight_evals.adapters.openai.OpenAI")
    def test_generate_error_handling(self, mock_openai_class):
        """Test error handling in generate method."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock an exception
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        adapter = OpenAIAdapter(api_key="test-key")
        response = adapter.generate("Test prompt")

        assert "Error generating response" in response
        assert "API Error" in response

    @patch("lightweight_evals.adapters.openai.OpenAI")
    def test_generate_none_content(self, mock_openai_class):
        """Test handling of None content in response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter(api_key="test-key")
        response = adapter.generate("Test prompt")

        assert response == ""
