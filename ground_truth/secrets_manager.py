# ground_truth/secrets_manager.py
"""
Secrets management for LLM Distillery
Supports secrets.ini (local) and environment variables (CI/CD)
"""

import os
import configparser
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manage secrets from multiple sources with fallback priority"""
    
    def __init__(self, secrets_file: Optional[str] = None):
        """
        Initialize secrets manager

        Priority:
        1. Environment variables (CI/CD, local .env)
        2. secrets.ini file (local development)
        """
        # Default location for secrets file
        if secrets_file is None:
            current_path = Path(__file__).resolve()

            # Walk up to find project root (directory with both 'config' and 'ground_truth')
            project_root = None
            for parent in current_path.parents:
                # Check if this is the project root by looking for config and ground_truth
                if (parent / "config").exists() and (parent / "ground_truth").exists():
                    project_root = parent
                    break

            # Fallback if we couldn't find project root
            if project_root is None:
                # Assume we're in ground_truth/, so go up 1 level
                project_root = current_path.parent.parent

            self.secrets_file = project_root / "config" / "credentials" / "secrets.ini"
        else:
            self.secrets_file = Path(secrets_file)
        
        self.secrets: Dict[str, str] = {}
        self._load_secrets()
            
    def _load_secrets(self):
        """Load secrets with priority: Environment Variables > secrets.ini"""
        
        # 1. Try to load from secrets.ini (local development)
        if self.secrets_file.exists():
            self._load_from_ini_file()
            logger.info(f"Loaded secrets from {self.secrets_file}")
        else:
            logger.info(f"No {self.secrets_file} found, using environment variables only")
        
        # 2. Override/supplement with environment variables (GitHub Actions)
        self._load_from_environment()
        
        # 3. Log what we found (without revealing values)
        self._log_available_secrets()
    
    def _load_from_ini_file(self):
        """Load secrets from secrets.ini file"""
        try:
            config = configparser.ConfigParser()
            config.read(self.secrets_file, encoding='utf-8')
            
            # Load API keys section
            if 'api_keys' in config:
                for key, value in config['api_keys'].items():
                    if value and value.strip():
                        normalized_key = f"API_{key.upper()}"
                        self.secrets[normalized_key] = value.strip()
            
            # Load email credentials section
            if 'email_credentials' in config:
                for key, value in config['email_credentials'].items():
                    if value and value.strip():
                        normalized_key = f"EMAIL_{key.upper()}"
                        self.secrets[normalized_key] = value.strip()
            
            # Load notification credentials section
            if 'notification_credentials' in config:
                for key, value in config['notification_credentials'].items():
                    if value and value.strip():
                        normalized_key = key.upper()
                        self.secrets[normalized_key] = value.strip()
        
        except Exception as e:
            logger.warning(f"Error reading {self.secrets_file}: {e}")
    
    def _load_from_environment(self):
        """Load secrets from environment variables"""

        # Environment variable mappings
        env_mappings = {
            # Anthropic Claude
            'ANTHROPIC_API_KEY': 'API_ANTHROPIC_API_KEY',
            'CLAUDE_API_KEY': 'API_ANTHROPIC_API_KEY',

            # Google Gemini
            'GEMINI_API_KEY': 'API_GEMINI_API_KEY',
            'GOOGLE_API_KEY': 'API_GEMINI_API_KEY',

            # OpenAI GPT-4
            'OPENAI_API_KEY': 'API_OPENAI_API_KEY',

            # Weights & Biases (experiment tracking)
            'WANDB_API_KEY': 'API_WANDB_API_KEY',
        }

        for env_var, internal_key in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self.secrets[internal_key] = value
                logger.debug(f"Loaded {internal_key} from environment variable {env_var}")
    
    def _log_available_secrets(self):
        """Log available secrets without revealing values"""
        available = []

        if self.get_anthropic_key():
            available.append("Anthropic Claude")
        if self.get_gemini_key():
            available.append("Google Gemini")
        if self.get_openai_key():
            available.append("OpenAI GPT-4")
        if self.get_wandb_key():
            available.append("Weights & Biases")

        if available:
            logger.info(f"Available credentials: {', '.join(available)}")
        else:
            logger.warning("No LLM API credentials configured")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value by key"""
        return self.secrets.get(key, default)
    
    def has_key(self, key: str) -> bool:
        """Check if a key exists"""
        return key in self.secrets and bool(self.secrets[key])

    # LLM API key methods
    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic Claude API key"""
        return self.get('API_ANTHROPIC_API_KEY')

    def get_gemini_key(self) -> Optional[str]:
        """Get Google Gemini API key (prioritize billing key for higher rate limits)"""
        # Try billing key first (150 RPM)
        billing_key = self.get('API_GEMINI_BILLING_API_KEY')
        if billing_key:
            return billing_key
        # Fall back to regular key (2 RPM free tier)
        return self.get('API_GEMINI_API_KEY')

    def get_openai_key(self) -> Optional[str]:
        """Get OpenAI GPT-4 API key"""
        return self.get('API_OPENAI_API_KEY')

    def get_wandb_key(self) -> Optional[str]:
        """Get Weights & Biases API key"""
        return self.get('API_WANDB_API_KEY')

    def get_llm_key(self, provider: str) -> Optional[str]:
        """
        Get LLM API key for the specified provider

        Args:
            provider: One of 'claude', 'gemini', 'openai'

        Returns:
            API key for the provider, or None if not found

        Raises:
            ValueError: If provider is not recognized
        """
        provider = provider.lower()
        if provider in ('claude', 'anthropic'):
            return self.get_anthropic_key()
        elif provider in ('gemini', 'gemini-flash', 'google'):
            return self.get_gemini_key()
        elif provider in ('openai', 'gpt4', 'gpt-4'):
            return self.get_openai_key()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Use 'claude', 'gemini', 'gemini-flash', or 'openai'")
    
    def is_production(self) -> bool:
        """Check if running in production (CI/CD environment)"""
        return (os.getenv('GITHUB_ACTIONS') == 'true' or
                os.getenv('CI') == 'true' or
                os.getenv('GITLAB_CI') == 'true')

    def get_deployment_context(self) -> Dict[str, any]:
        """Get deployment context information"""
        return {
            'is_production': self.is_production(),
            'is_ci': os.getenv('CI') == 'true',
            'is_github_actions': os.getenv('GITHUB_ACTIONS') == 'true',
            'is_gitlab_ci': os.getenv('GITLAB_CI') == 'true',
            'secrets_source': 'environment' if self.is_production() else 'secrets.ini'
        }


# Global instance
_secrets_manager = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def get_anthropic_key() -> Optional[str]:
    """Get Anthropic Claude API key"""
    return get_secrets_manager().get_anthropic_key()


def get_gemini_key() -> Optional[str]:
    """Get Google Gemini API key"""
    return get_secrets_manager().get_gemini_key()


def get_openai_key() -> Optional[str]:
    """Get OpenAI GPT-4 API key"""
    return get_secrets_manager().get_openai_key()


def get_llm_key(provider: str) -> Optional[str]:
    """Get LLM API key for specified provider"""
    return get_secrets_manager().get_llm_key(provider)


def main():
    """Test secrets manager"""
    print("\n" + "="*60)
    print("LLM Distillery Secrets Manager Test")
    print("="*60)

    secrets = SecretsManager()

    print(f"\nSecrets file: {secrets.secrets_file}")
    print(f"Secrets file exists: {secrets.secrets_file.exists()}")
    print(f"Is production: {secrets.is_production()}")

    print("\n" + "-"*60)
    print("Available Credentials")
    print("-"*60)

    # Test Anthropic Claude
    anthropic_key = secrets.get_anthropic_key()
    print(f"Anthropic Claude: {'[OK] Configured' if anthropic_key else '[MISSING]'}")
    if anthropic_key:
        print(f"  Key preview: {anthropic_key[:8]}...")

    # Test Google Gemini
    gemini_key = secrets.get_gemini_key()
    print(f"Google Gemini: {'[OK] Configured' if gemini_key else '[MISSING]'}")
    if gemini_key:
        print(f"  Key preview: {gemini_key[:8]}...")

    # Test OpenAI GPT-4
    openai_key = secrets.get_openai_key()
    print(f"OpenAI GPT-4: {'[OK] Configured' if openai_key else '[MISSING]'}")
    if openai_key:
        print(f"  Key preview: {openai_key[:8]}...")

    # Test Weights & Biases
    wandb_key = secrets.get_wandb_key()
    print(f"Weights & Biases: {'[OK] Configured' if wandb_key else '[MISSING]'}")
    if wandb_key:
        print(f"  Key preview: {wandb_key[:8]}...")

    print("\n" + "="*60)

    # Deployment context
    context = secrets.get_deployment_context()
    print("\nDeployment Context:")
    for key, value in context.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()