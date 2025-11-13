import tiktoken

class BaseAgent:
    """
    Base class for all agents providing common utility methods.
    """
    
    def __init__(self):
        """Initialize token counter for debugging."""
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return 0
    
    def _print_llm_input(self, call_name: str, system_prompt: str, user_message: str):
        """Print LLM input details for debugging (production mode: minimal output)."""
        # Production mode: Only show call name and token count
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_message)
        total_tokens = system_tokens + user_tokens
        
        print(f"🤖 {call_name} | Tokens: {total_tokens} (system: {system_tokens}, user: {user_tokens})")
        
        # Uncomment below for detailed debugging:
        # print("\n" + "="*80)
        # print(f"🤖 LLM CALL: {call_name}")
        # print("="*80)
        # print(f"\n📝 SYSTEM PROMPT ({len(system_prompt)} chars):")
        # print("-" * 80)
        # print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        # print(f"\n💬 USER MESSAGE ({len(user_message)} chars):")
        # print("-" * 80)
        # print(user_message)
        # print("="*80 + "\n")
