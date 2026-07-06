import os
from rich import print as rprint
from llm_sdk import Small_LLM_Model

def debug_prompt(generated:str) -> None:
  """Print constructed prompt"""
  content_start_idx = generated.index('"prompt":')
  if content_start_idx != -1:
    rprint(f"Generated: [bold green]{generated[content_start_idx:]}[/bold green]")

def debug_decoded_candidates(context: str, candidates_tokens: list[int], model: Small_LLM_Model) -> None:
  """Print first candidate tokens"""
  if os.getenv("DEBUG") == "True":
    decoded_candidates_top_5 = [model.decode([tok]) for tok in candidates_tokens][:6]
    rprint(f"authorized tokens ids for {context}: {candidates_tokens}")
    rprint(f"decoded tokens for {context}[bold cyan]:", decoded_candidates_top_5)

def debug_title(name:str) -> None:
  """Print with special title format"""
  if os.getenv("DEBUG") == "True":
    rprint(f"\n[bold yellow on black] === { name.upper() } === [/bold yellow on black]\n")
