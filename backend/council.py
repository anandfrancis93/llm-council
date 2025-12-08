"""3-stage LLM Council orchestration."""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL

# Overall timeout for the entire council process (in seconds)
# This prevents indefinite hanging even if individual models behave poorly
# 180s = 3 minutes, reasonable for 9 API calls across 3 stages
COUNCIL_TIMEOUT = 180.0

# Per-stage timeouts (models run in parallel within each stage)
STAGE1_TIMEOUT = 60.0  # Initial responses - 4 parallel calls
STAGE2_TIMEOUT = 60.0  # Rankings (longer prompts) - 4 parallel calls
STAGE3_TIMEOUT = 60.0  # Chairman synthesis - 1 call


async def stage1_collect_responses(user_query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (results list, errors list)
        - Results: List of dicts with 'model' and 'response' keys
        - Errors: List of dicts with 'model', 'error_type', and 'message' keys
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel with stage-specific timeout
    responses = await query_models_parallel(COUNCIL_MODELS, messages, timeout=STAGE1_TIMEOUT)

    # Format results and collect errors
    stage1_results = []
    stage1_errors = []
    
    for model, response in responses.items():
        if response.get("success"):
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })
        else:
            # Track the error
            error_info = response.get("error", {"model": model, "error_type": "unknown", "message": "Unknown error"})
            stage1_errors.append(error_info)
            print(f"[Stage 1] {model} failed: {error_info.get('message', 'Unknown error')}")

    return stage1_results, stage1_errors


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping, errors list)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. Evaluate each response individually with bullet points on SEPARATE LINES.
2. End with a final ranking.

STRICT FORMAT - Follow this EXACTLY:

## Response A
- **Strengths:** [what it does well]
- **Weaknesses:** [what it does poorly]

## Response B
- **Strengths:** [what it does well]
- **Weaknesses:** [what it does poorly]

(continue for all responses)

## FINAL RANKING
1. Response X
2. Response Y
3. Response Z

CRITICAL: Each bullet point MUST be on its own line. Do NOT put multiple bullets on the same line.

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel with stage-specific timeout
    responses = await query_models_parallel(COUNCIL_MODELS, messages, timeout=STAGE2_TIMEOUT)

    # Format results and collect errors
    stage2_results = []
    stage2_errors = []
    
    for model, response in responses.items():
        if response.get("success"):
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })
        else:
            # Track the error
            error_info = response.get("error", {"model": model, "error_type": "unknown", "message": "Unknown error"})
            stage2_errors.append(error_info)
            print(f"[Stage 2] {model} failed: {error_info.get('message', 'Unknown error')}")

    return stage2_results, label_to_model, stage2_errors


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Tuple of (result dict, error dict or None)
        - Result: Dict with 'model' and 'response' keys
        - Error: Dict with error info if chairman failed, None otherwise
    """
    from .openrouter import query_model_with_retry
    
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with explicit timeout
    response = await query_model_with_retry(CHAIRMAN_MODEL, messages, timeout=STAGE3_TIMEOUT)

    if not response.get("success"):
        # Return error info
        error_info = response.get("error", {"model": CHAIRMAN_MODEL, "error_type": "unknown", "message": "Unknown error"})
        print(f"[Stage 3] Chairman {CHAIRMAN_MODEL} failed: {error_info.get('message', 'Unknown error')}")
        return {
            "model": CHAIRMAN_MODEL,
            "response": f"⚠️ Chairman failed to synthesize: {error_info.get('message', 'Unknown error')}. Please review Stage 1 responses directly.",
            "error": True
        }, error_info

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }, None


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order (unique, first occurrence only)
    """
    import re

    def get_unique_ordered(matches):
        """Return unique matches preserving first occurrence order."""
        seen = set()
        result = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    # Look for "FINAL RANKING:" or "## FINAL RANKING" section
    ranking_section = None
    for marker in ["## FINAL RANKING", "FINAL RANKING:"]:
        if marker in ranking_text:
            parts = ranking_text.split(marker)
            if len(parts) >= 2:
                ranking_section = parts[1]
                break

    if ranking_section:
        # Only look at the first few lines after the marker (the actual ranking)
        lines = ranking_section.strip().split('\n')[:10]  # Limit to first 10 lines
        ranking_section = '\n'.join(lines)

        # Try to extract numbered list format (e.g., "1. Response A")
        numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
        if numbered_matches:
            labels = [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            return get_unique_ordered(labels)

        # Fallback: Extract all "Response X" patterns in order
        matches = re.findall(r'Response [A-Z]', ranking_section)
        return get_unique_ordered(matches)

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return get_unique_ordered(matches)


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def _run_council_stages(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Internal function that runs all council stages.
    This is wrapped by run_full_council with timeout protection.
    """
    all_errors = []
    
    # Stage 1: Collect individual responses
    stage1_results, stage1_errors = await stage1_collect_responses(user_query)
    all_errors.extend(stage1_errors)

    # If no models responded successfully, return error immediately
    if not stage1_results:
        error_details = ", ".join([f"{e['model']}: {e['message']}" for e in stage1_errors]) if stage1_errors else "Unknown error"
        return [], [], {
            "model": "error",
            "response": f"⚠️ All models failed to respond in Stage 1.\n\n**Errors:**\n{error_details}\n\nPlease try again.",
            "error": True
        }, {"errors": all_errors}

    # Stage 2: Collect rankings
    stage2_results, label_to_model, stage2_errors = await stage2_collect_rankings(user_query, stage1_results)
    all_errors.extend(stage2_errors)

    # Calculate aggregate rankings (only if we have stage2 results)
    aggregate_rankings = []
    if stage2_results:
        aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result, stage3_error = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )
    if stage3_error:
        all_errors.append(stage3_error)

    # Prepare metadata with errors
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "errors": all_errors if all_errors else None,
        "partial_failure": len(all_errors) > 0
    }

    return stage1_results, stage2_results, stage3_result, metadata


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process with timeout protection.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
        Metadata includes 'errors' list if any models failed
    """
    try:
        # Apply overall timeout to prevent indefinite hanging
        return await asyncio.wait_for(
            _run_council_stages(user_query),
            timeout=COUNCIL_TIMEOUT
        )
    except asyncio.TimeoutError:
        # Overall timeout exceeded - return error
        print(f"[Council] Overall timeout of {COUNCIL_TIMEOUT}s exceeded!")
        return [], [], {
            "model": "error",
            "response": f"⚠️ The council deliberation timed out after {int(COUNCIL_TIMEOUT)} seconds.\n\nThis usually happens when one or more models are experiencing high load. Please try again.",
            "error": True
        }, {
            "errors": [{
                "model": "council",
                "error_type": "timeout",
                "message": f"Overall council timeout ({COUNCIL_TIMEOUT}s) exceeded"
            }]
        }
