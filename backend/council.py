"""3-stage LLM Council orchestration."""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL

# NO TIMEOUTS - wait forever until we get a response
COUNCIL_TIMEOUT = None
STAGE1_TIMEOUT = None
STAGE2_TIMEOUT = None
STAGE3_TIMEOUT = None


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


def _get_best_stage1_response(
    stage1_results: List[Dict[str, Any]], 
    aggregate_rankings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get the best Stage 1 response based on rankings, or first response as fallback.
    """
    if aggregate_rankings and stage1_results:
        # Get the top-ranked model
        best_model = aggregate_rankings[0]["model"]
        for result in stage1_results:
            if result["model"] == best_model:
                return {
                    "model": best_model,
                    "response": f"*[Chairman unavailable - showing top-ranked response from {best_model.split('/')[-1]}]*\n\n{result['response']}",
                    "fallback": True
                }
    
    # Fallback to first response
    if stage1_results:
        first = stage1_results[0]
        return {
            "model": first["model"],
            "response": f"*[Chairman unavailable - showing response from {first['model'].split('/')[-1]}]*\n\n{first['response']}",
            "fallback": True
        }
    
    return {
        "model": "error",
        "response": "No responses available.",
        "error": True
    }


async def _run_council_stages(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Internal function that runs all council stages.
    Designed to ALWAYS return something useful - never fails completely.
    """
    all_errors = []
    
    # Stage 1: Collect individual responses
    print("[Council] Starting Stage 1...")
    stage1_results, stage1_errors = await stage1_collect_responses(user_query)
    all_errors.extend(stage1_errors)
    print(f"[Council] Stage 1 complete: {len(stage1_results)} responses, {len(stage1_errors)} errors")

    # If no models responded successfully, we still need to return something
    if not stage1_results:
        error_details = ", ".join([f"{e['model']}: {e['message']}" for e in stage1_errors]) if stage1_errors else "Unknown error"
        return [], [], {
            "model": "error",
            "response": f"⚠️ All models failed to respond in Stage 1.\n\n**Errors:**\n{error_details}\n\nPlease try again.",
            "error": True
        }, {"errors": all_errors}

    # Stage 2: Collect rankings (optional - proceed even if all fail)
    print("[Council] Starting Stage 2...")
    stage2_results, label_to_model, stage2_errors = await stage2_collect_rankings(user_query, stage1_results)
    all_errors.extend(stage2_errors)
    print(f"[Council] Stage 2 complete: {len(stage2_results)} rankings, {len(stage2_errors)} errors")

    # Calculate aggregate rankings (only if we have stage2 results)
    aggregate_rankings = []
    if stage2_results:
        aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    print("[Council] Starting Stage 3 (Chairman)...")
    stage3_result, stage3_error = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )
    
    # If Chairman failed, use the best Stage 1 response as fallback
    if stage3_error:
        all_errors.append(stage3_error)
        print(f"[Council] Chairman failed, using fallback from Stage 1")
        stage3_result = _get_best_stage1_response(stage1_results, aggregate_rankings)
    else:
        print("[Council] Stage 3 complete")

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
    Run the complete 3-stage council process.
    
    GUARANTEE: This function will ALWAYS return a usable response.
    - If Chairman fails, returns best-ranked Stage 1 response
    - If Stage 2 fails, still returns Chairman or first Stage 1 response
    - Only fails if ALL Stage 1 models fail
    
    NO OVERALL TIMEOUT - we wait as long as needed. User paid for these API calls.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
        Metadata includes 'errors' list if any models failed
    """
    # No overall timeout - just run the stages
    # Each stage has its own per-model timeouts
    return await _run_council_stages(user_query)

