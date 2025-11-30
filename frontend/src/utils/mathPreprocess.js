/**
 * Preprocesses markdown text to normalize math delimiters for KaTeX.
 * Converts various formats to standard $...$ and $$...$$ delimiters.
 */
export function preprocessMath(text) {
  if (!text) return text;

  let result = text;

  // Convert \[ ... \] to $$ ... $$ (display math)
  result = result.replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$');

  // Convert \( ... \) to $ ... $ (inline math)
  result = result.replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');

  // Convert [ ... ] on its own line to $$ ... $$ (display math)
  // Only match if it contains LaTeX-like content (backslashes, ^, _, etc.)
  result = result.replace(/^\s*\[\s*((?:[^[\]]*(?:\\[a-zA-Z]+|[\^_{}])[^[\]]*)+)\s*\]\s*$/gm, '$$$$$1$$$$');

  // Convert inline ( ... ) that contains LaTeX commands to $ ... $
  // Match ( ... ) containing backslash commands like \frac, \mid, etc.
  result = result.replace(/\(\s*((?:[^()]*\\[a-zA-Z]+[^()]*)+)\s*\)/g, '$$$1$$');

  return result;
}
