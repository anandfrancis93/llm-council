import { useState } from 'react';
import './ErrorBanner.css';

/**
 * Displays errors from failed model queries.
 * Shows a summary by default, expandable to see all errors.
 */
export default function ErrorBanner({ errors }) {
    const [expanded, setExpanded] = useState(false);

    if (!errors || errors.length === 0) return null;

    // Group errors by type
    const timeoutCount = errors.filter(e => e.error_type === 'timeout').length;
    const apiErrorCount = errors.filter(e => e.error_type === 'api_error').length;
    const otherCount = errors.length - timeoutCount - apiErrorCount;

    const getSummary = () => {
        const parts = [];
        if (timeoutCount > 0) parts.push(`${timeoutCount} timeout${timeoutCount > 1 ? 's' : ''}`);
        if (apiErrorCount > 0) parts.push(`${apiErrorCount} API error${apiErrorCount > 1 ? 's' : ''}`);
        if (otherCount > 0) parts.push(`${otherCount} other error${otherCount > 1 ? 's' : ''}`);
        return parts.join(', ');
    };

    const getErrorIcon = (type) => {
        switch (type) {
            case 'timeout': return '⏱️';
            case 'rate_limit': return '🚫';
            case 'api_error': return '❌';
            case 'network': return '🔌';
            default: return '⚠️';
        }
    };

    return (
        <div className="error-banner">
            <div className="error-summary" onClick={() => setExpanded(!expanded)}>
                <span className="error-icon">⚠️</span>
                <span className="error-text">
                    Some models failed: {getSummary()}
                </span>
                <button className="expand-button">
                    {expanded ? '▲' : '▼'}
                </button>
            </div>

            {expanded && (
                <div className="error-details">
                    {errors.map((error, index) => (
                        <div key={index} className="error-item">
                            <span className="error-type-icon">{getErrorIcon(error.error_type)}</span>
                            <span className="error-model">{error.model?.split('/').pop() || 'Unknown'}</span>
                            <span className="error-message">{error.message}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
