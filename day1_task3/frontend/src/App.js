import React, { useState } from 'react';
import Editor from '@monaco-editor/react';
import './App.css';

function App() {
  const [code, setCode] = useState(`def calculate_sum(a, b):
    result = a + b
    return result`);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleReview = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Review failed');
      }
      
      const data = await response.json();
      setFeedback(data);
    } catch (error) {
      setError('Error: ' + error.message);
      console.error('Fetch error:', error);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🔍 Self-Reflecting Code Review Agent</h1>
        <p>Submit Python code for intelligent iterative feedback</p>
      </header>
      
      <div className="container">
        <div className="editor-section">
          <h2>Python Code</h2>
          <div className="editor-container">
            <Editor
              height="350px"
              language="python"
              value={code}
              onChange={setCode}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                fontSize: 14,
              }}
            />
          </div>
          <button onClick={handleReview} disabled={loading} className="review-btn">
            {loading ? '⏳ Analyzing...' : '✨ Review Code'}
          </button>
        </div>

        {error && (
          <div className="error-box">
            <strong>⚠️ Error:</strong> {error}
          </div>
        )}

        {feedback && (
          <div className="feedback-section">
            {/* AST Issues Section */}
            <div className="feedback-card ast-issues">
              <h2>📋 AST Issues</h2>
              {feedback.ast_issues && feedback.ast_issues.length > 0 ? (
                <ul>
                  {feedback.ast_issues.map((issue, idx) => (
                    <li key={idx} className={`issue issue-${issue.type}`}>
                      <span className="category">[{issue.category}]</span>
                      <span className="message">{issue.message}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="no-issues">✅ No AST issues detected</p>
              )}
            </div>

            {/* Round 1 Suggestions */}
            <div className="feedback-card round1">
              <h2>💡 Round 1: Initial Suggestions</h2>
              <div className="suggestion-text">
                {feedback.round_1_suggestions ? (
                  <pre>{feedback.round_1_suggestions}</pre>
                ) : (
                  <p>No suggestions available</p>
                )}
              </div>
            </div>

            {/* Round 2 Improved Suggestions */}
            <div className="feedback-card round2">
              <h2>🎯 Round 2: Improved Suggestions</h2>
              <div className="suggestion-text">
                {feedback.round_2_suggestions ? (
                  <pre>{feedback.round_2_suggestions}</pre>
                ) : (
                  <p>No improvements available</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;