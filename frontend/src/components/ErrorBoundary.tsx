import { Component, type ErrorInfo, type ReactNode } from "react";

// A page that fails to draw says so, with its error, instead of leaving the screen blank; the
// sidebar stays, and another page (or a reload) starts afresh.

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ resetKey: string; children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(previous: { resetKey: string }) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("A page failed to draw:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="page">
        <div className="callout error" role="alert">
          <div>
            <strong>This page failed to draw.</strong>
            <div className="callout-body">
              {this.state.error.message}. Reload the page; if it happens again, the backend may be
              older than the page: restart it.
            </div>
          </div>
        </div>
      </div>
    );
  }
}
