import { Component, type ErrorInfo, type ReactNode } from 'react';
import { ErrorBanner } from './ErrorBanner';

export interface WorkspaceErrorBoundaryProps {
  workspaceName: string;
  children: ReactNode;
}

interface WorkspaceErrorBoundaryState {
  error: Error | null;
  componentStack: string | null;
}

/**
 * Rule 5: Fail Fast & Surface Authentic Errors.
 * Prevents an unexpected payload shape or rendering error from unmounting the entire shell
 * into a blank screen, preserving the full underlying stack trace directly in the UI.
 */
export class WorkspaceErrorBoundary extends Component<
  WorkspaceErrorBoundaryProps,
  WorkspaceErrorBoundaryState
> {
  public override state: WorkspaceErrorBoundaryState = {
    error: null,
    componentStack: null,
  };

  public static getDerivedStateFromError(error: Error): WorkspaceErrorBoundaryState {
    return { error, componentStack: null };
  }

  public override componentDidCatch(error: Error, info: ErrorInfo): void {
    this.setState({ error, componentStack: info.componentStack ?? null });
  }

  public override render(): ReactNode {
    if (this.state.error) {
      const fullError = new Error(
        `${this.state.error.message}${this.state.error.stack ? `\n\n${this.state.error.stack}` : ''}${
          this.state.componentStack ? `\n\nComponent Stack:${this.state.componentStack}` : ''
        }`,
      );
      return (
        <div style={{ padding: '18px' }}>
          <ErrorBanner
            error={fullError}
            title={`Workspace Rendering Error (${this.props.workspaceName})`}
            onRetry={() => this.setState({ error: null, componentStack: null })}
          />
        </div>
      );
    }

    return this.props.children;
  }
}
