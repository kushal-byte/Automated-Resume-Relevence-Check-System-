# llm_analysis/langsmith_logger.py - LangSmith Observability & Debugging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

# Note: LangSmith requires API key for full functionality
# For hackathon demo, we'll create a local logging system that mimics LangSmith

class LangSmithLogger:
    """LangSmith-style logging and observability for LLM chains"""
    
    def __init__(self, project_name="resume-relevance-system"):
        self.project_name = project_name
        self.session_id = str(uuid.uuid4())
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize log files
        self.trace_log = f"{self.logs_dir}/langsmith_traces.jsonl"
        self.metrics_log = f"{self.logs_dir}/langsmith_metrics.jsonl"
        
        print(f"✅ LangSmith Logger initialized - Project: {project_name}")
        print(f"📊 Session ID: {self.session_id}")
    
    def start_trace(self, trace_name: str, inputs: Dict[str, Any]) -> str:
        """Start a new trace for an LLM chain"""
        trace_id = str(uuid.uuid4())
        
        trace_start = {
            "trace_id": trace_id,
            "session_id": self.session_id,
            "project_name": self.project_name,
            "trace_name": trace_name,
            "start_time": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "status": "started",
            "type": "trace_start"
        }
        
        self._log_event(trace_start, self.trace_log)
        print(f"🔍 LangSmith: Started trace '{trace_name}' - ID: {trace_id[:8]}...")
        return trace_id
    
    def end_trace(self, trace_id: str, outputs: Dict[str, Any], 
                  status: str = "success", error: Optional[str] = None,
                  token_usage: Optional[Dict] = None):
        """End a trace with results"""
        
        trace_end = {
            "trace_id": trace_id,
            "session_id": self.session_id,
            "end_time": datetime.utcnow().isoformat(),
            "outputs": outputs,
            "status": status,
            "error": error,
            "token_usage": token_usage or {},
            "type": "trace_end"
        }
        
        self._log_event(trace_end, self.trace_log)
        status_emoji = "✅" if status == "success" else "❌"
        print(f"{status_emoji} LangSmith: Ended trace {trace_id[:8]}... - Status: {status}")
    
    def log_llm_call(self, trace_id: str, step_name: str, 
                     prompt: str, response: str, model: str,
                     latency_ms: float, token_usage: Optional[Dict] = None):
        """Log an individual LLM call within a trace"""
        
        llm_call = {
            "trace_id": trace_id,
            "step_name": step_name,
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,  # Truncate long prompts
            "response": response[:500] + "..." if len(response) > 500 else response,
            "latency_ms": latency_ms,
            "token_usage": token_usage or {},
            "type": "llm_call"
        }
        
        self._log_event(llm_call, self.trace_log)
        print(f"🤖 LangSmith: LLM call logged - {step_name} ({latency_ms:.1f}ms)")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        
        metric_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "type": "metrics"
        }
        
        self._log_event(metric_entry, self.metrics_log)
        print(f"📊 LangSmith: Metrics logged - {list(metrics.keys())}")
    
    def log_evaluation(self, trace_id: str, evaluation_results: Dict[str, Any]):
        """Log evaluation results for testing and debugging"""
        
        evaluation = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "evaluation_results": evaluation_results,
            "type": "evaluation"
        }
        
        self._log_event(evaluation, self.trace_log)
        print(f"🧪 LangSmith: Evaluation logged for trace {trace_id[:8]}...")
    
    def _log_event(self, event: Dict[str, Any], log_file: str):
        """Write event to log file"""
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"⚠️ LangSmith: Failed to write log - {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        try:
            traces = []
            metrics = []
            
            # Read trace logs
            if os.path.exists(self.trace_log):
                with open(self.trace_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line.strip())
                            if event.get("session_id") == self.session_id:
                                if event.get("type") == "trace_start":
                                    traces.append(event)
            
            # Read metrics logs  
            if os.path.exists(self.metrics_log):
                with open(self.metrics_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line.strip())
                            if event.get("session_id") == self.session_id:
                                metrics.append(event)
            
            return {
                "session_id": self.session_id,
                "project_name": self.project_name,
                "total_traces": len(traces),
                "total_metrics": len(metrics),
                "traces": traces[-5:],  # Last 5 traces
                "metrics": metrics[-5:]  # Last 5 metrics
            }
            
        except Exception as e:
            print(f"⚠️ LangSmith: Failed to get session summary - {e}")
            return {"error": str(e)}
    
    def export_session_data(self, filename: Optional[str] = None) -> str:
        """Export session data for analysis"""
        if not filename:
            filename = f"{self.logs_dir}/session_{self.session_id[:8]}_export.json"
        
        summary = self.get_session_summary()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📁 LangSmith: Session data exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ LangSmith: Export failed - {e}")
            return ""

# Global logger instance
logger = LangSmithLogger()

def trace_llm_analysis(func):
    """Decorator to trace LLM analysis functions"""
    def wrapper(*args, **kwargs):
        # Start trace
        trace_id = logger.start_trace(
            func.__name__,
            {"args_count": len(args), "kwargs": list(kwargs.keys())}
        )
        
        start_time = datetime.utcnow()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            latency = (end_time - start_time).total_seconds() * 1000
            
            # End trace
            logger.end_trace(
                trace_id,
                {"result_type": type(result).__name__},
                "success"
            )
            
            # Log metrics
            logger.log_metrics({
                "function": func.__name__,
                "latency_ms": latency,
                "success": True
            })
            
            return result
            
        except Exception as e:
            # Log error
            logger.end_trace(
                trace_id,
                {},
                "error", 
                str(e)
            )
            
            logger.log_metrics({
                "function": func.__name__, 
                "success": False,
                "error": str(e)
            })
            
            raise e
    
    return wrapper

# Test function
def test_langsmith_logging():
    """Test LangSmith logging functionality"""
    
    # Test trace
    trace_id = logger.start_trace("test_analysis", {"test": True})
    
    logger.log_llm_call(
        trace_id,
        "test_llm_call",
        "Test prompt",
        "Test response", 
        "grok-4-fast",
        150.5,
        {"tokens": 100}
    )
    
    logger.end_trace(trace_id, {"test_result": "success"}, "success")
    
    # Test metrics
    logger.log_metrics({
        "test_metric": 95.5,
        "accuracy": 0.85
    })
    
    # Get summary
    summary = logger.get_session_summary()
    print(f"✅ LangSmith test completed - {summary['total_traces']} traces logged")
    
    return summary['total_traces'] > 0

if __name__ == "__main__":
    test_langsmith_logging()
