# [WIP] SDK Examples

This is under construction scripts. We provide SDK for client to connect with Llama Stack Distros.

# Example SDK Scripts
```
python -m sdk_examples.agentic_system.client devgpu004.eag5.facebook.com 5000
```

You should be able to see stdout of the form ---
```
AgenticSystemCreateResponse(agent_id='83f87db2-7ecf-4c46-8cc7-29bb30631774')
SessionCreateResponse(session_id='522f6220-73fa-4add-adee-d3af8b801ac5')
data: {"event":{"payload":{"event_type":"turn_start","turn_id":"7225d9ef-3eae-4f2e-9da3-f2d92a8fef18"}}}

data: {"event":{"payload":{"event_type":"step_start","step_type":"inference","step_id":"3882ffd2-82bf-4fa1-ace7-d7826cbbac40","metadata":{}}}}

data: {"event":{"payload":{"event_type":"step_complete","step_type":"inference","step_details":{"turn_id":"7225d9ef-3eae-4f2e-9da3-f2d92a8fef18","step_id":"3882ffd2-82bf-4fa1-ace7-d7826cbbac40","started_at":null,"completed_at":null,"step_type":"inference","model_response":{"role":"assistant","content":"The capital of France is Paris.","stop_reason":"end_of_turn","tool_calls":[]}}}}}

data: {"event":{"payload":{"event_type":"turn_complete","turn":{"turn_id":"7225d9ef-3eae-4f2e-9da3-f2d92a8fef18","session_id":"522f6220-73fa-4add-adee-d3af8b801ac5","input_messages":[{"role":"user","content":"What is the capital of France?","context":null}],"steps":[{"turn_id":"7225d9ef-3eae-4f2e-9da3-f2d92a8fef18","step_id":"3882ffd2-82bf-4fa1-ace7-d7826cbbac40","started_at":null,"completed_at":null,"step_type":"inference","model_response":{"role":"assistant","content":"The capital of France is Paris.","stop_reason":"end_of_turn","tool_calls":[]}}],"output_message":{"role":"assistant","content":"The capital of France is Paris.","stop_reason":"end_of_turn","tool_calls":[]},"output_attachments":[],"started_at":"2024-09-09T12:33:26.254952","completed_at":"2024-09-09T12:33:28.642333"}}}}
```
