import os
import json
import asyncio
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

# Flask setup
app = Flask(__name__)
CORS(app)

# Memory initialization
active_sessions = {}


@app.route("/initialize", methods=["POST"])
def initialize_memory():
    """
    Initialize memory for a new book. Queries LlamaStack for characters and relationships,
    and sets up memory for the user to interact with.
    """
    data = request.json
    book_title = data.get("title")
    if not book_title:
        return jsonify({"error": "Book title is required"}), 400

    # Clear the previous session if any
    active_sessions.clear()

    # Create a new memory agent and process the book
    response = asyncio.run(process_book(book_title))
    return jsonify(response), 200


async def async_generator_wrapper(sync_gen):
    for item in sync_gen:
        yield item


async def get_graph_response(text_response, client):
    # Create prompt for graph conversion
    graph_prompt = f"""
    Convert this character description into a graph format with nodes and links.
    Format the response as a JSON object with "nodes" and "links" arrays.
    Each node should have "id", "name", and "val" properties.
    Each link should have "source" and "target" properties using the node ids.
    Set all node "val" to 10.

    Text to convert:
    {text_response}

    Expected format example (return only the JSON object, no additional text):
    {{
        "nodes": [
            {{"id": id1, "name": "Character Name", "val": 10}}
        ],
        "links": [
            {{"source": id1, "target": "id2"}}
        ]
    }}
    id1 and id2 are example variables and you should not generate those exact values.
    """

    # Get graph structure from LlamaStack
    graph_response = client.inference.chat_completion(
        model_id=os.environ["INFERENCE_MODEL"],
        messages=[
            {"role": "system", "content": "You are a data structure expert. Convert text descriptions into graph JSON format."},
            {"role": "user", "content": graph_prompt}
        ]
    )
    return graph_response


def jsonify_graph_response(response):
    """Extract and parse JSON content from graph response."""
    try:
        content = response.completion_message.content
        print("content: ", content)
        # Find indices of first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No valid JSON object found in response")

        # Extract JSON string
        json_str = content[start_idx:end_idx + 1]

        # Parse JSON
        return json.loads(json_str)

    except Exception as e:
        logging.error(f"Error parsing graph response: {e}")
        return None


async def process_book(book_title):
    """
    Process the book title, query LlamaStack for characters and relationships,
    and initialize memory.
    """
    client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")
    agent_config = AgentConfig(
        model=os.environ["INFERENCE_MODEL"],
        instructions="You are a helpful assistant",
        tools=[{"type": "memory"}],  # Enable memory
        enable_session_persistence=True,
        max_infer_iters=5,
    )

    # Create the agent and session
    agent = Agent(client, agent_config)
    session_id = agent.create_session(f"{book_title}-session")
    active_sessions["agent"] = agent
    active_sessions["session_id"] = session_id
    logging.info(f"Created session_id={session_id} for book: {book_title}")

    # Query LlamaStack for characters and relationships
    prompt = f"Who are the characters in the book '{book_title}', and what are their relationships?"

    response = client.inference.chat_completion(
        model_id=os.environ["INFERENCE_MODEL"],
        messages=[
            {"role": "system", "content": "You are a knowledgeable book expert. Provide detailed information about characters and their relationships in the book."},
            {"role": "user", "content": prompt}
        ]
    )
    text_response = response.completion_message.content

    file_name = f"{book_title.replace(' ', '_').lower()}_memory.txt"
    with open(file_name, "w") as f:
        f.write(text_response)

    graph_response = await get_graph_response(text_response, client)

    print("graph_response: ", graph_response)

    graph_data = ""
    try:
        graph_data = jsonify_graph_response(graph_response)
        logging.info("Graph data generated:", json.dumps(graph_data, indent=2))
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing graph response: {e}")

    # Push to memory agent (optional if further memory setup is needed)
    memory_prompt = "Save this knowledge about the book into memory for future questions."
    memory_response = agent.create_turn(
        messages=[{"role": "user", "content": memory_prompt}],
        attachments=[
            {"content": text_response, "mime_type": "text/plain"}
        ],
        session_id=session_id,
    )

    async for log in async_generator_wrapper(EventLogger().log(memory_response)):
        log.print()

    return graph_data


def convert_to_graph_format(entities, relationships):
    """
    Converts entities and relationships into a graph dictionary format.
    """
    nodes = []
    links = []
    node_id_map = {}
    node_counter = 1

    # Add entities as nodes
    for entity in entities:
        name = entity["text"]
        if name not in node_id_map:
            node_id_map[name] = f"id{node_counter}"
            nodes.append({"id": f"id{node_counter}", "name": name, "val": 10})
            node_counter += 1

    # Add relationships as links
    for relationship in relationships.split("\n"):  # Assuming relationships are line-separated
        parts = relationship.split(" and ")
        if len(parts) == 2:
            source_name = parts[0].strip()
            target_name = parts[1].split(" are")[0].strip()

            if source_name in node_id_map and target_name in node_id_map:
                links.append({
                    "source": node_id_map[source_name],
                    "target": node_id_map[target_name]
                })

    return {"nodes": nodes, "links": links}


def convert_to_graph_format_old(response):
    """
    Converts LlamaStack's text response into a graph dictionary format.
    """
    nodes = []
    links = []

    # Simplified parsing logic (replace with actual NLP or regex parsing)
    lines = [line for line in response if line.strip()]
    node_id_map = {}
    node_counter = 1

    for line in lines:
        if " and " in line:  # Assumes relationships are described as "X and Y are friends"
            parts = line.split(" and ")
            if len(parts) == 2:
                source_name = parts[0].strip()
                target_name = parts[1].split(" are")[0].strip()

                # Add nodes if they don't already exist
                for name in [source_name, target_name]:
                    if name not in node_id_map:
                        node_id_map[name] = f"id{node_counter}"
                        nodes.append({"id": f"id{node_counter}", "name": name, "val": 10})
                        node_counter += 1

                # Add the relationship as a link
                links.append({
                    "source": node_id_map[source_name],
                    "target": node_id_map[target_name]
                })

    return {"nodes": nodes, "links": links}


@app.route("/query", methods=["POST"])
def query_memory():
    """
    Handles user queries and returns answers based on memory.
    """
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    # Query the memory agent
    response = asyncio.run(query_llama_stack(query))
    return jsonify({"response": response})


async def query_llama_stack(prompt):
    """
    Queries the active LlamaStack session with a user prompt.
    """
    if "agent" not in active_sessions:
        return "No active agent session. Please initialize a book first."

    agent = active_sessions["agent"]
    session_id = active_sessions["session_id"]

    response = agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )

    # Process response logs
    result = []
    async for log in async_generator_wrapper(EventLogger().log(response)):
        result.append(str(log))

    inference_start_idx = result.index("inference> ")
    inference_logs = result[inference_start_idx + 1:]

    return "\n".join(inference_logs)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
