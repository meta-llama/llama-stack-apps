import React, { useState } from "react";
import axios from "axios";

const QueryMemory = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const handleQuery = async () => {
    if (!query) {
      setResponse("Please enter a query.");
      return;
    }
    try {
      const result = await axios.post("http://localhost:5001/query", { query });
      setResponse(result.data.response);
    } catch (error) {
      console.error(error);
      setResponse("Error querying memory.");
    }
  };

  return (
    <div>
      <h3>Ask a Question</h3>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter your question"
      />
      <button onClick={handleQuery}>Ask</button>
      {response && (
        <p>
          <strong>Response:</strong> {response}
        </p>
      )}
    </div>
  );
};

export default QueryMemory;
