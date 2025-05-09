import React, { useState } from "react";
import axios from "axios";

const InitializeMemory = ({ setGraphData }) => {
  const [bookTitle, setBookTitle] = useState("");
  const [message, setMessage] = useState("");

  const handleInitialize = async () => {
    if (!bookTitle) {
      setMessage("Please enter a book title.");
      return;
    }
    try {
      const response = await axios.post("http://localhost:5001/initialize", {
        title: bookTitle,
      });
      setGraphData(response.data);
      setMessage("Memory initialized successfully!");
    } catch (error) {
      console.error(error);
      setMessage("Error initializing memory.");
    }
  };

  return (
    <div>
      <h3>Initialize Memory</h3>
      <input
        type="text"
        value={bookTitle}
        onChange={(e) => setBookTitle(e.target.value)}
        placeholder="Enter book title"
      />
      <button onClick={handleInitialize}>Initialize</button>
      {message && <p>{message}</p>}
    </div>
  );
};

export default InitializeMemory;
