import React, { useState } from "react";
import InitializeMemory from "./InitializeMemory";
import QueryMemory from "./QueryMemory";
import GraphVisualization from "./GraphVisualization";

const BookSearch = () => {
  const [graphData, setGraphData] = useState(null);

  return (
    <div>
      <h1>Book Memory Agent</h1>
      <InitializeMemory setGraphData={setGraphData} />
      <hr />
      {graphData && <GraphVisualization graphData={graphData} />}
      <hr />
      <QueryMemory />
    </div>
  );
};

export default BookSearch;
