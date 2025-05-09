import React, { useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

const MindMap = () => {
  const [hoveredNode, setHoveredNode] = useState(null);

  // Graph data: Nodes and Edges
  const elements = [
    { data: { id: "Harry", label: "Harry Potter" } },
    { data: { id: "Hermione", label: "Hermione Granger" } },
    { data: { id: "Ron", label: "Ron Weasley" } },
    { data: { id: "Dumbledore", label: "Albus Dumbledore" } },
    {
      data: {
        id: "friendship1",
        source: "Harry",
        target: "Hermione",
        label: "Friends",
      },
    },
    {
      data: {
        id: "friendship2",
        source: "Harry",
        target: "Ron",
        label: "Best Friends",
      },
    },
    {
      data: {
        id: "mentor",
        source: "Dumbledore",
        target: "Harry",
        label: "Mentor",
      },
    },
  ];

  // Cytoscape Styles for Nodes and Edges
  const style = [
    {
      selector: "node",
      style: {
        "background-color": "#0074D9",
        label: "data(label)",
        "text-valign": "center",
        "text-halign": "center",
        color: "#ffffff",
        "font-size": "10px",
        width: "40px",
        height: "40px",
      },
    },
    {
      selector: "edge",
      style: {
        "line-color": "#AAAAAA",
        "target-arrow-color": "#AAAAAA",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        label: "data(label)",
        "font-size": "8px",
        color: "#333333",
        "text-outline-color": "#ffffff",
        "text-outline-width": "1px",
      },
    },
    {
      selector: ":selected",
      style: {
        "background-color": "#FF4136",
        "line-color": "#FF4136",
        "target-arrow-color": "#FF4136",
        "source-arrow-color": "#FF4136",
      },
    },
  ];

  // Handle hover events
  const handleMouseOver = (event) => {
    const node = event.target.data();
    setHoveredNode(node);
  };

  const handleMouseOut = () => {
    setHoveredNode(null);
  };

  return (
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      <div
        style={{ width: "800px", height: "600px", border: "1px solid #ddd" }}
      >
        <CytoscapeComponent
          elements={elements}
          style={{ width: "100%", height: "100%" }}
          stylesheet={style}
          layout={{ name: "circle" }}
          cy={(cy) => {
            // Add event listeners for hover
            cy.on("mouseover", "node", handleMouseOver);
            cy.on("mouseout", "node", handleMouseOut);
          }}
        />
      </div>
      {hoveredNode && (
        <div
          style={{
            marginTop: "10px",
            padding: "10px",
            border: "1px solid #ccc",
            borderRadius: "4px",
            background: "#f9f9f9",
          }}
        >
          <strong>Node Details:</strong>
          <p>ID: {hoveredNode.id}</p>
          <p>Label: {hoveredNode.label}</p>
        </div>
      )}
    </div>
  );
};

export default MindMap;
