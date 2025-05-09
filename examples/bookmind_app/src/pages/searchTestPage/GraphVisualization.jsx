import React from "react";
import CytoscapeComponent from "react-cytoscapejs";

const GraphVisualization = ({ graphData }) => {
  const elements = [
    ...graphData.nodes.map((node) => ({
      data: { id: node.id, label: node.name },
    })),
    ...graphData.links.map((link) => ({
      data: { source: link.source, target: link.target },
    })),
  ];

  const layout = { name: "circle" };

  const style = [
    {
      selector: "node",
      style: {
        "background-color": "#0074D9",
        label: "data(label)",
        "text-valign": "center",
        "text-halign": "center",
        color: "#fff",
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
      },
    },
  ];

  return (
    <div>
      <h3>Graph Visualization</h3>
      <CytoscapeComponent
        elements={elements}
        style={{ width: "800px", height: "600px" }}
        layout={layout}
        stylesheet={style}
      />
    </div>
  );
};

export default GraphVisualization;
