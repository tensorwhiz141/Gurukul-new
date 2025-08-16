import React from "react";
import { useParams } from "react-router-dom";

const ResourcePage = () => {
  const { resourceId } = useParams();
  return (
    <div style={{ padding: 40, textAlign: "center" }}>
      <h2>Resource Details</h2>
      <p>Resource ID: {resourceId}</p>
    </div>
  );
};

export default ResourcePage;
